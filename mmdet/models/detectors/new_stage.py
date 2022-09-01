from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.ops import resize
import numpy as np


@DETECTORS.register_module
class NewStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin, MaskTestMixin):
    def __init__(
        self,
        backbone,
        neck=None,
        rpn_head=None,
        bbox_roi_extractor=None,
        bbox_head=None,
        seg_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        freezeBackBone=None,
        freezeDet=None,
        freezeSeg=None,
    ):
        super(NewStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if seg_head is not None:
            self.seg_head = builder.build_head(seg_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        if freezeBackBone:

            backbone.requires_grad = False
            neck.requires_grad = False
            print("[Info] freezed backbone")

        if freezeDet:

            bbox_head.requires_grad = False
            rpn_head.requires_grad = False
            bbox_roi_extractor.requires_grad = False
            print("[Info] freezed detection head")

        if freezeSeg:

            seg_head.requires_grad = False
            print("[Info] freezed segmentation head")

    @property
    def with_rpn(self):
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(NewStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_seg:
            self.seg_head.init_weights()

    def extract_feat(self, img):
        x0 = self.backbone(img)
        if self.with_neck:
            x = self.neck(x0)
        return x0, x

    def forward_train(
        self,
        img,
        img_meta,
        gt_detection: dict[dict],
        gt_semantic_seg: list,
        proposals=None,
    ):
        x0, x = self.extract_feat(img)

        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            gt_bboxes = gt_detection["gt_bboxes"]
            gt_labels = gt_detection["gt_labels"]
            gt_bboxes_ignore = gt_detection["gt_bboxes_ignore"]

            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            self.detection(
                x, proposal_list, img, gt_labels, gt_bboxes, losses, gt_bboxes_ignore
            )
        else:
            gt_bboxes = gt_detection["gt_bboxes"]
            gt_labels = gt_detection["gt_labels"]
            gt_bboxes_ignore = gt_detection["gt_bboxes_ignore"]
            proposal_list = proposals
            self.detection(
                x, proposal_list, img, gt_labels, gt_bboxes, losses, gt_bboxes_ignore
            )

        # seg head forward and loss
        if self.with_seg:
            seg_logits = self.seg_head.forward_train(x0)
            loss_seg = self.seg_head.losses(seg_logits, gt_semantic_seg)
            losses.update(loss_seg)
        return losses

    def detection(
        self, x, proposal_list, img, gt_labels, gt_bboxes, losses, gt_bboxes_ignore
    ):
        # assign gts and sample proposals
        if self.with_bbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[: self.bbox_roi_extractor.num_inputs], rois
            )
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn
            )
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x0, x = self.extract_feat(img)
        out = self.seg_head.forward_test(x0)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=self.seg_head.align_corners,
        )
        return out

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]["img_shape"][:2]
                seg_logit = seg_logit[:, :, : resize_shape[0], : resize_shape[1]]
                size = img_meta[0]["ori_shape"][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode="bilinear",
                align_corners=self.seg_head.align_corners,
                warning=False,
            )

        return seg_logit

    def inference(self, img, x, img_meta, rescale):
        """Inference with slide/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            Tensor: The output segmentation map.
        """

        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)

        seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]["flip"]
        if flip:
            output = output.flip(dims=(3,))

        return output

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x0, x = self.extract_feat(img)

        proposal_list = (
            self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
            if proposals is None
            else proposals
        )

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale
        )
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)

        if not self.with_seg:
            return bbox_results
        else:
            seg_logit = self.inference(img, x0, img_meta, rescale)
            seg_pred = seg_logit.argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                return seg_pred
            seg_pred = seg_pred.cpu().numpy()
            seg_pred = np.squeeze(seg_pred)
            # unravel batch dim
            return bbox_results, seg_pred

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn
        )
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list, self.test_cfg.rcnn
        )

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]["scale_factor"]
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_seg:
            seg_logit = self.inference(
                imgs[0], self.extract_feat(imgs[0])[1], img_metas[0], rescale
            )
            for i in range(1, len(imgs)):
                cur_seg_logit = self.inference(
                    imgs[i], self.extract_feat(imgs[i])[1], img_metas[i], rescale
                )
                seg_logit += cur_seg_logit
            seg_logit /= len(imgs)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return bbox_results, seg_pred
        else:
            return bbox_results

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)[1]
