from __future__ import annotations
import json
import cv2
import os.path as osp
from typing import OrderedDict
import pathlib
from functools import reduce
from pathlib import Path
from mmcv.utils import print_log
import mmcv
from mmdet.core import mean_iou
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import ImageTransform, BboxTransform, SegTransform, Numpy2Tensor
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation
import zipfile
import io
from PIL import Image
import glob
import os
from collections import OrderedDict


def pre_load_files(file_dir: str, file_prefix: str):
    with os.scandir(path=osp.join(file_dir, file_prefix)) as it:
        files = [osp.basename(entry.path) for entry in it]
    return files


def load_img(dirname: str, path: str, seg=False):
    if seg == True:
        img = Image.open(osp.join(dirname, path)).convert("L")
    else:
        img = Image.open(osp.join(dirname, path)).convert("RGB")
    img = np.array(img, dtype=np.float32)

    return img


def load_json(dirname: str, path: str):
    json_dict = json.load(open(osp.join(dirname, path), "r"))
    return json_dict


class Hybrid(Dataset):

    CLASSES = (
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    )
    CLASSES_SEG = ("road", "lane")

    def __init__(
        self,
        img_dir: str,
        img_prefix: str,
        seg_prefix: dict,
        det_prefix: dict,
        original_scale: tuple,
        img_scale: tuple,
        img_norm_cfg: dict,
        size_divisor: int,
        with_label=True,
        with_seg=True,
        flip_ratio=0,
        resize_keep_ratio=True,
        test_mode=False,
    ):
        self.ignore_index = 0
        self.shape = original_scale
        self.img_transform = ImageTransform(size_divisor=size_divisor, **img_norm_cfg)
        self.test_mode = test_mode
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio
        self.img_scales = img_scale if isinstance(img_scale, list) else [img_scale]
        self.bbox_transform = BboxTransform()
        self.seg_transform = SegTransform()
        self.with_crowd = None
        self.with_label = with_label
        self.with_seg = with_seg
        self.img_dir = img_dir
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.det_prefix = det_prefix
        self.db: list[str] = self._get_db()
        if not self.test_mode:
            self._set_group_flag()

    def __len__(
        self,
    ):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def _get_db(self):
        """
        TODO: add docs
        """
        print("building database...")
        gt_db = []

        images = pre_load_files(self.img_dir, self.img_prefix)
        dets = []
        for zipfilename, prefix in self.det_prefix.items():
            dets.append(pre_load_files(zipfilename, prefix))
        segs = []
        for zipfilename, prefix in self.seg_prefix.items():
            segs.append(pre_load_files(zipfilename, prefix))
        # テストが完了したら戻す
        images = images[:]
        for name in images:
            for det in dets:
                exist = name.replace(".jpg", ".json") in set(det)
                if not exist:
                    break
            if not exist:
                continue
            for seg in segs:
                exist = name.replace(".jpg", ".png") in set(seg)
                if not exist:
                    break
            if not exist:
                continue
            gt_db.append(name)
        print("database build finish")
        return gt_db

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)

        data = self.prepare_train_img(idx)
        return data

    def prepare_train_img(self, idx: int):
        img, ann = self.get_ann_info(idx)
        gt_bboxes = ann["bboxes"]
        gt_labels = ann["labels"]
        seg = ann["seg"]
        gt_bboxes_ignore = ann["bboxes_ignore"]
        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # choose scales
        img_scale = random_scale(self.img_scales)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio
        )
        img = img.copy()
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor, flip)
        if self.with_seg:
            seg = self.seg_transform(seg, pad_shape, scale_factor, flip)

        ori_shape = self.shape
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
        )
        gt_detection = dict()
        gt_detection["gt_bboxes"] = DC(to_tensor(gt_bboxes))
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_detection=gt_detection,
        )
        # ToDo02
        gt_detection["gt_bboxes_ignore"] = gt_bboxes_ignore
        if self.with_label:
            gt_detection["gt_labels"] = DC(to_tensor(gt_labels))
        if self.with_seg:
            data["gt_semantic_seg"] = DC(to_tensor(seg).long().unsqueeze(0), stack=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        name = self.db[idx]
        # load image
        path = osp.join(self.img_prefix, name)
        img = load_img(self.img_dir, path)

        def prepare_single(img, scale, flip):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio
            )
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=self.shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip,
            )
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta = prepare_single(img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def get_ann_info(self, idx):
        name = self.db[idx]
        # load image
        path = osp.join(self.img_prefix, name)
        img = load_img(self.img_dir, path)
        # ToDo01:enable to use proposal file
        seg_list = []
        for dirfilename, prefix in self.seg_prefix.items():
            path = osp.join(prefix, name.replace("jpg", "png"))
            seg_list.append(load_img(dirfilename, path, seg=True))
        json_list = []
        for dirfilename, prefix in self.det_prefix.items():
            path = osp.join(prefix, name.replace("jpg", "json"))
            json_list.append(load_json(dirfilename, path)["frames"][0]["objects"])
        seg = np.where((seg_list[1] == 0), seg_list[0], seg_list[1])
        # いらない
        seg = np.where(seg == 127, 1, seg)
        seg = np.where(seg == 191, 1, seg)
        seg = np.where(seg == 255, 0, seg)
        # いつか撤去

        # ToDo02:enable to use bboxes_ignore
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        gt_bboxes = []
        gt_labels = []
        for j in json_list:
            for idx in range(len(j)):
                if not any(i == j[idx]["category"] for i in self.CLASSES):
                    continue
                if j[idx].get("box2d") == None:
                    continue
                pre_box = [
                    j[idx].get("box2d")["x1"],
                    j[idx].get("box2d")["x2"],
                    j[idx].get("box2d")["y1"],
                    j[idx].get("box2d")["y2"],
                ]
                box = self.convert(self.shape, pre_box)
                gt_bboxes.append(box)
                gt_labels.append(int(self.CLASSES.index(j[idx]["category"])))
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            seg=seg,
            bboxes_ignore=gt_bboxes_ignore,
        )
        return img, ann

    def convert(self, size, box):
        dw = 1.0 / (size[0])
        dh = 1.0 / (size[1])
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [x, y, w, h]

    def evaluate(self, result, gt_seg_maps, metric="mIoU", logger=None):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["mIoU"]
        if metric not in allowed_metrics:
            raise KeyError("metric {} is not supported".format(metric))

        eval_results = {}

        if self.CLASSES_SEG is None:
            num_seg_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps])
            )
        else:
            num_seg_classes = len(self.CLASSES_SEG)
        all_acc, acc, iou = mean_iou(
            result, gt_seg_maps, num_seg_classes, ignore_index=self.ignore_index
        )
        summary_str = ""
        summary_str += "per class results:\n"

        line_format = "{:<15} {:>10} {:>10}\n"
        summary_str += line_format.format("Class", "IoU", "Acc")
        if self.CLASSES_SEG is None:
            class_names = tuple(range(num_seg_classes))
        else:
            class_names = self.CLASSES_SEG
        for i in range(num_seg_classes):
            iou_str = "{:.2f}".format(iou[i] * 100)
            acc_str = "{:.2f}".format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str)
        summary_str += "Summary:\n"
        line_format = "{:<15} {:>10} {:>10} {:>10}\n"
        summary_str += line_format.format("Scope", "mIoU", "mAcc", "aAcc")

        iou_str = "{:.2f}".format(np.nanmean(iou) * 100)
        acc_str = "{:.2f}".format(np.nanmean(acc) * 100)
        all_acc_str = "{:.2f}".format(all_acc * 100)
        summary_str += line_format.format("global", iou_str, acc_str, all_acc_str)
        print_log(summary_str, logger)

        eval_results["mIoU"] = np.nanmean(iou)
        eval_results["mAcc"] = np.nanmean(acc)
        eval_results["aAcc"] = all_acc

        return eval_results
