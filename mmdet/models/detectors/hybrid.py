from .new_stage import NewStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class HybridHr(NewStageDetector):
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
        super(HybridHr, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            seg_head=seg_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            freezeBackBone=freezeBackBone,
            freezeDet=freezeDet,
            freezeSeg=freezeSeg,
        )
