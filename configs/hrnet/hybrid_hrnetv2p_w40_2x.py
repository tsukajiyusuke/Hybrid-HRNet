# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="HybridHr",
    freezeBackBone=None,
    freezeDet=None,
    freezeSeg=True,
    backbone=dict(
        type="HighResolutionNet",
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
                fuse_method="SUM",
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(18, 36),
                fuse_method="SUM",
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
                fuse_method="SUM",
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                fuse_method="SUM",
            ),
        ),
    ),
    neck=dict(type="HRFPN", in_channels=[18, 36, 72, 144], out_channels=256),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True,
    ),
    bbox_roi_extractor=dict(
        type="SingleRoIExtractor",
        roi_layer=dict(type="RoIAlign", out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    bbox_head=dict(
        type="SharedFCBBoxHead",
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=10,
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
    ),
    seg_head=dict(
        type="FCNHead",
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=sum([18, 36, 72, 144]),
        input_transform="resize_concat",
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        drop_out_ratio=-1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler",
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False,
        ),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False,
    ),
    rcnn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler",
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True,
        ),
        pos_weight=-1,
        debug=False,
    ),
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0,
    ),
    rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# NOTE:
# dataset settings
# if you use zip format to store all images of coco, please use CocoZipDataset
dataset_type = "Hybrid"
data_root = "/content/drive/MyDrive/3年/画像・映像コンテンツ演習3/"
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False)
# else
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_dir=data_root + "bdd100k_images_100k",
        img_prefix="bdd100k/images/100k/train/",
        seg_prefix={
            data_root + "da_seg_annotations": "bdd_seg_gt/train/",
            data_root + "ll_seg_annotations": "bdd_lane_gt/train/",
        },
        det_prefix={
            data_root + "det_annotations": "data2/zwt/bdd/bdd100k/labels/100k/train/"
        },
        with_label=True,
        with_seg=True,
        original_scale=(720, 1280, 3),
        img_scale=(640, 384),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
    ),
    val=dict(
        type=dataset_type,
        img_dir=data_root + "bdd100k_images_100k",
        img_prefix="bdd100k/images/100k/train/",
        seg_prefix={
            data_root + "da_seg_annotations": "bdd_seg_gt/train/",
            data_root + "ll_seg_annotations": "bdd_lane_gt/train/",
        },
        det_prefix={
            data_root + "det_annotations": "data2/zwt/bdd/bdd100k/labels/100k/train/"
        },
        with_label=True,
        with_seg=True,
        original_scale=(720, 1280, 3),
        img_scale=(640, 384),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
    ),
    test=dict(
        type=dataset_type,
        img_dir=data_root + "bdd100k_images_100k",
        img_prefix="bdd100k/images/100k/train/",
        seg_prefix={
            data_root + "da_seg_annotations": "bdd_seg_gt/train/",
            data_root + "ll_seg_annotations": "bdd_lane_gt/train/",
        },
        det_prefix={
            data_root + "det_annotations": "data2/zwt/bdd/bdd100k/labels/100k/train/"
        },
        with_label=True,
        with_seg=True,
        original_scale=(720, 1280, 3),
        img_scale=(640, 384),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        test_mode=True,
    ),
)
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 23],
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "./work_dirs/faster_rcnn_hrnetv2p_w40_2x"
load_from = None
resume_from = None
workflow = [("train", 1)]
