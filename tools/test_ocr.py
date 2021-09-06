#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import replace_ImageToTensor

from mmocr.apis.inference import disable_text_recog_aug_test, model_inference
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils import revert_sync_batchnorm


def imdenormalize_torch(img, mean, std, to_bgr=True):
    device = img.device
    img = torch.transpose(img, 1, 3)
    img = torch.transpose(img, 1, 2)
    batch_size = img.shape[0]
    mean = torch.from_numpy(mean.reshape(
        (1, 1, 1, -1))).to(device).expand(batch_size, -1, -1, -1)
    std = torch.from_numpy(std.reshape(
        (1, 1, 1, -1))).to(device).expand(batch_size, -1, -1, -1)
    img = img * std + mean
    if to_bgr:
        img = img[:, :, :, (2, 1, 0)]
    return img


def resize_boundary_to_score_map(boundary, scale_factor):
    """Rescale boundaries via scale_factor.

    Args:
        boundaries (list[list[float]]): The boundary list. Each boundary
        with size 2k+1 with k>=4.
        scale_factor(ndarray): The scale factor of size (4,).

    Returns:
        boundaries (list[list[float]]): The scaled boundaries.
    """

    boundary_len = len(boundary)
    boundary[:boundary_len -
             1] = (np.array(boundary[:boundary_len - 1]) *
                   (np.tile(scale_factor[:2], int(
                       (boundary_len - 1) / 2)).reshape(
                           1, boundary_len - 1))).flatten().tolist()
    return boundary


class OcrPipline(nn.Module):
    """ocr end to end pipline for test.

    Args:
        text_det_config (mmcv.Config): dbnet config.
        text_det_ckpt_path (str): dbnet model ckpt path
        text_recog_config (mmcv.Config): crnn config.
        text_recog_ckpt_path (str): crnn model ckpt path
    Returns:
        list[dict]: e2e_results
        [
            {
                "filename": "img_xxx.jpg"
                "polygons":
                    [{
                        "polygon": [159, 82, 488, 428 ...],
                        "det_score":"0.620622",
                        "text_label":"horse123",
                        "recog_score": [0.3, 0.5, ...]}
                    ],
            }
        ]
    """

    def __init__(self, text_det_config, text_det_ckpt_path, text_recog_config,
                 text_recog_ckpt_path):
        super().__init__()
        # build the model and load checkpoint
        text_det_config.model.train_cfg = None
        self.dbnet_model = build_detector(
            text_det_config.model, test_cfg=text_det_config.get('test_cfg'))
        self.dbnet_model = revert_sync_batchnorm(self.dbnet_model)
        fp16_cfg = text_det_config.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.dbnet_model)
        load_checkpoint(
            self.dbnet_model, text_det_ckpt_path, map_location='cpu')
        text_recog_config.model.train_cfg = None

        self.crnn_model = build_detector(
            text_recog_config.model,
            test_cfg=text_recog_config.get('test_cfg'))
        self.crnn_model = revert_sync_batchnorm(self.crnn_model)
        fp16_cfg = text_recog_config.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.crnn_model)
        load_checkpoint(
            self.crnn_model, text_recog_ckpt_path, map_location='cpu')
        self.crnn_model.cfg = text_recog_config

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        end2end_res = []
        output = self.dbnet_model(img, img_metas, return_loss, **kwargs)
        bboxes_list = [res['boundary_result'] for res in output]
        for dc_index, img_meta in enumerate(img_metas):
            batch_imgs = imdenormalize_torch(
                img[dc_index], img_meta[0]['img_norm_cfg']['mean'],
                img_meta[0]['img_norm_cfg']['std'])
            for info, bboxes, single_img in zip(img_meta, bboxes_list,
                                                batch_imgs):

                img_e2e_res = {}
                img_e2e_res['filename'] = info['filename']
                img_e2e_res['polygons'] = []
                for bbox in bboxes:
                    box_res = {}
                    score_map_bbox = resize_boundary_to_score_map(
                        bbox[:], info['scale_factor'])
                    box_res['polygon'] = [round(x) for x in bbox[:-1]]
                    box_res['det_score'] = float(bbox[-1])
                    box = score_map_bbox[:8]
                    if len(score_map_bbox) > 9:
                        min_x = min(score_map_bbox[0:-1:2])
                        min_y = min(score_map_bbox[1:-1:2])
                        max_x = max(score_map_bbox[0:-1:2])
                        max_y = max(score_map_bbox[1:-1:2])
                        box = [
                            min_x, min_y, max_x, min_y, max_x, max_y, min_x,
                            max_y
                        ]
                    box_img = crop_img(single_img.squeeze(),
                                       box).cpu().numpy().astype(np.uint8)
                    if box_img.shape[0] * box_img.shape[1] < 16:
                        recog_result = {'text': '', 'score': -1}
                    else:
                        recog_result = model_inference(self.crnn_model,
                                                       box_img)
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, list):
                        text_score = sum(text_score) / max(1, len(text))
                    box_res['text_label'] = text
                    box_res['recog_score'] = text_score
                    img_e2e_res['polygons'].append(box_res)
                end2end_res.append(img_e2e_res)
        return end2end_res


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMOCR test (and eval) a model.')
    parser.add_argument(
        'text_det_config', help='Test text det config file path.')
    parser.add_argument(
        'text_recog_config', help='Test text recog config file path.')
    parser.add_argument(
        'text_det_checkpoint', help='Text det checkpoint file.')
    parser.add_argument(
        'text_recog_checkpoint', help='Text recog checkpoint file.')
    parser.add_argument('--out', help='Output result file in pickle format.')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed.')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without performing evaluation. It is'
        'useful when you want to format the results to a specific format and '
        'submit them to the test server.')
    parser.add_argument(
        '--eval',
        type=str,
        default='hmean-iou',
        nargs='+',
        help='The evaluation metrics, which depends on the dataset, e.g.,'
        '"bbox", "seg", "proposal" for COCO, and "mAP", "recall" for'
        'PASCAL VOC.')
    parser.add_argument('--show', action='store_true', help='Show results.')
    parser.add_argument(
        '--show-dir', help='Directory where the output images will be saved.')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='Score threshold (default: 0.3).')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='Whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='The tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into the config file. If the value '
        'to be overwritten is a list, it should be of the form of either '
        'key="[a,b]" or key=a,b. The argument also allows nested list/tuple '
        'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
        'are necessary and that no white space is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='Custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options.')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert (
        args.out or args.eval or args.format_only or args.show
        or args.show_dir), (
            'Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir".')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified.')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    text_det_config = Config.fromfile(args.text_det_config)
    text_recog_config = Config.fromfile(args.text_recog_config)
    for cfg in [text_det_config, text_recog_config]:
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if cfg.model.get('pretrained'):
            cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            samples_per_gpu = (cfg.data.get('test_dataloader', {})).get(
                'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
            if samples_per_gpu > 1:
                # Support batch_size > 1 in test for text recognition
                # by disable MultiRotateAugOCR
                # since it is useless for most case
                cfg = disable_text_recog_aug_test(cfg)
                if cfg.data.test.get('pipeline', None) is not None:
                    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                    cfg.data.test.pipeline = replace_ImageToTensor(
                        cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **text_det_config.dist_params)
    text_det_config.data.test.type = 'E2eIcdarDataset'
    # build the dataloader
    dataset = build_dataset(text_det_config.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=text_det_config.get('seed'),
            drop_last=False,
            dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               #    pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'workers_per_gpu',
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {}),
        **dict(samples_per_gpu=samples_per_gpu)
    }

    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = OcrPipline(text_det_config, args.text_det_checkpoint,
                       text_recog_config, args.text_recog_checkpoint)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        eval_results, dataset_merge_results = dataset.evaluate(outputs)
        return eval_results, dataset_merge_results


if __name__ == '__main__':
    eval_results, dataset_merge_results = main()
    print(eval_results)
