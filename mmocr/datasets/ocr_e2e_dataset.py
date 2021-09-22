import numpy as np
from mmdet.datasets.builder import DATASETS

import mmocr.utils as utils
from mmocr.core.evaluation.cleval import eval_hmean
from mmocr.datasets.icdar_dataset import IcdarDataset


@DATASETS.register_module()
class E2eIcdarDataset(IcdarDataset):

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []
        gt_texts = []
        gt_ignore_texts = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing
                gt_ignore_texts.append('###')
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_texts.append(ann.get('text_label'))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            gt_texts=gt_texts,
            gt_ignore_texts=gt_ignore_texts)

        return ann

    def evaluate(self, results):
        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            eval_results (dict[str: float]): The evaluation results.
            dataset_merge_results (list[dict]):
                {
                    "filename": "img_xxx.jpg",
                    //[polygon + score] i.e.[x1, y1, ... xn, yn, score]
                    "pred_det": list[list[float]],
                    //[polygon] i.e.[x1, y1, ... xn, yn]
                    "gt_det": list[list[float]],
                    "pred_text": list[str],
                    "gt_text": list[str],
                    "hmean": 0.65,
                    "recall": 0.7,
                    "precision": 0.61
                }
        """
        assert utils.is_type_list(results, dict)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))

        eval_results, dataset_merge_results = eval_hmean(
            results, img_infos, ann_infos)

        return eval_results, dataset_merge_results
