# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import LabelData

from mmocr.structures import (TextDetDataSample, TextRecogDataSample,
                              TextSpottingDataSample)


def det_to_spotting(det_data_samples: TextDetDataSample,
                    spotting_data_samples: TextSpottingDataSample):
    """Convert detection data sample to spotting data sample.

    Args:
        det_data_sample (dict): Detection data sample.
        spotting_datga_sample (dict): Spotting data sample.

    Returns:
        dict: Spotting data sample.
    """
    for det_data_sample, spotting_data_sample in zip(det_data_samples,
                                                     spotting_data_samples):
        spotting_data_sample.pred_instances = det_data_sample.pred_instances
    return spotting_data_samples


def spotting_to_det(spotting_data_sample: TextSpottingDataSample):
    """Convert spotting data sample to detection data sample."""

    return spotting_data_sample


def merge_recog2spotting(recog_data_samples,
                         spotting_data_samples: TextSpottingDataSample):
    """Convert recognition data sample to spotting data sample.

    Args:
        recog_data_sample (dict): Recognition data sample.
        spotting_datga_sample (dict): Spotting data sample.

    Returns:
        dict: Spotting data sample.
    """
    texts = [ds.pred_text.item for ds in recog_data_samples]
    start = 0
    for spotting_data_sample in spotting_data_samples:
        end = start + len(spotting_data_sample.pred_instances)
        spotting_data_sample.pred_instances.texts = texts[start:end]
        start = end
    return spotting_data_samples


def instance_data2recog(instance_datas, training=False):
    """Convert instance data to recognition data.

    Args:
        instance_data (dict): Instance data.

    Returns:
        dict: Recognition data.
    """
    ds_list = []
    if training:
        for instance_data in instance_datas:
            texts = instance_data.texts
            for text in texts:
                data_sample = TextRecogDataSample()
                gt_text_data = dict(item=text)
                gt_text = LabelData(**gt_text_data)
                data_sample.gt_text = gt_text
                ds_list.append(data_sample)
    else:
        len_instance = sum(
            [len(instance_data) for instance_data in instance_datas])
        ds_list = [TextRecogDataSample() for _ in range(len_instance)]
    return ds_list
