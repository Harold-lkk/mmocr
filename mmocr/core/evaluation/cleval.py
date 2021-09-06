# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from shapely.geometry import Point

import mmocr.utils.check_argument as check
from mmocr.core.evaluation import utils as eval_utils


def eval_hmean(results, img_infos, ann_infos):
    """calculate single image use cleval.

    Args:
        results (list[list[list[int]]]): Ground truth poly.
        img_infos (list[str]): contain test img names.
        ann_infos (list[dict]): Each dict contains annotation
            infos of one image, containing following keys:
            masks, masks_ignore, gt_texts, gt_ignore_texts
    Returns:
        img_char_info (dict): single img of gt_num pred_num
                                recall_num precision_num
        img_result (dict): single img of recall precision h-mean
    """
    dataset_gt_num = 0
    dataset_pred_num = 0
    dataset_hit_recall_num = 0.0
    dataset_hit_precision_num = 0.0
    gt_dataset_boxes, gts_dataset_ignore, gt_dataset_texts, gt_ignore_texts = \
        extract_gt_info(ann_infos)
    pred_dataset_polys, pred_dataset_scores, pred_dataset_texts = \
        extract_pred_info(results)
    img_num = len(pred_dataset_polys)
    assert img_num == len(gt_dataset_boxes)
    assert img_num == len(gts_dataset_ignore)
    single_img_metrics = []
    for i in range(img_num):
        gt_boxes = gt_dataset_boxes[i]
        gt_ignored = gts_dataset_ignore[i]
        gt_texts = gt_dataset_texts[i]
        det_boxes = pred_dataset_polys[i]
        det_texts = pred_dataset_texts[i]
        img_char_info, single_img_metric = single_eval_cleval(
            gt_boxes, gt_ignored, gt_texts, det_boxes, det_texts)
        dataset_gt_num += img_char_info['num_gts']
        dataset_pred_num += img_char_info['num_dets']
        dataset_hit_recall_num += img_char_info['num_recall']
        dataset_hit_precision_num += img_char_info['num_precision']
        single_img_metrics.append(single_img_metric)
    total_r, total_p, total_h = eval_utils.compute_hmean(
        dataset_hit_recall_num, dataset_hit_precision_num, dataset_gt_num,
        dataset_pred_num)

    dataset_metric = {
        'num_gts': dataset_gt_num,
        'num_dets': dataset_pred_num,
        'num_recall': dataset_hit_recall_num,
        'num_precision': dataset_hit_precision_num,
        'recall': total_r,
        'precision': total_p,
        'hmean': total_h
    }
    dataset_merge_results = merge_all_info(img_infos, gt_dataset_boxes,
                                           gts_dataset_ignore,
                                           gt_dataset_texts, gt_ignore_texts,
                                           pred_dataset_polys,
                                           pred_dataset_texts,
                                           pred_dataset_scores,
                                           single_img_metrics)
    return dataset_metric, dataset_merge_results


def single_eval_cleval(gt_boxes, gt_ignored, gt_texts, det_boxes, det_texts):
    """calculate single image use cleval.

    Args:
        gt_boxes (list[list[list[int]]]): Ground truth poly.
        gt_ignored (list[list[list[int]]]): Ignored ground truth poly.
        gt_texts (list[str]): Ground truth texts. same length with gt_masks
        det_boxes (list[list[list[int]]]): predict poly.
        det_texts (list[str]): predict texts. same length with det_boxes
    Returns:
        img_char_info (dict): single img of gt_num pred_num
                                recall_num precision_num
        img_result (dict): single img of recall precision h-mean
    """
    gt_num = len(gt_boxes)
    gt_all = gt_boxes + gt_ignored
    # TODO
    gt_texts_with_ignore = gt_texts + [
        '#' * 10 for i in range(len(gt_ignored))
    ]
    gt_polys = [eval_utils.points2polygon(p) for p in gt_all]
    gt_dont_care_indices = [gt_num + i for i in range(len(gt_ignored))]
    gt_num = len(gt_polys)
    det_polys = [eval_utils.points2polygon(p) for p in det_boxes]
    area_precision_matrix = compute_area_precision(gt_polys, det_polys)
    gt_pcc_points = compute_pcc_matrix(gt_all, gt_texts_with_ignore)
    pcc_count_matrix = compute_pcc_inclusion(gt_polys, det_polys,
                                             gt_pcc_points)
    det_dont_care_indices = filter_det_dont_care(det_boxes,
                                                 area_precision_matrix,
                                                 pcc_count_matrix,
                                                 gt_dont_care_indices)
    match_matrix = calc_match_matrix(gt_all, det_boxes, gt_dont_care_indices,
                                     det_dont_care_indices,
                                     area_precision_matrix, pcc_count_matrix,
                                     gt_pcc_points)
    e2e_result_matrix = np.zeros((len(gt_all) + 2, len(det_boxes) + 2))
    gt_trans_not_found = gt_texts_with_ignore[:]
    det_trans_not_found = det_texts[:]

    for gt_idx in range(len(gt_all)):
        if gt_idx in gt_dont_care_indices:
            continue

        if match_matrix.sum(axis=1)[gt_idx] > 0:
            matched_det_indices = np.where(match_matrix[gt_idx] > 0)[0]

            sorted_det_indices = sort_detbox_order_by_pcc(
                gt_idx, matched_det_indices.tolist(), gt_pcc_points,
                pcc_count_matrix)
            corrected_num_chars = lcs_elimination(gt_texts_with_ignore, gt_idx,
                                                  gt_trans_not_found,
                                                  det_trans_not_found,
                                                  sorted_det_indices)

            e2e_result_matrix[gt_idx][len(det_boxes)] = corrected_num_chars
            e2e_result_matrix[gt_idx][len(det_boxes) + 1] = granularity_score(
                len(matched_det_indices))

    for det_index in range(len(det_boxes)):
        if det_index in det_dont_care_indices:
            continue

        if match_matrix.sum(axis=0)[det_index] > 0:
            matched_gt_indices = np.where(match_matrix[:, det_index] == 1)[0]
            e2e_result_matrix[len(gt_all) + 1][det_index] = granularity_score(
                len(matched_gt_indices))
        e2e_result_matrix[len(gt_all)][det_index] = len(
            det_texts[det_index]) - len(det_trans_not_found[det_index])

    chars_recog = get_element_total_length(
        [x for k, x in enumerate(det_texts) if k not in det_dont_care_indices])
    e2e_correct_num_recall = max(
        np.sum(e2e_result_matrix[:, len(det_boxes)]) -
        np.sum(e2e_result_matrix[:, len(det_boxes) + 1]), 0)
    e2e_correct_num_precision = max(
        np.sum(e2e_result_matrix[len(gt_all)]) -
        np.sum(e2e_result_matrix[len(gt_all) + 1]), 0)

    chars_gt, _ = total_character_counts(gt_all, gt_texts_with_ignore,
                                         det_boxes, gt_dont_care_indices,
                                         det_dont_care_indices,
                                         pcc_count_matrix)

    e2e_recall, e2e_precision, e2e_hmean = eval_utils.compute_hmean(
        e2e_correct_num_recall, e2e_correct_num_precision, chars_gt,
        chars_recog)
    img_result = {
        'recall': e2e_recall,
        'precision': e2e_precision,
        'hmean': e2e_hmean
    }
    img_char_info = {
        'num_gts': chars_gt,
        'num_dets': chars_recog,
        'num_recall': e2e_correct_num_recall,
        'num_precision': e2e_correct_num_precision,
    }
    return img_char_info, img_result


def compute_area_precision(gt_polys, pred_polys):
    """Compute the recall and the precision matrices between gt and predicted
    polygons.

    Args:
        gt_polys (list[Polygon]): List of gt polygons.
        pred_polys (list[Polygon]): List of predicted polygons.

    Returns:
        precision (ndarray): Precision matrix of size gt_num x det_num.
    """
    assert isinstance(gt_polys, list)
    assert isinstance(pred_polys, list)

    gt_num = len(gt_polys)
    det_num = len(pred_polys)
    sz = [gt_num, det_num]

    precision = np.zeros(sz)
    # compute area recall and precision for each (gt, det) pair
    # in one img
    for gt_id in range(gt_num):
        for pred_id in range(det_num):
            gt = gt_polys[gt_id]
            det = pred_polys[pred_id]
            inter_area = eval_utils.poly_intersection(det, gt)
            det_area = det.area
            if det_area != 0:
                precision[gt_id, pred_id] = inter_area / det_area

    return precision


def compute_pcc_matrix(gt_boxes, gt_texts):
    """calculate single image pcc.

    Args:
        gt_boxes (list[list[list[int]]]): Ground truth poly.
        gt_texts (list[str]): Ground truth texts. same length with gt_masks
    Returns:
        (list[list[float, float]]): each ground truth poly of pcc points
    """

    def pseudo_character_center(gt_box, transcription):
        '''
            gt_box(list[int]): x1,y1, x2, y2,...xn, yn.
        '''
        chars = list()
        length = len(transcription)
        num_points = len(gt_box) // 2
        # Prepare polygon line estimation with interpolation
        point_x = gt_box[0::2]
        point_y = gt_box[1::2]
        points_x_top = point_x[:num_points // 2]
        points_x_bottom = point_x[num_points // 2:]
        points_y_top = point_y[:num_points // 2]
        points_y_bottom = point_y[num_points // 2:]

        # reverse bottom point order from left to right
        points_x_bottom = points_x_bottom[::-1]
        points_y_bottom = points_y_bottom[::-1]

        num_interpolation_section = (num_points // 2) - 1
        num_points_to_interpolate = length

        new_point_x_top, new_point_x_bottom = list(), list()
        new_point_y_top, new_point_y_bottom = list(), list()

        for sec_idx in range(num_interpolation_section):
            start_x_top, end_x_top = points_x_top[sec_idx], points_x_top[
                sec_idx + 1]
            start_y_top, end_y_top = points_y_top[sec_idx], points_y_top[
                sec_idx + 1]
            start_x_bottom, end_x_bottom = points_x_bottom[
                sec_idx], points_x_bottom[sec_idx + 1]
            start_y_bottom, end_y_bottom = points_y_bottom[
                sec_idx], points_y_bottom[sec_idx + 1]

            diff_x_top = (end_x_top - start_x_top) / num_points_to_interpolate
            diff_y_top = (end_y_top - start_y_top) / num_points_to_interpolate
            diff_x_bottom = (end_x_bottom -
                             start_x_bottom) / num_points_to_interpolate
            diff_y_bottom = (end_y_bottom -
                             start_y_bottom) / num_points_to_interpolate

            new_point_x_top.append(start_x_top)
            new_point_x_bottom.append(start_x_bottom)
            new_point_y_top.append(start_y_top)
            new_point_y_bottom.append(start_y_bottom)

            for num_pt in range(1, num_points_to_interpolate):
                new_point_x_top.append(int(start_x_top + diff_x_top * num_pt))
                new_point_x_bottom.append(
                    int(start_x_bottom + diff_x_bottom * num_pt))
                new_point_y_top.append(int(start_y_top + diff_y_top * num_pt))
                new_point_y_bottom.append(
                    int(start_y_bottom + diff_y_bottom * num_pt))
        new_point_x_top.append(points_x_top[-1])
        new_point_y_top.append(points_y_top[-1])
        new_point_x_bottom.append(points_x_bottom[-1])
        new_point_y_bottom.append(points_y_bottom[-1])

        len_section_for_single_char = (len(new_point_x_top) -
                                       1) / len(transcription)

        for c in range(len(transcription)):
            center_x = (
                new_point_x_top[int(c * len_section_for_single_char)] +
                new_point_x_top[int((c + 1) * len_section_for_single_char)] +
                new_point_x_bottom[int(c * len_section_for_single_char)] +
                new_point_x_bottom[int(
                    (c + 1) * len_section_for_single_char)]) / 4

            center_y = (
                new_point_y_top[int(c * len_section_for_single_char)] +
                new_point_y_top[int((c + 1) * len_section_for_single_char)] +
                new_point_y_bottom[int(c * len_section_for_single_char)] +
                new_point_y_bottom[int(
                    (c + 1) * len_section_for_single_char)]) / 4

            chars.append((center_x, center_y))
        return chars

    gt_pcc_points = []
    for gt_box, gt_text in zip(gt_boxes, gt_texts):
        gt_pcc_points.append(pseudo_character_center(gt_box, gt_text))
    return gt_pcc_points


def compute_pcc_inclusion(gt_polys, det_polys, gt_pcc_points):
    """Compute the recall and the precision matrices between gt and predicted
    polygons.

    Args:
        gt_polys (list[Polygon]): List of gt polygons.
        pred_polys (list[Polygon]): List of predicted polygons.
        gt_pcc_points (list[list[float, float]]): each ground
                                    truth poly of pcc points.
    Returns:
        pcc_count_matrix (List[list[ndarray): matrix of pred poly contain pcc
                                                point
    """
    pcc_count_matrix = []
    for gt_id, gt_box in enumerate(gt_polys):
        det_char_counts = []
        for det_poly in det_polys:
            det_char_count = np.zeros(len(gt_pcc_points[gt_id]))
            for pcc_id, pcc_point in enumerate(gt_pcc_points[gt_id]):
                if det_poly.contains(Point(pcc_point[0], pcc_point[1])):
                    det_char_count[pcc_id] = 1
            det_char_counts.append(det_char_count)
        pcc_count_matrix.append(det_char_counts)
    return pcc_count_matrix


def filter_det_dont_care(det_boxes,
                         area_precision_matrix,
                         pcc_count_matrix,
                         gt_dont_care_indices,
                         iou_thr=0.5):
    """Filter detection Don't care boxes.

    Args:
        det_boxes (list[list[list[int]]]): predict poly.
        area_precision_matrix (ndarray): Precision matrix of size
                                            gt_num x det_num.
        pcc_count_matrix (List[list[ndarray): matrix of pred poly contain pcc
                                                point
        gt_dont_care_indices (List[int]) index of ground truth ignore poly.
        iou_thr (float | 0.5):
    Returns:
        det_dont_care_indices (List[int]): index of pred ignore poly
                                            with has iou > iou_thr.
    """
    det_dont_care_indices = []
    if len(gt_dont_care_indices) > 0:
        for det_id in range(len(det_boxes)):
            area_precision_sum = 0
            for gt_id in gt_dont_care_indices:
                if sum(pcc_count_matrix[gt_id][det_id]) > 0:
                    area_precision_sum += area_precision_matrix[gt_id][det_id]
            if area_precision_sum > iou_thr:
                det_dont_care_indices.append(det_id)
            else:
                for gt_id in gt_dont_care_indices:
                    if area_precision_matrix[gt_id, det_id] > iou_thr:
                        det_dont_care_indices.append(det_id)
                        break
    return det_dont_care_indices


def one_to_one_match(gt_id,
                     det_id,
                     area_precision_matrix,
                     pcc_count_matrix,
                     iou_thr=0.5):
    """One-to-One match condition."""
    cont = 0
    for j in range(len(area_precision_matrix[0])):
        if sum(pcc_count_matrix[gt_id][j]) > 0 and area_precision_matrix[
                gt_id, j] >= iou_thr:
            cont = cont + 1
    if cont != 1:
        return False
    cont = 0
    for i in range(len(area_precision_matrix)):
        if sum(pcc_count_matrix[i][det_id]) > 0 and area_precision_matrix[
                i, det_id] >= iou_thr:
            cont = cont + 1
    if cont != 1:
        return False

    if sum(pcc_count_matrix[gt_id][det_id]) > 0 and area_precision_matrix[
            gt_id, det_id] >= iou_thr:
        return True
    return False


def one_to_many_match(gt_id,
                      area_precision_matrix,
                      pcc_count_matrix,
                      det_dont_care_indices,
                      iou_thr=0.5):
    """One gt to many det condition."""
    many_sum = 0
    detRects = []
    for det_idx in range(len(area_precision_matrix[0])):
        if det_idx not in det_dont_care_indices:
            if area_precision_matrix[gt_id, det_idx] >= iou_thr and sum(
                    pcc_count_matrix[gt_id][det_idx]) > 0:
                many_sum += sum(pcc_count_matrix[gt_id][det_idx])
                detRects.append(det_idx)

    if many_sum > 0 and len(detRects) >= 2:
        return True, detRects
    else:
        return False, []


def many_to_one_match(det_id,
                      area_precision_matrix,
                      pcc_count_matrix,
                      gt_dont_care_indices,
                      iou_thr=0.5):
    """Many-to-One match condition."""
    many_sum = 0
    gtRects = []
    for gt_idx in range(len(area_precision_matrix)):
        if gt_idx not in gt_dont_care_indices:
            if sum(pcc_count_matrix[gt_idx][det_id]) > 0:
                many_sum += area_precision_matrix[gt_idx][det_id]
                gtRects.append(gt_idx)
    if many_sum >= iou_thr and len(gtRects) >= 2:
        return True, gtRects
    else:
        return False, []


def calc_match_matrix(gt_boxes, det_boxes, gt_dont_care_indices,
                      det_dont_care_indices, area_precision_matrix,
                      pcc_count_matrix, gt_pcc_points):
    """Calculate match matrix with PCC counting matrix information.

    Args:
        gt_boxes (list[list[list[int]]]): Ground truth poly.
        det_boxes (list[list[list[int]]]): predict poly.
        gt_dont_care_indices (List[int]) index of ground truth ignore poly.
        det_dont_care_indices (List[int]): index of pred ignore poly
                                            with has iou > iou_thr.
        area_precision_matrix (ndarray): Precision matrix of size
                                            gt_num x det_num.
        pcc_count_matrix (List[list[ndarray): matrix of pred poly contain pcc
                                                point
        gt_pcc_points (list[list[float, float]]): each ground
                                    truth poly of pcc points.
    Returns:
        match_matrix (ndarray): match matrix with PCC counting
                                matrix information.
    """
    match_matrix = np.zeros([len(gt_boxes), len(det_boxes)])
    pairs = []
    for gt_id in range(len(gt_boxes)):
        for det_id in range(len(det_boxes)):
            if gt_id not in gt_dont_care_indices and \
                    det_id not in det_dont_care_indices:
                match = one_to_one_match(gt_id, det_id, area_precision_matrix,
                                         pcc_count_matrix)
                if match:
                    pairs.append({
                        'gt': [gt_id],
                        'det': [det_id],
                        'type': 'OO'
                    })

    # one-to-many match
    for gt_id in range(len(gt_boxes)):
        if gt_id not in gt_dont_care_indices:
            match, matched_det = one_to_many_match(gt_id,
                                                   area_precision_matrix,
                                                   pcc_count_matrix,
                                                   det_dont_care_indices)
            if match:
                pairs.append({'gt': [gt_id], 'det': matched_det, 'type': 'OM'})

    # many-to-one match
    for det_id in range(len(det_boxes)):
        if det_id not in det_dont_care_indices:
            match, matched_gt = many_to_one_match(det_id,
                                                  area_precision_matrix,
                                                  pcc_count_matrix,
                                                  gt_dont_care_indices)
            if match:
                pairs.append({'gt': matched_gt, 'det': [det_id], 'type': 'MO'})
    for pair in pairs:
        match_matrix[pair['gt'], pair['det']] = 1

    # clear pcc count flag for not matched pairs
    for gt_idx in range(len(gt_boxes)):
        for det_idx in range(len(det_boxes)):
            if not match_matrix[gt_idx][det_idx]:
                for pcc in range(len(gt_pcc_points[gt_idx])):
                    pcc_count_matrix[gt_idx][det_idx][pcc] = 0
    return match_matrix


def sort_detbox_order_by_pcc(gt_idx, det_indices, gt_pcc_points,
                             pcc_count_matrix):
    """sort detected box order by pcc information."""
    char_len = len(gt_pcc_points[gt_idx])

    not_ordered_yet = det_indices
    ordered_indices = list()

    for c in range(char_len):
        if len(not_ordered_yet) == 1:
            break

        for det_id in not_ordered_yet:
            if pcc_count_matrix[gt_idx][det_id][c] == 1:
                ordered_indices.append(det_id)
                not_ordered_yet.remove(det_id)
                break

    ordered_indices.append(not_ordered_yet[0])
    return ordered_indices


def lcs_elimination(gt_texts, gt_idx, gt_trans_not_found, det_trans_not_found,
                    sorted_det_indices):
    """longest common sequence elimination by sorted detection boxes."""
    standard_script = gt_texts[gt_idx]
    lcs_length, lcs_string = longest_common_sequence(
        standard_script,
        ''.join(det_trans_not_found[idx] for idx in sorted_det_indices))
    for c in lcs_string:
        gt_trans_not_found[gt_idx] = gt_trans_not_found[gt_idx].replace(
            c, '', 1)

        for det_idx in sorted_det_indices:
            if not det_trans_not_found[det_idx].find(c) < 0:
                det_trans_not_found[det_idx] = det_trans_not_found[
                    det_idx].replace(c, '', 1)
                break
    return lcs_length


def granularity_score(num_splitted, granularity_weight=1):
    """get granularity penalty given number of how many splitted."""
    return max(num_splitted - 1, 0) * granularity_weight


def total_character_counts(gt_boxes, gt_texts, det_boxes, gt_dont_care_indices,
                           det_dont_care_indices, pcc_count_matrix):
    """get TotalNum for detection evaluation."""
    total_num_recall = 0
    total_num_precision = 0
    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in gt_dont_care_indices:
            total_num_recall += len(gt_texts[gt_idx])

        for det_idx in range(len(det_boxes)):
            if det_idx not in det_dont_care_indices:
                total_num_precision += sum(pcc_count_matrix[gt_idx][det_idx])
    return total_num_recall, total_num_precision


def longest_common_sequence(s1, s2):
    """Longeset Common Sequence between s1 & s2."""

    if len(s1) == 0 or len(s2) == 0:
        return 0, ''
    matrix = [['' for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)
    cs = matrix[-1][-1]
    return len(cs), cs


def get_element_total_length(indices):
    return sum([len(x) for x in indices])


def extract_gt_info(ann_infos):
    """Get ground truth masks and ignored masks.

    Args:
        ann_infos (list[dict]): Each dict contains annotation
            infos of one image, containing following keys:
            masks, masks_ignore, gt_texts, gt_ignore_texts
    Returns:
        gt_masks (list[list[list[int]]]): Ground truth masks.
        gt_masks_ignore (list[list[list[int]]]): Ignored masks.
        gt_texts (list[str]): Ground truth texts. same length with gt_masks
        gt_ignore_texts (list[str]): Ignored texts. same length with gt_masks
    """
    assert check.is_type_list(ann_infos, dict)

    gt_masks = []
    gt_masks_ignore = []
    gt_texts = []
    gt_ignore_texts = []
    for ann_info in ann_infos:
        masks = ann_info['masks']
        mask_gt = []
        for mask in masks:
            assert len(mask[0]) >= 8 and len(mask[0]) % 2 == 0
            mask_gt.append(mask[0])
        gt_texts.append(ann_info['gt_texts'])
        gt_masks.append(mask_gt)

        masks_ignore = ann_info['masks_ignore']
        mask_gt_ignore = []
        for mask_ignore in masks_ignore:
            assert len(mask_ignore[0]) >= 8 and len(mask_ignore[0]) % 2 == 0
            mask_gt_ignore.append(mask_ignore[0])
        gt_masks_ignore.append(mask_gt_ignore)
        gt_ignore_texts.append(ann_info['gt_ignore_texts'])

    return gt_masks, gt_masks_ignore, gt_texts, gt_ignore_texts


def extract_pred_info(results):
    """Get pred masks scores and texts.

    Args:
        results (list[dict]): Each dict contains annotation
            infos of one image,
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
    Returns:
        pred_polys (list[list[list[int]]]): predict polygons.
        pred_scores (list[list[list[int]]]): predict scores.
        gt_texts (list[str]): predict texts. same length with pred_polys
    """
    pred_polys = []
    pred_scores = []
    pred_texts = []
    for result in results:
        pred_poly = []
        pred_score = []
        pred_text = []
        for info in result['polygons']:
            pred_poly.append(info['polygon'])
            pred_score.append(info['det_score'])
            pred_text.append(info['text_label'])
        if len(pred_poly) > 0:
            assert check.valid_boundary(pred_poly[0], False)
        pred_polys.append(pred_poly)
        pred_scores.append(pred_score)
        pred_texts.append(pred_text)

    return pred_polys, pred_scores, pred_texts


def merge_all_info(img_infos, gt_dataset_boxes, gts_dataset_ignore_boxes,
                   gt_dataset_texts, gt_dataset_ignore_texts,
                   pred_dataset_polys, pred_dataset_texts, pred_dataset_scores,
                   single_img_metrics):
    """Get dataset merge results.
    Returns:
        List[dict] dataset merge results
        [
            {
            "filename": "NGE.base64url.jpg",
            //[polygon + score] i.e.[x1, y1, ... xn, yn, score]
            "pred_det": list[list[float]],
            "gt_det": list[list[float]], //[polygon] i.e.[x1, y1, ... xn, yn]
            "pred_text": list[str],
            "gt_text": list[str],
            "hmean": 0.667,
            "recall":0.636
            "precision":0.7
            }
        ]
    """
    dataset_merge_results = []
    for i in range(len(img_infos)):
        single_merge_result = {}
        single_merge_result.update(img_infos[i])
        pred_det = []
        for poly, score in zip(pred_dataset_polys[i], pred_dataset_scores[i]):
            pred_det.append(poly + [score])
        single_merge_result['pred_det'] = pred_det
        single_merge_result['pred_text'] = pred_dataset_texts[i]
        single_merge_result[
            'gt_det'] = gt_dataset_boxes[i] + gts_dataset_ignore_boxes[i]
        single_merge_result[
            'gt_text'] = gt_dataset_texts[i] + gt_dataset_ignore_texts[i]
        single_merge_result.update(single_img_metrics[i])
        dataset_merge_results.append(single_merge_result)
    return dataset_merge_results
