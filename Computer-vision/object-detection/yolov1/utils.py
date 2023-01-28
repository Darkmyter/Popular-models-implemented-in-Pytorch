import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os


def intersection_over_union(box1, box2):
    """
    Compute iou between two tensors of bboxs : [..., 4]

    Args:
        box1:
            tensor 1
        box2:
            tensor 2
    
    Returns:
        iou computer on last dim, keep the dims.
    """
    box1x1 = box1[..., 0] - box1[..., 2] / 2
    box1y1 = box1[..., 1] - box1[..., 3] / 2
    box1x2 = box1[..., 0] + box1[..., 2] / 2
    box1y2 = box1[..., 1] + box1[..., 3] / 2
    box2x1 = box2[..., 0] - box2[..., 2] / 2
    box2y1 = box2[..., 1] - box2[..., 3] / 2
    box2x2 = box2[..., 0] + box2[..., 2] / 2
    box2y2 = box2[..., 1] + box2[..., 3] / 2

    box1area = torch.abs((box1x1 - box1x2) * (box1y1 - box1y2))
    box2area = torch.abs((box2x1 - box2x2) * (box2y1 - box2y2))

    x1 = torch.max(box1x1, box2x1)
    y1 = torch.max(box1y1, box2y1)
    x2 = torch.min(box1x2, box2x2)
    y2 = torch.min(box1y2, box2y2)

    intersection_area = torch.clamp(x2 - x1, min = 0) * torch.clamp(y2 - y1, min = 0)

    iou = intersection_area / (box1area + box2area - intersection_area + 1e-6)
    return iou

def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Filter bboxes using Non Max Suppression

    Args:
        bboxs:
            list of list of bboxes [class_pred, prob_score, x1, y1, h, w]
        iou_threshold:
            iou threshold where predicted bboxes is accepted as correct
        threshold:
            threshold to remove predicted bboxes based on confidence

    Returns:
        bboxes after performing NMS
    """
    
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
            torch.tensor(chosen_box[2:]),
            torch.tensor(box[2:]),
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(pred_bboxes, labels_bboxes, iou_threshold=0.5, num_classes=20):
    """
    Compute mean average precision

    Args:
        pred_bbox:
            list of list of bboxes [train_idx, class_pred, prob_score, x1, y1, h, w]
        labels_bbox:
            list of list of target bboxes [train_idx, class_pred, prob_score, x1, y1, h, w]
        iou_threshold:
            iou threshold where predicted bboxes is accepted as correct
        num_classes:
            number of classes

    Returns:
        mean average precision value across all classes
    """

    # AP for each class
    average_precision = []

    epsilon = 1e-6


    for c in range(num_classes):
        detections = []
        ground_truths = []

        # filter by class c
        for true_box in labels_bboxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        for pred_box in pred_bboxes:
            if pred_box[1] == c:
                detections.append(pred_box)

        
        # count number of target bboxes per image:
        # count_bboxes = {0:3, 1:2}
        count_bboxes = Counter([gt[0] for gt in ground_truths])

        # count_bboxes = {0: torch.tensor[0,0,0], 0: torch.tensor[0,0]}
        for key, val in count_bboxes:
            count_bboxes[key] = torch.zeros(val)

        # sort by box probability
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # if none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[:3]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                
            if best_iou > iou_threshold:
                # only detect gt once
                if count_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    count_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            # if iou less than threshold
            else:
                FP[detection_idx] = 1

        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precision.append(torch.trapz(precisions, recalls))
    
    return sum(average_precision) / len(average_precision)  

def save_checkpoint(model, optimizer, scheduler, filename):
    filename = "checkpoints/" + filename + ".pth.tar"
    print(f"Saving model to {filename}")
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    if not os.path.exists('checkpoints'): 
        os.makedirs('checkpoints')
    
    torch.save(state, filename)

def load_checkpoint(model, optimizer, scheduler, filename): 
    filename = "checkpoints/" + filename + ".pth.tar"
    if not os.path.isfile(filename): 
        print("Cannot find checkpoint file! Generating new model...")
        return
    
    print(f"Loading model from {filename}")
    checkpoint = torch.load(filename)
    if model: model.load_state_dict(checkpoint['model'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler: scheduler.load_state_dict(checkpoint['scheduler'])