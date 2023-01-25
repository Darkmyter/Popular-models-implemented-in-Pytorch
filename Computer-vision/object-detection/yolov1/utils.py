import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os


def intersection_over_union(box1, box2, box_format="midpoint"):

    if box_format == "corners":
        box1x1 = box1[..., 0:1]
        box1y1 = box1[..., 1:2]
        box1x2 = box1[..., 2:3]
        box1y2 = box1[..., 3:4]  # (N, 1)
        box2x1 = box2[..., 0:1]
        box2y1 = box2[..., 1:2]
        box2x2 = box2[..., 2:3]
        box2y2 = box2[..., 3:4]
    if box_format == "midpoint":
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