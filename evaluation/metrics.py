import numpy as np


def dice_score(pred, target, threshold=0.5, eps=1e-7):
    pred = np.asarray(pred) >= threshold
    target = np.asarray(target) >= threshold
    intersection = np.logical_and(pred, target).sum()
    return float((2 * intersection + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred = np.asarray(pred) >= threshold
    target = np.asarray(target) >= threshold
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + eps) / (union + eps))
