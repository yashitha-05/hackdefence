import torch

def compute_iou(preds, targets, num_classes=10):
    ious = []
    preds = torch.argmax(preds, dim=1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    mean_iou = sum([iou for iou in ious if not torch.isnan(torch.tensor(iou))]) / len(ious)
    return mean_iou, ious