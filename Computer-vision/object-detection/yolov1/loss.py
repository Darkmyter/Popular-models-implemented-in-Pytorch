import torch
from torch import nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        batch_size = target.size()[0]
        print(target.shape)

        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        class_predictions = predictions[..., :self.C]

        class_target = target[..., :self.C]
        indicator_i = target[..., self.C].unsqueeze(3)

        box_predictions = predictions[..., self.C:].reshape(-1, self.S, self.S, self.B, 5)

        box_target = target[..., self.C:self.C+5]
        box_target = torch.cat((box_target, box_target), dim=3).reshape(-1, self.S, self.S, self.B, 5)
        print(box_predictions.shape)
        print(box_target.shape)
        iou = torch.cat(
            [
                intersection_over_union(
                    box_predictions[..., i, 1:],
                    box_target[..., i, 1:],
                ).unsqueeze(3).unsqueeze(0)
                for i in range(self.B)
            ],
            dim = 0
        )

        best_iou, best_box = torch.max(iou, dim=0)

        first_box_mask = torch.cat((torch.ones_like(indicator_i), torch.zeros_like(indicator_i)), dim=3)
        second_box_mask = torch.cat((torch.zeros_like(indicator_i), torch.ones_like(indicator_i)), dim=3)

        indicator_ij = (indicator_i * ((1-best_box) * first_box_mask + best_box * second_box_mask)).unsqueeze(4)

        box_target[..., 0] = torch.cat((best_iou, best_iou), dim=3)
        box_target = indicator_ij * box_target

        # For box coordinates
        xy_loss = self.lambda_coord * self.mse(
            indicator_ij * box_predictions[..., 1:3],
            indicator_ij * box_target[..., 1:3]
        )

        wh_loss = self.lambda_coord * self.mse(
            indicator_ij * torch.sign(box_predictions[..., 3:5]) * torch.sqrt(torch.abs(box_predictions[..., 3:5]) + 1e-6),
            indicator_ij * torch.sign(box_target[..., 3:5]) * torch.sqrt(torch.abs(box_target[..., 3:5]) + 1e-6)
        )


        # For object loss
        object_loss = self.mse(
            indicator_ij * box_predictions[..., 0:1],
            indicator_ij * box_target[..., 0:1]
        )

        # for no object loss
        no_object_loss = self.lambda_noobj * self.mse(
            (1-indicator_ij) * box_predictions[..., 0:1],
            (1-indicator_ij) * box_target[..., 0:1]
        )


        # FOr class loss
        class_loss = self.mse(
            indicator_i * class_predictions, 
            indicator_i * class_target
        )


        loss = (
            xy_loss + wh_loss + object_loss + no_object_loss + class_loss
        ) / float(batch_size)

        return loss