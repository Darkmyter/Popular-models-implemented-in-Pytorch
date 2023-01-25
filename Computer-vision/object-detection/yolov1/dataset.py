import torch
import os
import pandas as pd
import PIL.Image as Image
import albumentations as A

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, split_size=7, num_boxes=2, num_classes=20, transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = split_size
        self.C = num_classes
        self.B = num_boxes

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        label_path = self.label_dir/ self.annotations.iloc[index, 1]
        label = {"class_ids": [], "bboxes": []}
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                label["class_ids"].append(int(class_label))
                label["bboxes"].append([x, y, width, height])

        img_path = self.img_dir/ self.annotations.iloc[index, 0]
        image = Image.open(img_path)

        transformed = self.transform(image=image, boxes=label["bboxes"], class_ids=label["class_ids"])

        image = torch.tensor(transformed['image'], dtype=torch.float).permute(2,0,1) / 255

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in transformed['bboxes']:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * x), int(self.S * y)
            x_cell, y_cell = self.S * x - i, self.S * y - j

            if label_matrix[i, j, 20] == 0:
                
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width, height]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
    

def get_VOCDataset(params, augment):

    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(width=448, height=448, scale=(0.7, 1.0)),
            A.ColorJitter(),
            A.Blur(blur_limit = 7, always_apply = False, p = 0.5),
            A.GaussNoise(var_limit = 40, mean = 0, per_channel = True, always_apply = False, p=0.5)
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.8, label_fields=['class_ids'])
        ) 
    else:
         transform = A.Compose([
            A.Resize(width=448, height=448),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.75, label_fields=['class_ids'])
        )

    return VOCDataset(
        params['training_csv'],
        transform=transform,
        img_dir=params['img_dir'],
        label_dir=params['label_dir']
    )
        