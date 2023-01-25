import torch
import os
import pandas as pd
import PIL.Image as Image

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
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = self.img_dir/ self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
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
        