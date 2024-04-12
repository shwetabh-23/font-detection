from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        #breakpoint()
        image = image.resize((512, 512))
        num_instances = self.data.iloc[idx, 1] - 1
        bounding_boxes = self.data.iloc[idx, 2]
        sample = {'image': image, 'num_instances': num_instances, 'bounding_boxes': bounding_boxes}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
