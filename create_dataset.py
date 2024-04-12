import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

font_mapping = {
    "Oswald-Regular": 0,
    "Roboto-Regular": 1,
    "OpenSans-Regular": 2,
    "Ubuntu-Regular": 3,
    "PTSerif-Regular": 4,
    "DancingScript-Regular": 5,
    "FredokaOne-Regular": 6,
    "Arimo-Regular": 7,
    "NotoSans-Regular": 8,
    "PatuaOne-Regular": 9
}

class FontDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [file for file in os.listdir(data_dir) if file.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        
        # Extract font label from filename
        font_name = self.image_files[idx].split('_')[1]
        font_label = font_mapping[font_name]
        return image, font_label
