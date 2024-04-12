from torchvision.transforms import ToTensor
from dataset_new import CustomDataset
from torch.utils.data import DataLoader
from model_new import Model
import pytorch_lightning as pl
from torchvision.transforms import Resize
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import os
import torchvision.transforms as transforms


transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)
os.makedirs('model_new', exist_ok= True)
# Load dataset
train_dataset = CustomDataset(csv_file='synthetic_dataset.csv', transform=transform)
train_data, val_data = train_test_split(train_dataset, test_size=0.3, random_state=42)
# def custom_collate(batch):
#     # Separate the batch into lists of images, num_instances, and bounding_boxes
#     images = [item['image'] for item in batch]
#     num_instances = [item['num_instances'] for item in batch]
#     bounding_boxes = [item['bounding_boxes'] for item in batch]
    
#     # Resize images to (512, 512)
#     resize_transform = Resize((512, 512))
#     resized_images = [resize_transform(image) for image in images]

#     # Convert resized images to tensors
#     resized_images = [torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) for image in resized_images]

#     # Return the resized images along with num_instances and bounding_boxes
#     return {'image': torch.stack(resized_images), 
#             'num_instances': torch.tensor(num_instances), 
#             'bounding_boxes': bounding_boxes}

#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn= custom_collate)
# Train the model
model = Model()
model_path = r'models_new/model_4.pth'
logger = TensorBoardLogger(save_dir= r'models_new/', name= 'v4')
if not os.path.exists(model_path):
    trainer = pl.Trainer(max_epochs= 200, accelerator= 'cpu', enable_progress_bar= True, logger= logger, callbacks= [early_stop_callback])
    trainer.fit(model, DataLoader(train_data, batch_size=16), DataLoader(val_data, batch_size=16))
    breakpoint()
    #torch.save(model.state_dict(), model_path)
