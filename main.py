from create_dataset import FontDataset
from sklearn.model_selection import train_test_split
from model import FontClassifier
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks import EarlyStopping
import os

data_dir = r'synthetic_data'

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)

dataset = FontDataset(data_dir=data_dir)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
num_classes = 10
model = FontClassifier(num_classes)

# Initialize Lightning trainer
model_path = r'models/model_10.pth'
logger = TensorBoardLogger(save_dir= r'models/', name= 'v7')
if not os.path.exists(model_path):
    trainer = pl.Trainer(max_epochs= 200, accelerator= 'cuda', enable_progress_bar= True, logger= logger, callbacks= [early_stop_callback])
    trainer.fit(model, DataLoader(train_data, batch_size=32, num_workers=4), DataLoader(val_data, batch_size=32, num_workers=4))
    torch.save(model.state_dict(), model_path)

# state_dict = torch.load(model_path)
# model.load_state_dict(state_dict=state_dict)
# model = model.to('cpu')
# trainer = pl.Trainer(logger=logger, accelerator= 'cuda')
# #result = trainer.test(model=model)
# sample = model.sample(n = 1)
# breakpoint()

trainer = pl.Trainer(max_epochs=10)

# Train the model
breakpoint()