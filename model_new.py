import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(262144, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = F.tanh(self.fc3(x))
        #x = self.dropout(x)
        x =  (self.fc4(x))
        
        return x

    def training_step(self, batch, batch_idx):
        images, num_instances, bounding_boxes = batch['image'], batch['num_instances'], batch['bounding_boxes']
        num_instances_pred = self(images)
        # num_instances = num_instances.view(-1, 1)
        # num_instances = num_instances.float()


        loss_num_instances = F.cross_entropy(num_instances_pred, num_instances)
        #loss_bounding_boxes = F.mse_loss(bounding_boxes_pred, bounding_boxes)
        total_loss = loss_num_instances 
        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, num_instances, bounding_boxes = batch['image'], batch['num_instances'], batch['bounding_boxes']
        num_instances_pred= self(images)
        # num_instances = num_instances.view(-1, 1)
        # num_instances = num_instances.float()


        loss_num_instances = F.cross_entropy(num_instances_pred, num_instances)
        #loss_bounding_boxes = F.mse_loss(bounding_boxes_pred, bounding_boxes)
        total_loss = loss_num_instances 
        self.log('val_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer