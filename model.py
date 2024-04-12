import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class FontClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(FontClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = self.pool(F.tanh(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = F.tanh(self.fc3(x))
        #x = self.dropout(x)
        x =  (self.fc4(x))
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc)
        #print('train loss is : ', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc)
        #print('val loss is : ', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.00001)
