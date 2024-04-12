import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class RPN(pl.LightningModule):
    def __init__(self, w_feature_map, h_feature_map, k, n_anchors):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.regressor = nn.Conv2d(512, 4*k, kernel_size=1)
        self.classifier = nn.Conv2d(512, k, kernel_size=1)
        self.n_anchors = n_anchors

    def forward(self, x):
        x = F.relu(self.conv1(x))
        regressor = self.regressor(x)
        classifier = torch.sigmoid(self.classifier(x))
        return regressor, classifier

    def smooth_l1_loss(self, y_true, y_pred):
        """
        Calculates Smooth L1 loss
        """
        print("y_true shape:", y_true.shape)
        print("y_pred shape:", y_pred.shape)
        x = torch.abs(y_true - y_pred)
        mask = torch.lt(x, 1.0).float()
        loss = (mask * 0.5 * x ** 2) + (1 - mask) * (x - 0.5)
        return loss.mean()

    def custom_l1_loss(self, y_true, y_pred):
        offset_list = y_true[:, :, :-1]
        label_list = y_true[:, :, -1]
        y_pred = y_pred.view(-1, self.n_anchors, 4)
        positive_idxs = torch.where(label_list == 1)
        bbox = y_pred[positive_idxs]
        target_bbox = offset_list[positive_idxs]
        loss = self.smooth_l1_loss(target_bbox, bbox)
        return loss

    def custom_binary_loss(self, y_true, y_pred_objectiveness):
        y_pred = y_pred_objectiveness.view(-1, self.n_anchors)
        y_true = y_true.squeeze(-1)
        indices = torch.where(y_true != -1)
        rpn_match_logits = y_pred[indices]
        anchor_class = y_true[indices]
        loss = F.binary_cross_entropy(rpn_match_logits, anchor_class.float())
        return loss

    def training_step(self, batch, batch_idx):
        x, offset_list_label_list, label_list = batch
        regressor, classifier = self(x)
        l1_loss = self.custom_l1_loss(offset_list_label_list, regressor)
        binary_loss = self.custom_binary_loss(label_list, classifier)
        total_loss = l1_loss + binary_loss
        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
# k = 9
# # Assuming you have feature_maps, offset_list_label_list, and label_list as torch tensors
# feature_maps = torch.randn(1, 512, 50, 50)
# offset_list_label_list = torch.randn(1, 4*k, 50, 50)
# label_list = torch.randn(1, k, 50, 50)

# w_feature_map, h_feature_map, k = 50, 50, 9
# n_anchors = 50 * 50

# # Initialize Lightning module
# rpn_model = RPN(w_feature_map, h_feature_map, k, n_anchors)

# # Wrap tensors into DataLoader
# dataset = TensorDataset(feature_maps, offset_list_label_list, label_list)
# train_loader = DataLoader(dataset, batch_size=1)

# # Define Lightning Trainer
# trainer = pl.Trainer(max_epochs=100)

# # Train the model
# trainer.fit(rpn_model, train_loader)
