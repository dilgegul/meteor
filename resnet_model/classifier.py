import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import models
from torch import nn

class Classifier(pl.LightningModule):
    def __init__(self, model="resnet18"):
        super().__init__()
        if model == "resnet18":
            self.model = models.resnet18()
            self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, 1)
            self.model.fc.register_forward_hook(self.get_features)
        else:
            return NotImplementedError()

        self.criterion = nn.BCEWithLogitsLoss()

    def get_features(self, module, inputs, outputs):
        self.features = inputs

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y, id = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        return loss

    def training_step_2(self, x, y):
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        return loss

    def training_step_end(self, losses):
        self.log("train_loss", losses.cpu().detach().mean())

    def validation_step(self, batch, batch_idx):
        x, y, id = batch
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        y_scores = torch.sigmoid(y_hat)
        return {"y_scores": y_scores.cpu().detach(), "y_true": y.cpu().detach(), "loss": loss.cpu().numpy()}

    def validation_step_2(self, x, y):
        y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        y_scores = torch.sigmoid(y_hat)
        return {"y_scores": y_scores.cpu().detach(), "y_true": y.cpu().detach(), "loss": loss.cpu().detach().numpy()}

    def validation_epoch_end(self, outputs):
        y_true = np.hstack([o["y_true"] for o in outputs])
        y_scores = np.hstack([o["y_scores"] for o in outputs])
        loss = np.hstack([o["loss"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_scores = y_scores.reshape(-1)
        y_pred = y_scores > 0.5

        print()
        self.log("val_loss", loss.mean())
        self.log("val_accuracy", (y_true == y_pred).mean())

    def validation_epoch_end_2(self, outputs):
        y_true = np.hstack(outputs["y_true"])
        y_scores = np.hstack(outputs["y_scores"])
        loss = outputs["loss"]

        y_true = y_true.reshape(-1).astype(int)
        y_scores = y_scores.reshape(-1)
        y_pred = y_scores > 0.5

        print()
        self.log("val_loss", loss.mean())
        self.log("val_accuracy", (y_true == y_pred).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-8)
