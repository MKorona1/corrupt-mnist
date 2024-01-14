import logging

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn, optim

import wandb

log = logging.getLogger(__name__)


class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self, input_dim, first_hidden_dim, second_hidden_dim, third_hidden_dim, output_dim):
        super().__init__()
        # self.backbone = nn.Sequential(
        #     nn.Linear(input_dim, first_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(first_hidden_dim, second_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(second_hidden_dim, third_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(third_hidden_dim, output_dim),
        #     nn.LogSoftmax()
        # )

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # [B, 1, 28, 28] -> [B, 32, 26, 26]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # [B, 32, 26, 26] -> [B, 64, 24, 24]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),  # [B, 64, 24, 24] -> [B, 128, 22, 22]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 22, 22] -> [B, 128, 11, 11]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 128, 11, 11] -> [B, 64 * 12 * 12]
            nn.Linear(128 * 11 * 11, 10),
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = self.criterium(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)
