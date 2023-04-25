import numpy as np

import torch
import torch.nn as nn
import torchmetrics

import pytorch_lightning as pl


class Seq2SeqModel(pl.LightningModule):
    def __init__(
        self,
        loss=nn.MSELoss(),
        optimizer="Adam",
        lr=0.01,
        num_classes=None,
    ):
        super().__init__()
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.save_hyperparameters(ignore=["loss"])

        # print(num_classes)
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.accuracy = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes
            )
            self.valid_accuracy = torchmetrics.Accuracy(
                task="multiclass", top_k=1, num_classes=num_classes
            )

    def forward(self, x):
        pass

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        #     "interval": "epoch",
        #     "frequency": 1,
        #     "monitor": "train_loss_epoch",
        # }

        return {"optimizer": optimizer}
        # return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.num_classes is not None:
            y_hat = y_hat[:, -1, :]
            self.accuracy(y_hat, y)
            self.log(
                "train_acc",
                self.accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        trainloss = self.loss(y_hat, y)
        self.log(
            "train_loss",
            trainloss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        grad_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (1.0 / 2)
        self.log(
            "grad_norm",
            grad_norm,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return trainloss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.num_classes is not None:
            y_hat = y_hat[:, -1, :]
            self.valid_accuracy(y_hat, y)
            self.log(
                "valid_acc",
                self.valid_accuracy,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        validloss = self.loss(y_hat, y)
        self.log(
            "valid_loss",
            validloss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return validloss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.DoubleTensor(x, dtype=torch.float64)
            y_hat = self(x)
            if self.num_classes is not None:
                y_hat = y_hat[:, -1, :]

            return y_hat.detach().cpu().numpy()
