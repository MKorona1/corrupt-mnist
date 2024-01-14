import logging

import click
import hydra
import torch
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch import nn, optim

from models.model import MyAwesomeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @click.group()
# def cli():
#     """Command line interface."""
#     pass


# @click.command()
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
# @click.option("--batch_size", default=256, help="batch size to use for training")
# @click.option("--num_epochs", default=20, help="number of epochs to train for")
# @click.option(
#     "--model_location", default="mnist_classifier/models/trained_models/", help="number of epochs to train for"
# )


@hydra.main(config_path="conf", config_name="default_training_conf.yaml")
def train(config):
    """Train a model on MNIST."""

    hparams = config.experiment.hyperparameters
    log = logging.getLogger(__name__)

    log.info("Training day and night")
    log.info(f"Learning rate: {hparams.lr}")
    log.info(f"Batch size: {hparams.batch_size}")
    log.info(f"Number of epochs: {hparams.num_epochs}")

    # TODO: Implement training loop here
    model_config = OmegaConf.load(to_absolute_path("mnist_classifier/models/conf/experiment/exp1.yaml"))
    dimensions = model_config.dimensions

    model = MyAwesomeModel(
        dimensions.input_dim,
        dimensions.first_hidden_dim,
        dimensions.second_hidden_dim,
        dimensions.third_hidden_dim,
        dimensions.output_dim,
    ).to(device)
    train_data = torch.load(to_absolute_path("data/processed/processed_data_train.pt"))
    train_labels = torch.load(to_absolute_path("data/processed/processed_labels_train.pt"))

    train_set = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
    losses = []
    for epoch in range(hparams.num_epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            pred_labels = model.forward(images)
            loss = criterion(pred_labels, labels)

            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
        log.info(f"Epoch {epoch} Loss {loss}")
        losses.append(loss.item())

    # plt.plot(list(range(num_epochs)), losses)
    # plt.savefig("reports/figures/training_curve.png")

    torch.save(model, to_absolute_path(f"{hparams.model_location}/model.pt"))


# cli.add_command(train)

if __name__ == "__main__":
    train()
