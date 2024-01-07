import click
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from models.model import MyAwesomeModel

# from data import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
@click.option(
    "--model_location", default="mnist_classifier/models/trained_models/", help="number of epochs to train for"
)
def train(lr, batch_size, num_epochs, model_location):
    """Train a model on MNIST."""
    print("Training day and night")
    print("Learning rate:", lr)
    print("Batch size:", batch_size)
    print("Number of epochs:", num_epochs)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)

    train_data = torch.load("data/processed/processed_data_train.pt")
    train_labels = torch.load("data/processed/processed_labels_train.pt")

    train_set = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            pred_labels = model.forward(images)
            loss = criterion(pred_labels, labels)

            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
        print(f"Epoch {epoch} Loss {loss}")
        losses.append(loss.item())

    plt.plot(list(range(num_epochs)), losses)
    plt.savefig("reports/figures/training_curve.png")

    torch.save(model, f"{model_location}/model.pt")


cli.add_command(train)

if __name__ == "__main__":
    train()
