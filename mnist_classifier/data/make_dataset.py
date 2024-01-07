import click
import torch


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--raw_location", default="data/raw/", help="Path to load data from")
@click.option("--processed_location", default="data/processed/", help="Path to save processed data")
# @click.argument('raw_location')
# @click.argument('processed_location')
def get_data(raw_location, processed_location):
    """Return train and test dataloaders for MNIST."""
    
    n = 6  # number of files

    train_data = [torch.load(f"{raw_location}/train_images_{i}.pt") for i in range(n)]
    train_labels = [torch.load(f"{raw_location}/train_target_{i}.pt") for i in range(n)]

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(f"{raw_location}/test_images.pt")
    test_labels = torch.load(f"{raw_location}/test_target.pt")

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    mu_train = torch.mean(train_data, dim=(2, 3), keepdim=True)
    std_train = torch.std(train_data, dim=(2, 3), keepdim=True)
    train_data = (train_data - mu_train) / std_train

    mu_test = torch.mean(test_data, dim=(2, 3), keepdim=True)
    std_test = torch.std(test_data, dim=(2, 3), keepdim=True)
    test_data = (test_data - mu_test) / std_test

    torch.save(train_data, f"{processed_location}/processed_data_train.pt")
    torch.save(train_labels, f"{processed_location}/processed_labels_train.pt")

    torch.save(test_data, f"{processed_location}/processed_data_test.pt")
    torch.save(test_labels, f"{processed_location}/processed_labels_test.pt")


cli.add_command(get_data)

if __name__ == "__main__":
    # Get the data and process it
    get_data()
