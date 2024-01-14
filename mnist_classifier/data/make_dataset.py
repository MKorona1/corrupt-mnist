import click
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule


# @click.group()
# def cli():
#     """Command line interface."""
#     pass


# @click.command()
# @click.option("--raw_location", default="data/raw/", help="Path to load data from")
# @click.option("--processed_location", default="data/processed/", help="Path to save processed data")
# @click.argument('raw_location')
# @click.argument('processed_location')

class DataModule(LightningDataModule):
    def __init__(self, raw_location="data/raw/", processed_location="data/processed/"):
        super().__init__()
        self.raw_location = raw_location
        self.processed_location = processed_location

    def process_data(self):
        """Return train and test dataloaders for MNIST."""
        
        n = 6  # number of files

        train_data = [torch.load(f"{self.raw_location}/train_images_{i}.pt") for i in range(n)]
        train_labels = [torch.load(f"{self.raw_location}/train_target_{i}.pt") for i in range(n)]

        train_data = torch.cat(train_data, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_data = torch.load(f"{self.raw_location}/test_images.pt")
        test_labels = torch.load(f"{self.raw_location}/test_target.pt")

        train_data = train_data.unsqueeze(1)
        test_data = test_data.unsqueeze(1)

        mu_train = torch.mean(train_data, dim=(2, 3), keepdim=True)
        std_train = torch.std(train_data, dim=(2, 3), keepdim=True)
        train_data = (train_data - mu_train) / std_train

        mu_test = torch.mean(test_data, dim=(2, 3), keepdim=True)
        std_test = torch.std(test_data, dim=(2, 3), keepdim=True)
        test_data = (test_data - mu_test) / std_test

        torch.save(train_data, f"{self.processed_location}/processed_data_train.pt")
        torch.save(train_labels, f"{self.processed_location}/processed_labels_train.pt")

        torch.save(test_data, f"{self.processed_location}/processed_data_test.pt")
        torch.save(test_labels, f"{self.processed_location}/processed_labels_test.pt")

    def train_dataloader(self):
        train_data = torch.load(f"{self.processed_location}/processed_data_train.pt")
        train_labels = torch.load(f"{self.processed_location}/processed_labels_train.pt")
        train_dataset = TensorDataset(train_data, train_labels)
        return DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    def test_dataloader(self):
        test_data = torch.load(f"{self.processed_location}/processed_data_test.pt")
        test_labels = torch.load(f"{self.processed_location}/processed_labels_test.pt")
        test_dataset = TensorDataset(test_data, test_labels)
        return DataLoader(test_dataset, batch_size=64, shuffle=True)




if __name__ == "__main__":
    # Get the data and process it
    data = DataModule()
    data.process_data()
    train_loader = data.train_dataloader()
    images, labels = next(iter(train_loader))
    # print(dataiter._dataset)
    # images1, labels1 = dataiter.next()
    print(images.shape)

