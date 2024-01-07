import click
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--model", default="mnist_classifier/models/trained_models/model.pt", help="Model location")
@click.option("--n_images", default=10, help="Number of images to predict")
def predict(model, n_images):
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    model = torch.load(model)
    test_data = torch.load("data/processed/processed_data_test.pt")[: int(n_images)]
    test_labels = torch.load("data/processed/processed_labels_test.pt")[: int(n_images)]

    test_set = torch.utils.data.TensorDataset(test_data, test_labels)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            pred_labels = model.forward(images)

            test_preds.append(pred_labels.argmax(dim=1).cpu())
            test_labels.append(labels.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    print(test_preds, test_labels)
    return test_preds, test_labels


# cli.add_command(predict)

if __name__ == "__main__":
    # cli()
    test_preds, test_labels = predict()
