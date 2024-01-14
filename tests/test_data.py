from mnist_classifier.data.make_dataset import DataModule

def test_data():
    data = DataModule()
    train_dataset = data.get_train_data()
    assert len(train_dataset) == 30000

    test_dataset = data.get_test_data()
    assert len(test_dataset) == 5000