import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.make_dataset import DataModule
from models.lightning_model import MyAwesomeModel

model_config = OmegaConf.load('mnist_classifier/models/conf/experiment/exp1.yaml')
dimensions = model_config.dimensions

model = MyAwesomeModel(
    dimensions.input_dim,
    dimensions.first_hidden_dim,
    dimensions.second_hidden_dim,
    dimensions.third_hidden_dim,
    dimensions.output_dim,
)

checkpoint_callback = ModelCheckpoint(dirpath='./models', monitor='train_loss', mode='min')

trainloader = DataModule().train_dataloader()
trainer = Trainer(limit_train_batches=0.2, max_epochs=5, logger=pl.loggers.WandbLogger(project='mnist_classifier'), accelerator='cpu')
trainer.fit(model, trainloader)
