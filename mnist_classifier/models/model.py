import logging

import hydra
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

log = logging.getLogger(__name__)


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, input_dim, first_hidden_dim, second_hidden_dim, third_hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, first_hidden_dim)
        self.fc2 = nn.Linear(first_hidden_dim, second_hidden_dim)
        self.fc3 = nn.Linear(second_hidden_dim, third_hidden_dim)
        self.fc4 = nn.Linear(third_hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# hydra.core.global_hydra.GlobalHydra.instance().clear()
# @hydra.main(config_path="conf", config_name="default_model_conf.yaml")
# def define_model(cfg: DictConfig):
#     # Instantiate the model with dimensions from the config
#     dimensions = cfg.experiment.dimensions

#     model = MyAwesomeModel(
#         input_dim=dimensions.input_dim,
#         first_hidden_dim=dimensions.first_hidden_dim,
#         second_hidden_dim=dimensions.second_hidden_dim,
#         third_hidden_dim=dimensions.third_hidden_dim,
#         output_dim=dimensions.output_dim
#     )

#     log.info("Input dim:", dimensions.input_dim)

#     return model


# if __name__ == "__main__":
#     define_model()
