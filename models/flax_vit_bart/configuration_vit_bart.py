import copy

from transformers import BartConfig, ViTConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ViTBartConfig(PretrainedConfig):

    model_type = "vit-bart"
    is_composition = True

    def __init__(self, vit_config_dict, bart_config_dict, **kwargs):
        super().__init__(**kwargs)

        if bart_config_dict is None:
            raise ValueError("`bart_config_dict` can not be `None`.")

        if vit_config_dict is None:
            raise ValueError("`vit_config_dict` can not be `None`.")

        self.bart_config = BartConfig(**bart_config_dict)

        self.vit_config = ViTConfig(**vit_config_dict)

    @classmethod
    def from_vit_bart_configs(
        cls, vit_config: PretrainedConfig, bart_config: PretrainedConfig, **kwargs
    ):
        return cls(
            vit_config_dict=vit_config.to_dict(),
            bart_config_dict=bart_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vit_config"] = self.vit_config.to_dict()
        output["bart_config"] = self.bart_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
