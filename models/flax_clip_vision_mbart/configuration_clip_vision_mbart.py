import copy

from transformers import CLIPVisionConfig, MBartConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CLIPVisionMBartConfig(PretrainedConfig):

    model_type = "clip-vision-mbart"
    is_composition = True
    
    def __init__(self, clip_vision_config_dict, mbart_config_dict, **kwargs):
        super().__init__(**kwargs)

        if mbart_config_dict is None:
            raise ValueError("`mbart_config_dict` can not be `None`.")

        if clip_vision_config_dict is None:
            raise ValueError("`clip_vision_config_dict` can not be `None`.")

        self.mbart_config = MBartConfig(**mbart_config_dict)

        self.clip_vision_config = CLIPVisionConfig(**clip_vision_config_dict)

        self.is_encoder_decoder = True

    @classmethod
    def from_clip_vision_mbart_configs(
        cls,
        clip_vision_config: PretrainedConfig,
        mbart_config: PretrainedConfig,
        **kwargs
    ):
        return cls(
            clip_vision_config_dict=clip_vision_config.to_dict(),
            mbart_config_dict=mbart_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["clip_vision_config"] = self.clip_vision_config.to_dict()
        output["mbart_config"] = self.mbart_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
