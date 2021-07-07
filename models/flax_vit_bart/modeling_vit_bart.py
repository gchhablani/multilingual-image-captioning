from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, unfreeze
from jax import lax
from jax.random import PRNGKey
from transformers import BartConfig, FlaxBartModel, FlaxViTModel, ViTConfig
from transformers.modeling_flax_outputs import (
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from transformers.models.bart.modeling_flax_bart import (
    FlaxBartDecoder,
    FlaxPreTrainedModel,
    shift_tokens_right,
)
from transformers.models.vit.modeling_flax_vit import FlaxViTModule

from .configuration_vit_bart import ViTBartConfig


class FlaxViTBartModule(nn.Module):
    config: ViTBartConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.shared = nn.Embed(
            self.config.bart_config.vocab_size,
            self.config.bart_config.d_model,
            embedding_init=jax.nn.initializers.normal(
                self.config.bart_config.init_std, self.dtype
            ),
            dtype=self.dtype,
        )

        self.encoder = FlaxViTModule(self.config.vit_config, dtype=self.dtype)
        self.decoder = FlaxBartDecoder(
            self.config.bart_config, dtype=self.dtype, embed_tokens=self.shared
        )

        self.visual_projection = nn.Dense(
            self.config.bart_config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.bart_config.init_std, self.dtype
            ),
        )

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
        self,
        pixel_values,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        batch_size, sequence_length = encoder_outputs[0].shape[:2]
        encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        encoder_hidden_states = self.visual_projection(encoder_outputs[0])

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class FlaxViTBartForConditionalGenerationModule(nn.Module):
    config: ViTBartConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.model = FlaxViTBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.model.shared.num_embeddings,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.bart_config.init_std, self.dtype
            ),
        )
        self.final_logits_bias = self.param(
            "final_logits_bias", self.bias_init, (1, self.model.shared.num_embeddings)
        )

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        pixel_values,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables["params"]["shared"]["embedding"]
            lm_logits = self.lm_head.apply(
                {"params": {"kernel": shared_embedding.T}}, hidden_states
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits += self.final_logits_bias

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class FlaxViTBartPreTrainedModel(FlaxPreTrainedModel):
    config_class = ViTBartConfig
    base_model_prefix: str = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ViTBartConfig,
        input_shape: Tuple = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (
                (1, config.vit_config.image_size, config.vit_config.image_size, 3),
                (1, 1),
            )

        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config, module, input_shape=input_shape, seed=seed, dtype=dtype
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jax.random.normal(rng, input_shape[0])
        # # make sure initialization pass will work for FlaxBartForSequenceClassificationModule
        # input_ids = jax.ops.index_update(input_ids, (..., -1), self.config.eos_token_id)

        decoder_input_ids = jnp.zeros(input_shape[1], dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(
            rngs,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
        )["params"]

    def init_cache(self, batch_size, max_length, encoder_outputs):

        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]),
            decoder_input_ids.shape,
        )

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    def encode(
        self,
        pixel_values: jnp.ndarray,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, pixel_values, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(pixel_values, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            pixel_values=jnp.array(pixel_values, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `decoder_position_ids` when passing `past_key_values`."
                )

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxBartAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # # prepare encoder inputs
        # if attention_mask is None:
        #     attention_mask = jnp.ones_like(input_ids)
        # if position_ids is None:
        #     batch_size, sequence_length = input_ids.shape
        #     position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # prepare decoder inputs
        # if decoder_input_ids is None:
        #     decoder_input_ids = shift_tokens_right(
        #         input_ids, self.config.pad_token_id, decoder_start_token_id=self.config.decoder_start_token_id
        #     ) # TODO: Check how to use this
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        if decoder_position_ids is None:
            batch_size, sequence_length = decoder_input_ids.shape
            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            pixel_values=jnp.array(pixel_values, dtype=jnp.float32),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


class FlaxViTBartForConditionalGeneration(FlaxViTBartPreTrainedModel):
    module_class = FlaxViTBartForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        deterministic: bool = True,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `decoder_position_ids` when passing `past_key_values`."
                )

            decoder_position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxBartAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if self.config.tie_word_embeddings:
                shared_embedding = module.model.variables["params"]["shared"][
                    "embedding"
                ]
                lm_logits = module.lm_head.apply(
                    {"params": {"kernel": shared_embedding.T}}, hidden_states
                )
            else:
                lm_logits = module.lm_head(hidden_states)

            lm_logits += module.final_logits_bias
            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=jnp.array(encoder_attention_mask, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jnp.DeviceArray] = None,
        decoder_attention_mask: Optional[jnp.DeviceArray] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(
                extended_attention_mask, decoder_attention_mask, (0, 0)
            )
        else:
            position_ids = jnp.broadcast_to(
                jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
            )

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = (
            model_kwargs["decoder_position_ids"][:, -1:] + 1
        )
        return model_kwargs

    @classmethod
    def from_vit_bart_pretrained(
        cls,
        vit_model_name_or_path: str = None,
        bart_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> FlaxViTBartPreTrainedModel:

        kwargs_bart = {
            argument[len("bart_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("bart_")
        }

        kwargs_vit = {
            argument[len("vit_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vit_")
        }

        # remove bart, vit kwargs from kwargs
        for key in kwargs_bart.keys():
            del kwargs["bart_" + key]
        for key in kwargs_vit.keys():
            del kwargs["vit_" + key]

        # Load and initialize the bart and vit model
        bart_model = kwargs_bart.pop("model", None)
        if bart_model is None:
            assert (
                bart_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `bart_model_name_or_path` has to be defined"

            if "config" not in kwargs_bart:
                bart_config = BartConfig.from_pretrained(bart_model_name_or_path)
                kwargs_bart["config"] = bart_config

            bart_model = FlaxBartModel.from_pretrained(
                bart_model_name_or_path, *model_args, **kwargs_bart
            )

        vit_model = kwargs_vit.pop("model", None)
        if vit_model is None:
            assert (
                vit_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `vit_model_name_or_path` has to be defined"

            if "config" not in kwargs_vit:
                vit_config = ViTConfig.from_pretrained(vit_model_name_or_path)
                kwargs_vit["config"] = vit_config

            vit_model = FlaxViTModel.from_pretrained(
                vit_model_name_or_path, *model_args, **kwargs_vit
            )

        # instantiate config with corresponding kwargs
        dtype = kwargs.pop("dtype", jnp.float32)
        config = ViTBartConfig.from_vit_bart_configs(
            vit_model.config, bart_model.config, **kwargs
        )

        # init model
        model = cls(config, *model_args, dtype=dtype, **kwargs)
        model.params["model"]["encoder"] = vit_model.params
        model.params["model"]["decoder"] = bart_model.params["decoder"]
        model.params["model"]["shared"] = bart_model.params["shared"]
        # model.params["bart_model"] = bart_model.params

        return model


# flax_vit_bart_cg = FlaxViTBartForConditionalGeneration.from_vit_bart_pretrained('google/vit-base-patch16-224-in21k', 'facebook/bart-large')
# outputs = flax_vit_bart_cg(pixel_values, input_ids, attention_mask, position_ids, output_hidden_states=True)
