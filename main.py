from functools import partial
import gc
import logging
import nltk
import numpy as np
import pandas as pd
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import math
import json
from flax.serialization import to_bytes, from_bytes

import shutil
import torch
from transformers.file_utils import PushToHubMixin
from datasets import load_metric
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, GaussianBlur
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, shard, shard_prng_key, get_metrics, onehot
from models.flax_clip_vision_mbart.modeling_clip_vision_mbart import FlaxCLIPVisionMBartForConditionalGeneration
from transformers import MBart50TokenizerFast, HfArgumentParser, TrainingArguments, is_tensorboard_available, set_seed

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# Cache the result
has_tensorboard = is_tensorboard_available()
if has_tensorboard:
    try:
        from flax.metrics.tensorboard import SummaryWriter
    except ImportError as ie:
        has_tensorboard = False
        print(f"Unable to display metrics through TensorBoard because some package are not installed: {ie}")

else:
    print(
        "Unable to display metrics through TensorBoard because the package is not installed: "
        "Please run pip install tensorboard to enable."
    )

print("TPU cores available:", jax.device_count())

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    vision_model_name_or_path: str = field(
        default = 'openai/clip-vit-base-patch32',
        metadata={
            "help": "The image model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    text_model_name_or_path: str = field(
        default = 'facebook/mbart-large-50',
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    # mbart_from_pt: bool = field(
    #     default=True,
    #     metadata={"help": "whether to load the text using PyTorch checkpoints."},
    # )

    mbart_tokenizer_name: Optional[str] = field(
        default="facebook/mbart-large-50", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # use_fast_tokenizer: bool = field(
    #     default=True,
    #     metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    # )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default="/home/user/data/CC12M/images/",
        metadata={"help": "The data directory containing input files."}
    )
    train_file: Optional[str] = field(
        default=None,  # TODO
        metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,  # TODO
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples_per_lang: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=64,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need both training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension == "tsv", "`train_file` should be a tsv file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "tsv", "`validation_file` should be a tsv file."

class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.transforms = torch.nn.Sequential(
                    Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_size),
                    ConvertImageDtype(torch.float),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


class ImageTextDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        file_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        max_samples: int = None
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.captions = []
        self.image_paths = []
        self.lang = []

        examples = pd.read_csv(file_path, sep='\t')
        gc.collect()

        self.map_lang_code = {
            "en": "en_XX",
            "de": "de_DE",
            "fr": "fr_XX",
            "es": "es_XX",
        }

        for idx,img_file in enumerate(examples["image_file"].values):
            if os.path.exists(os.path.join(self.root,img_file)):
                self.image_paths.append(img_file)
                self.captions.append(examples["caption"].values[idx])
                self.lang.append(examples["lang_id"].values[idx])


        if max_samples is None:
            max_samples = len(self.image_paths)

        self.image_paths = self.image_paths[:max_samples]
        self.captions = self.captions[:max_samples]
        self.lang = self.lang[:max_samples]


    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(os.path.join(self.root,path), mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def _load_lang(self, idx):
        return self.lang[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)
        lang = self._load_lang(index)
        lang = self.map_lang_code[lang]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, lang

    def __len__(self) -> int:
        return len(self.captions)

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))



def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train/{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    writable_eval_metrics = {}
    for key,value in eval_metrics.items():
        if isinstance(value,dict):
            for sub_key,sub_value in value.items():
                writable_eval_metrics[sub_key+"/"+key] = sub_value
        else:
            writable_eval_metrics[key]=value

    for metric_name, value in writable_eval_metrics.items():
        if metric_name =="loss":
            summary_writer.scalar(f"eval_{metric_name}", value, step)
        else:
            summary_writer.scalar(f"{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

# utils
def mb_item(x):
    return x.item() if hasattr(x, "item") else x

#checkpoint functions
def save_model_checkpoint(model, save_dir, state, logger, organization,  with_opt:bool=False, push_to_hub:bool=False, overwrite=False, **kwargs):
    state = jax_utils.unreplicate(state)
    gc.collect()
    logger.info(f"Saving Checkpoint in {save_dir}")
    ckpt_save_dir = f"{save_dir}/ckpt-{mb_item(state.step)-1}"
    if os.path.exists(ckpt_save_dir) and not overwrite:
        logger.info("checkpoint exists, skipping overwrite")
    else:
        model.save_pretrained(
            ckpt_save_dir,
            params=state.params,
            push_to_hub=False,
            **kwargs
        )
        if with_opt:
            with open(os.path.join(ckpt_save_dir, "opt_state.msgpack"), "wb") as f:
                f.write(to_bytes(state.opt_state))
            with open(os.path.join(ckpt_save_dir, "training_state.json"), "w") as f:
                json.dump({"step": state.step.item()}, f)

        logger.info("checkpoint saved")
        gc.collect()

        if push_to_hub:
            repo_name = Path(save_dir).name
            repo_url = PushToHubMixin._get_repo_url_from_name(repo_name, organization=organization, private=False, use_auth_token=True)
            repo = PushToHubMixin._create_or_get_repo(save_dir, repo_url = repo_url, organization=organization, use_auth_token=True)
            commit_message=f"Saving weights and logs at step {mb_item(state.step)-1}"
            url = PushToHubMixin._push_to_hub(repo = repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")



def restore_model_checkpoint(save_dir, state, logger):
    logger.info(f"Restoring checkpoint from {save_dir}.")
    with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
        params = from_bytes(state.params, f.read())

    with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
        opt_state = from_bytes(state.opt_state, f.read())

    with open(os.path.join(save_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    step = training_state["step"]

    logger.info("checkpoint restored")
    #return state.replace(step=step, params=params, opt_state=opt_state), step
    return params, opt_state, step

def rotate_checkpoints(ckpt_dir:str, save_total_limit:int, logger):
    "Removes older checkpoints so that `save_total_limit` checkpoints are kept"
    # TODO: what to remove is decided using step number only, we might want to improve that
    ckpts = [str(x) for x in Path(ckpt_dir).glob("ckpt-*")]
    # sort checkpoints by step
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('-')[-1]))
    ckpts_to_delete = ckpts_sorted[:-save_total_limit]
    for ckpt in ckpts_to_delete:
        logger.info(f"Deleting older checkpoint [{ckpt}] due to save_total_limit ({save_total_limit})")
        shutil.rmtree(ckpt)

# In Flax, for seq2seq models we need to pass `decoder_input_ids`
# as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
# for that dynamically import the `shift_tokens_right` function from the model file
def shift_tokens_right(input_ids: np.array, pad_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros(input_ids.shape, dtype=np.int64)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = pad_token_id
    return shifted_input_ids


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    tokenizer = MBart50TokenizerFast.from_pretrained(model_args.mbart_tokenizer_name)

    map_lang_num = {
        "en_XX": 0,
        "de_DE": 1,
        "fr_XX": 2,
        "es_XX": 3,
    }

    map_bart_nltk = {
        "en_XX": "english",
        "de_DE": "german",
        "fr_XX": "french",
        "es_XX": "spanish",
    }

    if training_args.resume_from_checkpoint is None:
        model = FlaxCLIPVisionMBartForConditionalGeneration.from_clip_vision_mbart_pretrained(
            model_args.vision_model_name_or_path,
            model_args.text_model_name_or_path,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            mbart_from_pt=True
        )
    else:
        model = FlaxCLIPVisionMBartForConditionalGeneration.from_pretrained(training_args.resume_from_checkpoint)
    config = model.config
    # config = vision_model_name_or_path.config
    # set seed for torch dataloaders
    set_seed(training_args.seed)

    logger.info(f"Creating and jitting subscriptable transform")

    # Initialize torchvision transforms and jit them for faster processing
    preprocess = Transform(config.clip_vision_config.image_size)
    preprocess = torch.jit.script(preprocess)

    logger.info(f"Creating train_dataset from ImageTextDataset")

    # Initialize the image-text dataset
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        transform=preprocess,
        max_samples = data_args.max_train_samples
    )

    _df = pd.read_csv(data_args.validation_file, delimiter="\t", index_col=False)
    gc.collect()
    lang_list = ["en", "fr", "es", "de"]

    logger.info(f"Splitting validations TSVs")

    for i in lang_list:  # splits validation file into 4 subsets
        subset_lang_tsv = _df[_df["lang_id"]==i]
        subset_lang_tsv.reset_index(drop=True, inplace=True)
        path = os.path.join(os.path.dirname(data_args.validation_file), f"{i}_"+os.path.basename(data_args.validation_file))
        subset_lang_tsv.to_csv(path, index=False, sep="\t")

    val_paths = []
    for i in lang_list:
        val_paths.append(os.path.join(os.path.dirname(data_args.validation_file), f"{i}_"+os.path.basename(data_args.validation_file)))

    logger.info(f"creating eval dataset from ImageTextDataset")
    # gc.collect()

    eval_dataset = []
    for i in range(len(lang_list)):
        dataset = ImageTextDataset(
            data_args.data_dir,
            val_paths[i],
            transform=preprocess,
            max_samples=data_args.max_eval_samples_per_lang
        )
        eval_dataset.append(dataset)

    gc.collect()

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    logger.info(f"initialising shift tokens right from model")


    # Use collate function to tokenizer the text and convert the processed images to numpy
    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples]).permute(0, 2, 3, 1).numpy()
        captions = [example[1] for example in examples]
        lang_id = [example[2] for example in examples]

        # inputs = helper_collate(lang_id, captions)
        inputs = {}

        for num, (lang,caption) in enumerate(zip(lang_id,captions)):
            # tokenizer = map_tokenizer_lang[lang]
            tokenizer.tgt_lang = lang
            with tokenizer.as_target_tokenizer():
                tokens = tokenizer(str(caption), max_length=data_args.max_seq_length, padding="max_length", return_tensors="np", truncation=True)
            if num==0:
                inputs["input_ids"] = tokens["input_ids"]
                inputs["attention_mask"] = tokens["attention_mask"]
            else:
                inputs["input_ids"] = np.concatenate([inputs["input_ids"], tokens["input_ids"]])
                inputs["attention_mask"] = np.concatenate([inputs["attention_mask"], tokens["attention_mask"]])


        decoder_input_ids = shift_tokens_right(inputs["input_ids"], config.mbart_config.pad_token_id)

        batch = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }

        return batch

    def collate_fn_val(examples):
        pixel_values = torch.stack([example[0] for example in examples]).permute(0, 2, 3, 1).numpy()
        captions = [example[1] for example in examples]
        lang_id = [example[2] for example in examples]

        # tokenizer = map_tokenizer_lang[lang_id[0]]
        tokenizer.tgt_lang = lang_id[0]  # every validation loader has same language
        with tokenizer.as_target_tokenizer():
            tokens = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", return_tensors="np", truncation=True)

        decoder_input_ids = shift_tokens_right(tokens["input_ids"], config.mbart_config.pad_token_id)

        batch = {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }
        return batch

    logger.info(f"Creating train data loader")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        # persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Creating eval data loader")

    eval_loader = []
    for i in range(len(lang_list)):
        loader = torch.utils.data.DataLoader(
            eval_dataset[i],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=data_args.preprocessing_num_workers,
            # persistent_workers=True,
            drop_last=True,
            collate_fn=collate_fn_val,
        )
        eval_loader.append(loader)

    # Metric
    metric = load_metric("bleu")
    gc.collect()

    def postprocess_text(preds, labels, lang):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = [nltk.word_tokenize(pred, language=lang) for pred in preds]

        # put in another list as seen https://github.com/huggingface/datasets/blob/256156b29ce2f4bb1ccedce0638491e440b0d1a2/metrics/bleu/bleu.py#L82
        labels = [[nltk.word_tokenize(label, language=lang)] for label in labels]

        gc.collect()
        return preds, labels

    def compute_metrics(preds, labels, lang):

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, max_length=64)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, max_length=64)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels, map_bart_nltk[lang])

        result = {}
        for i in range(1,5):
            tmp = metric.compute(predictions=decoded_preds, references=decoded_labels, max_order=i)
            result[f"BLEU-{i}"] = tmp["bleu"]

        gc.collect()
        return result

    # Enable tensorboard only on the master node
    if has_tensorboard and jax.process_index() == 0:
        summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir).joinpath("logs").as_posix())

    # # Initialize our training
    # rng = jax.random.PRNGKey(training_args.seed)
    # # rng, dropout_rng = jax.random.split(rng)
    # dropout_rngs = jax.random.split(rng, jax.local_device_count())

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)


    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)
    # if training_args.resume_from_checkpoint is None:
    #     state = train_state.TrainState.create(
    #         apply_fn=model.__call__, params=model.params, tx=adamw
    #     )
    # else:
    #     state = train_state.TrainState.create(
    #         apply_fn=model.__call__, params=model.params, tx=adamw
    #     )
    #     params, opt_state, step = restore_model_checkpoint(training_args.resume_from_checkpoint, state, logger)
    #     state = state.replace(
    #         step=step,
    #         apply_fn=model.__call__,
    #         params=params,
    #         tx=adamw,
    #         opt_state=opt_state,
    #     )


    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """

        vocab_size = logits.shape[-1]

        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )

        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        return loss

     # Define gradient update step fn
    # def train_step(state, batch, dropout_rng, label_smoothing_factor=0.0):
    def train_step(state, batch, label_smoothing_factor=0.0):
        # dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("input_ids")
            # masks = batch.pop("attention_mask")
            # labels = batch["input_ids"],
            logits = state.apply_fn(batch["pixel_values"], batch["decoder_input_ids"], batch["attention_mask"], params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels, batch["attention_mask"], label_smoothing_factor)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        # new_state = state.apply_gradients(grads=grad)
        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        gc.collect()

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch["input_ids"]
        # labels = batch.pop("input_ids")
        # masks = batch.pop("attention_mask")
        logits = model(batch["pixel_values"], batch["decoder_input_ids"], batch["attention_mask"], params=params, train=False)[0]
        loss = loss_fn(logits, labels, batch["attention_mask"], label_smoothing_factor)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        gc.collect()
        return metrics

    num_beams = 4  # model has beam size 5, should we keep 4 or 5 here?
    gen_kwargs = {"max_length": data_args.max_seq_length, "num_beams": num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["pixel_values"], **gen_kwargs)
        return output_ids.sequences

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(partial(train_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch", donate_argnums=(0,1,2,))

    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()
    # state = jax_utils.replicate(state)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")
    if training_args.resume_from_checkpoint is not None:
        previous_step = int(jax_utils.unreplicate(state.step))
        epoch_start_point = math.ceil((previous_step*train_batch_size)/len(train_dataset))
    else:
        epoch_start_point = 0

    break_all = False
    train_time = 0
    epochs = tqdm(range(epoch_start_point, num_epochs), desc=f"Epoch:  ({epoch_start_point+1}/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        num_train_samples = len(train_dataset)

        epochs.desc = f"Epoch:  ({epoch+1}/{num_epochs})"

        steps_per_epoch = len(train_dataset) // train_batch_size

        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for step, batch in enumerate(train_loader):
            batch = shard(batch)
            # state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            cur_step = epoch * (num_train_samples // train_batch_size) + step + 1

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
                epochs.write(f"Log at Step: {cur_step} (Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})")

                train_metrics = [] # TODO: Check why is this being done? WHat is this needed for?

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:

                eval_metrics = []
                bleu_metrics_total = {}

                # TODO: Check if this is correct
                eval_steps = len(eval_dataset[0])*4 // eval_batch_size  # eval_dataset is a list containing loaders for diff langs

                eval_step_progress_bar = tqdm(total=eval_steps, desc="Evaluating: ", position=2, leave=False)
                for val, lang_eval_loader in enumerate(eval_loader):

                    eval_preds = []
                    eval_labels = []
                    li = ["en_XX", "fr_XX", "es_XX", "de_DE"]
                    curr_lang = li[val]

                    for batch in lang_eval_loader:
                        # Model forward
                        # lang = batch["lang"]
                        batch = shard(batch)
                        labels = batch["input_ids"] # TODO: Check if this works correctly since this is sharded
                        # print(labels.shape)

                        metrics = p_eval_step(state.params, batch)
                        eval_metrics.append(metrics)

                        # curr_lang = list(map_lang_num.keys())[lang[0]] # TODO: Check if we can directly replace with lists?
                        # generation
                        if data_args.predict_with_generate:
                            gen_kwargs.update({"decoder_start_token_id": tokenizer.lang_code_to_id[curr_lang]})
                            generated_ids = p_generate_step(state.params, batch)
                            # print("generated_ids:", generated_ids)
                            eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                            eval_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))

                        eval_step_progress_bar.update(1)


                    # compute BLEU metrics
                    if data_args.predict_with_generate:
                        bleu_metrics = compute_metrics(eval_preds, eval_labels, curr_lang)  # eval_langs would contain one lang only
                        bleu_metrics_total[curr_lang] = bleu_metrics
                        gc.collect()

                    gc.collect()

                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                eval_metrics.update(bleu_metrics_total)
                bleu_desc = " ".join([f"BLEU score {key}: {value} |" for key, value in bleu_metrics_total.items()])

                # Print metrics and update progress bar
                eval_step_progress_bar.close()
                epochs.write(f"Eval at Step: {cur_step} (Eval Loss: {eval_metrics['loss']} | {bleu_desc})")
                # epochs.write(f"Eval at Step: {cur_step} (Eval Loss: {eval_metrics['loss']})")


                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

                eval_metrics = []

            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    # params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                    # model.save_pretrained(
                    #     training_args.output_dir,
                    #     params=params,
                    #     push_to_hub=training_args.push_to_hub,
                    #     commit_message=f"Saving weights and logs of step {cur_step}",
                    # )
                    save_model_checkpoint(model, training_args.output_dir, state, logger, training_args.push_to_hub_organization, with_opt=True, push_to_hub=training_args.push_to_hub, overwrite=True)
                    gc.collect()
                    # if model_args.save_optimizer:
                    #     # this saves full state including optimizer
                    #     save_checkpoint(training_args.output_dir, state, state.step, keep=training_args.save_total_limit, overwrite=True)
                    if training_args.save_total_limit is not None:
                        rotate_checkpoints(training_args.output_dir, training_args.save_total_limit, logger)
                        # gc.collect()


            if cur_step==total_train_steps:
                break_all=True
                break

        train_step_progress_bar.close()
        epochs.update(1)
        gc.collect()

        if break_all:
            break

if __name__ == "__main__":
    main()