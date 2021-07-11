import csv
from functools import partial
import logging
import numpy as np
import pandas as pd
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, GaussianBlur
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, shard, shard_prng_key, get_metrics, onehot
from model.modeling_clip_vision_mbart import FlaxCLIPVisionMBartForConditionalGeneration
from transformers import MBart50TokenizerFast, HfArgumentParser, TrainingArguments, is_tensorboard_available, set_seed


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
    from_pt: bool = field(
        default=False,
        metadata={"help": "whether to load the text and vision model using PyTorch checkpoints."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
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
        default="/home/user/data/CC12M/val_file.tsv",  # TODO
        metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default="/home/user/data/CC12M/val_file.tsv",  # TODO
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
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
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension == "json", "`train_file` should be a json file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension == "json", "`validation_file` should be a json file."

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
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.captions = []
        self.image_paths = []
        self.lang = []

        examples = pd.read_csv(file_path, sep='\t')

        self.image_paths = examples["image_file"].values
        self.captions = examples["caption"].values
        self.lang = examples["lang_id"].values

        # with open(file_path, encoding="utf-8") as fd:
        #     examples = csv.DictReader(fd, delimiter="\t", quotechar='"')
        #     for row in examples:
        #         self.image_paths.append(os.path.join(self.root,row["image_file"]))
        #         self.captions.append(row["caption"])
        #         self.lang.append(row["lang_id"])


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

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, lang

    def __len__(self) -> int:
        return len(self.captions)

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)

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

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # ToDo
    # 1. output_dir
    # 2. upload_to_hub
    # 3. check others

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

    tokenizer_en = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")
    tokenizer_de = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="de_DE")
    tokenizer_fr = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="fr_XX")
    tokenizer_es = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="es_XX")

    map_tokenizer_lang = {
        "en": tokenizer_en,
        "de": tokenizer_de,
        "fr": tokenizer_fr,
        "es": tokenizer_es,
    }

    # if model_args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    #     )
    # elif model_args.text_model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_args.text_model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    #     )
    # else:
    #     raise ValueError(
    #         "You are instantiating a new tokenizer from scratch. This is not supported by this script."
    #         "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    #     )

    model = FlaxCLIPVisionMBartForConditionalGeneration.from_clip_vision_mbart_pretrained(
        model_args.vision_model_name_or_path,
        model_args.text_model_name_or_path,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
        mbart_from_pt=True
    )
    config = model.config
    # config = vision_model_name_or_path.config
    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize torchvision transforms and jit them for faster processing
    preprocess = Transform(config.clip_vision_config.image_size)
    preprocess = torch.jit.script(preprocess)

    # Initialize the image-text dataset
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        transform=preprocess,
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        transform=preprocess,
    )
    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    def helper_colate(lang_id, captions):
        inputs = {}

        for num, (lang,caption) in enumerate(zip(lang_id,captions)):
            tokenizer = map_tokenizer_lang[lang]
            with tokenizer.as_target_tokenizer():
                tokens = tokenizer(caption, max_length=data_args.max_seq_length, padding="max_length", return_tensors="np", truncation=True)
            if num==0:
                inputs["input_ids"] = tokens["input_ids"]
                inputs["attention_mask"] = tokens["attention_mask"]
            else:
                inputs["input_ids"] = np.concatenate([inputs["input_ids"], tokens["input_ids"]])
                inputs["attention_mask"] = np.concatenate([inputs["attention_mask"], tokens["attention_mask"]])

        return inputs


    # Use collate function to tokenizer the text and convert the processed images to numpy
    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples]).permute(0, 2, 3, 1).numpy()
        captions = [example[1] for example in examples]
        lang_id = [example[2] for example in examples]

        inputs = helper_colate(lang_id, captions)

        batch = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            # "lang": lang_id,
            # "caption": captions,
        }

        return batch

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # print(next(iter(eval_loader)))

    # Enable tensorboard only on the master node
    if has_tensorboard and jax.process_index() == 0:
        summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir).joinpath("logs").as_posix())

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

    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        # print("BLAHHHH")
        # print(labels[0])
        # print(labels[0].shape)
        # print("logits shape:", logits.shape)
        vocab_size = logits.shape[-1]

        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        # print("logits shape:", logits.shape)
        # print("$$$$$$$")
        # print("labels:", len(labels))
        soft_labels = onehot(labels[0], vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        # print("soft_labels shape:", soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        return loss

     # Define gradient update step fn
    def train_step(state, batch, label_smoothing_factor=0.0):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            # labels = batch.pop("labels")
            # masks = batch.pop("attention_mask")
            labels = batch["input_ids"],
            logits = state.apply_fn(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels, batch["attention_mask"], label_smoothing_factor)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch["input_ids"]
        # masks = batch.pop("attention_mask")
        logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], params=params, train=False)[0]
        loss = loss_fn(logits, labels, batch["attention_mask"], label_smoothing_factor)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # num_beams = data_args.num_beams
    # gen_kwargs = {"max_length": data_args.max_seq_length, "num_beams": num_beams}

    # def generate_step(params, batch):
    #     model.params = params
    #     output_ids = model.generate(batch["pixel_values"], attention_mask=batch["attention_mask"], **gen_kwargs)
    #     return output_ids.sequences

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(partial(train_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")


    # Replicate the train state on each device
    state = state.replicate()
    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch")
    # p_generate_step = jax.pmap(generate_step, "batch")


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    # Create sampling rng
    rng, input_rng = jax.random.split(rng)

    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        steps_per_epoch = len(train_dataset) // train_batch_size

        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_loader:
            batch = shard(batch)
            # batch = shard(next(iter(batch)))
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

        train_time += time.time() - train_start

        train_metric = unreplicate(train_metric)

        train_step_progress_bar.close()

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
        )

        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_steps = len(eval_dataset) // eval_batch_size
        eval_step_progress_bar = tqdm(total=eval_steps, desc="Evaluating...", position=2, leave=False)
        for batch in eval_loader:
            # Model forward
            batch = shard(batch)
            metrics = p_eval_step(state.params, batch)
            eval_metrics.append(metrics)

            eval_step_progress_bar.update(1)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)

        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # Print metrics and update progress bar
        eval_step_progress_bar.close()
        desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']})"
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            cur_step = epoch * (len(train_dataset) // train_batch_size)
            write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step)



if __name__ == "__main__":
    main()