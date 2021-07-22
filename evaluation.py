## Don't forget to run tqdm before starting the script

import argparse
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import jax
import matplotlib.pyplot as plt
import nltk
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_metric
from collections import Counter

from models.flax_clip_vision_mbart.modeling_clip_vision_mbart import FlaxCLIPVisionMBartForConditionalGeneration

from transformers import MBart50TokenizerFast


from torchvision.io import read_image, ImageReadMode
import torch
import numpy as np
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--tsv_path", type=str, default="/home/bhavitvya_malik/final_data/data/val_file_marian_final.tsv", help="path of directory where the dataset is stored")
parser.add_argument("--model_weights", type=str, help="Path to your model weights")

args = parser.parse_args()


class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x

transform = Transform(224)
metric = load_metric("bleu")

def get_transformed_image(image):
    if image.shape[-1] == 3 and isinstance(image, np.ndarray):
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
    return transform(image).unsqueeze(0).permute(0, 2, 3, 1).numpy()


DATASET_PATH = args.tsv_path

BATCH_SIZE = 512
lang_dict = {
    "en" : "en_XX",
    "es" : "es_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    # "ru": "ru_RU"  # removed Russian after Patrick's suggestions
}

root_dir = "/home/user/data/CC12M/images"
model = FlaxCLIPVisionMBartForConditionalGeneration.from_pretrained(args.model_weights)  # path to your model
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")

p_params = replicate(model.params)

def generateen_XX(params, batch):
    output_ids = model.generate(batch["pixel_values"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"], num_beams=4, max_length=64).sequences
    return output_ids

def generatefr_XX(params, batch):
      output_ids = model.generate(batch["pixel_values"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"], num_beams=4, max_length=64).sequences
      return output_ids

def generatees_XX(params, batch):
      output_ids = model.generate(batch["pixel_values"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"], num_beams=4, max_length=64).sequences
      return output_ids

def generatede_DE(params, batch):
      output_ids = model.generate(batch["pixel_values"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"], num_beams=4, max_length=64).sequences
      return output_ids


p_generate_en_XX = jax.pmap(generateen_XX, "batch")
p_generate_fr_XX = jax.pmap(generatefr_XX, "batch")
p_generate_es_XX = jax.pmap(generatees_XX, "batch")
p_generate_de_DE = jax.pmap(generatede_DE, "batch")

map_name = {
    "en_XX": p_generate_en_XX,
    "fr_XX": p_generate_fr_XX,
    "es_XX": p_generate_es_XX,
    "de_DE": p_generate_de_DE,
}

map_bart_nltk = {
        "en": "english",
        "de": "german",
        "fr": "french",
        "es": "spanish",
    }

def run_generate(input_str, p_generate):
    # inputs = tokenizer(input_str, return_tensors="np", padding="max_length", truncation=True, max_length=64)
    inputs = {}
    for q,file in enumerate(input_str):
        path = os.path.join(root_dir,file)
        image = plt.imread(path)
        transformed_image = get_transformed_image(image)
        if q==0:
            inputs["pixel_values"] = transformed_image
        else:
            inputs["pixel_values"] = np.concatenate([inputs["pixel_values"],transformed_image])


    # print("inputs[pixel_values] shape: ", inputs["pixel_values"].shape)
    p_inputs = shard(inputs)
    output_ids = p_generate(p_params, p_inputs)
    # print(output_ids)
    output_strings = tokenizer.batch_decode(output_ids.reshape(-1, 64), skip_special_tokens=True, max_length=64)
    # print(output_strings)
    return output_strings

def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", index_col=False)
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    return df

def postprocess_text(preds, labels, lang):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = [nltk.word_tokenize(pred, language=lang) for pred in preds]

        # put in another list as seen https://github.com/huggingface/datasets/blob/256156b29ce2f4bb1ccedce0638491e440b0d1a2/metrics/bleu/bleu.py#L82
        labels = [[nltk.word_tokenize(label, language=lang)] for label in labels]

        gc.collect()
        return preds, labels

def compute_metrics(preds, labels, lang):
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(preds, labels, map_bart_nltk[lang])

        result = {}
        for i in range(1,5):
            tmp = metric.compute(predictions=decoded_preds, references=decoded_labels, max_order=i)
            result[f"BLEU-{i}"] = tmp["bleu"]

        gc.collect()
        return result

def arrange_data(image_files, captions, lang):  # iterates through all the captions and save there translations
    try:
        p_generate = map_name[lang_dict[lang]]
        output = run_generate(image_files, p_generate)
        bleu_output = compute_metrics(output,captions,lang)

        return bleu_output

    except Exception as e:
        print(e, image_files, " skipped!")
        return


df = read_tsv_file(DATASET_PATH)

langs = ["en", "es", "de", "fr"]

bleu_metrics_total = {}

for j in langs:
    new_df = df[df["lang_id"]==j]
    tmp_dict = []
    sub_lan_dict = {}
    for i in tqdm(range(0,len(new_df),BATCH_SIZE)):
        output_batch = arrange_data(list(new_df["image_file"])[i:i+BATCH_SIZE], list(new_df["caption"])[i:i+BATCH_SIZE], j)
        tmp_dict.append(output_batch)
    df = pd.DataFrame(tmp_dict)
    answer = dict(df.mean())
    bleu_metrics_total[j] = answer

print(bleu_metrics_total)