## Don't forget to run tqdm before starting the script

import argparse
import csv
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import itertools
import jax
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import FlaxMBartForConditionalGeneration, MBart50TokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument("--tsv_path", type=str, default="images-list-clean.tsv", help="path of directory where the dataset is stored")
parser.add_argument("--val_split", type=int, default=0.1, help="Size of validation Subset")
parser.add_argument("--lang_list", nargs="+", default=["fr", "de", "es", "ru"], help="Language list (apart from English)")
parser.add_argument("--save_location_train", type=str, default=".", help="path of directory where the train dataset will be stored")
parser.add_argument("--save_location_val", type=str, default=".", help="path of directory where the validation dataset will be stored")
parser.add_argument("--is_train", type=int, default=0, help="train or validate")

args = parser.parse_args()

DATASET_PATH = args.tsv_path
VAL_SPLIT = args.val_split
LANG_LIST = args.lang_list
if args.save_location_train != None:
    SAVE_TRAIN = args.save_location_train
    SAVE_VAL = args.save_location_val

BATCH_SIZE = 1024
IS_TRAIN = args.is_train
num_devices = 8
lang_dict = {
    "es" : "es_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    # "ru": "ru_RU"  # removed Russian after Patrick's suggestions
}

model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

p_params = replicate(model.params)

def generatefr_XX(params, batch):
      output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"], num_beams=4, max_length=64).sequences
      return output_ids

def generatees_XX(params, batch):
      output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"], num_beams=4, max_length=64).sequences
      return output_ids

def generatede_DE(params, batch):
      output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"], num_beams=4, max_length=64).sequences
      return output_ids

# def generateru_RU(params, batch, rng):
#       output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], prng_key=rng, params=params, forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"]).sequences
#       return output_ids

p_generate_fr_XX = jax.pmap(generatefr_XX, "batch")
p_generate_es_XX = jax.pmap(generatees_XX, "batch")
p_generate_de_DE = jax.pmap(generatede_DE, "batch")
# p_generate_ru_RU = jax.pmap(generateru_RU, "batch")

map_name = {
    "fr_XX": p_generate_fr_XX,
    "es_XX": p_generate_es_XX,
    "de_DE": p_generate_de_DE,
    # "ru_RU": p_generate_ru_RU,
}

def run_generate(input_str, p_generate):
    inputs = tokenizer([input_str for i in range(num_devices)], return_tensors="jax", padding="max_length", truncation=True, max_length=64)
    p_inputs = shard(inputs.data)
    output_ids = p_generate(p_params, p_inputs)
    output_strings = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True, max_length=64)
    return output_strings

def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", index_col=False)
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    return df

def arrange_data(image_files, captions, image_urls):  # iterates through all the captions and save there translations
    try:
        lis_ = []
        for image_file, caption, image_url in zip(image_files, captions, image_urls):  # add english caption first
            lis_.append({"image_file":image_file, "caption":caption, "url":image_url, "lang_id": "en"})

        for lang in LANG_LIST:
            p_generate = map_name[lang_dict[lang]]

            for image_file, caption, image_url in zip(tqdm(image_files, total=len(image_files), position=0, leave=False, desc="processing for {lang} currently"), captions, image_urls):  # add other captions
                output = run_generate(caption, p_generate)
                lis_.append({"image_file":image_file, "caption":output[0], "url":image_url, "lang_id": lang})

            gc.collect()
        return lis_

    except Exception as e:
        print(caption, image_url, " skipped!")
        return


_df = read_tsv_file(DATASET_PATH)
train_df, val_df = train_test_split(_df, test_size=VAL_SPLIT, random_state=1234)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print("\n train/val dataset created. beginning translation")

if IS_TRAIN:
    df = train_df
    output_file_name = os.path.join(SAVE_VAL, "train_file.tsv")
    with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
        writer = csv.writer(outtsv, delimiter='\t')
        writer.writerow(["image_file", "caption", "url", "lang_id"])

else:
    df = val_df
    output_file_name = os.path.join(SAVE_VAL, "val_file.tsv")
    with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
        writer = csv.writer(outtsv, delimiter='\t')
        writer.writerow(["image_file", "caption", "url", "lang_id"])

for i in tqdm(range(0,len(df),BATCH_SIZE)):
    output_batch = arrange_data(list(df["image_file"])[i:i+BATCH_SIZE], list(df["caption"])[i:i+BATCH_SIZE], list(df["url"])[i:i+BATCH_SIZE])
    with open(output_file_name, "a", newline='') as f:
      writer = csv.DictWriter(f, fieldnames=["image_file", "caption", "url", "lang_id"], delimiter='\t')
      for batch in output_batch:
          writer.writerow(batch)