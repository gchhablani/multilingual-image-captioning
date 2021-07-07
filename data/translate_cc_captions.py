import argparse
import json
import multiprocessing
import os
import urllib.request

import mtranslate
import pandas as pd
from tqdm.auto import tqdm


def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", header=None)
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    print(df.head())
    return df


def arrange_data(caption, image_url):
    # print(caption, image_url)
    try:
        return {
            "captions": {
                **{"en": caption},
                **{
                    lang: mtranslate.translate(caption, lang, "en")
                    for lang in LANG_LIST
                },
            },
            "url": image_url,
        }
    except Exception as e:
        print(caption, image_url, " skipped!")
        return


def arrange_data_mp(df, number_required=1000):
    pool = multiprocessing.Pool(os.cpu_count())
    path_wise_data = pool.starmap(arrange_data, zip(list(df[0]), list(df[1])))
    return path_wise_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_store_path",
    type=str,
    default="conceptual_captions",
    help="path of directory where the dataset is to be saved",
)
parser.add_argument(
    "--no_of_train_samples",
    type=int,
    default=1000,
    help="Truncation size of Train Subset",
)
parser.add_argument(
    "--no_of_val_samples", type=int, default=100, help="Truncation size of Val Subset"
)
parser.add_argument("--lang_list", nargs="+", default=["fr"])
args = parser.parse_args()

DATASET_STORE_PATH = args.dataset_store_path
NO_OF_TRAIN_SAMPLES = args.no_of_train_samples
NO_OF_VAL_SAMPLES = args.no_of_val_samples
LANG_LIST = args.lang_list
train_save_dir = "train_data_translated.json"
val_save_dir = "val_data_translated.json"


os.chdir(DATASET_STORE_PATH)

train_tsv_path = "Train_GCC-training.tsv"
val_tsv_path = "Validation_GCC-1.1.0-Validation.tsv"

train_df = read_tsv_file(train_tsv_path)
val_df = read_tsv_file(val_tsv_path)

assert NO_OF_TRAIN_SAMPLES < len(train_df), "NO_OF_TRAIN_SAMPLES > len(train_df)"
assert NO_OF_TRAIN_SAMPLES < len(val_df), "NO_OF_VAL_SAMPLES > len(val_df)"


train_df = train_df.iloc[:NO_OF_TRAIN_SAMPLES]
val_df = val_df.iloc[:NO_OF_VAL_SAMPLES]


train_json = arrange_data_mp(train_df, train_save_dir)
val_json = arrange_data_mp(val_df, val_save_dir)

with open(train_save_dir, "w") as f:
    json.dump(train_json, f, indent=4)

with open(val_save_dir, "w") as f:
    json.dump(val_json, f, indent=4)
