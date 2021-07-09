import argparse
import itertools
import json
import os
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import mtranslate
import ray

parser = argparse.ArgumentParser()
parser.add_argument("--tsv_path", type=str, default="images-list-clean.tsv", help="path of directory where the dataset is stored")
parser.add_argument("--val_split", type=int, default=0.1, help="Size of validation Subset")
parser.add_argument("--lang_list", nargs="+", default=["fr", "de", "hi"], help="Language list (apart from English)")
parser.add_argument("--save_location_train", type=str, default=".", help="path of directory where the train dataset will be stored")
parser.add_argument("--save_location_val", type=str, default=".", help="path of directory where the validation dataset will be stored")
args = parser.parse_args()

DATASET_PATH = args.tsv_path
VAL_SPLIT = args.val_split
LANG_LIST = args.lang_list
if args.save_location_train != None:
    SAVE_TRAIN = args.save_location_train
    SAVE_VAL = args.save_location_val

def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", index_col=False)
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    return df

@ray.remote
def arrange_data(image_file, caption, image_url):
    try:
        lis_ = []
        lis_.append({"image_file":image_file, "caption":caption, "url":image_url, "lang_id": "en"})
        for lang in LANG_LIST:
            lis_.append({"image_file":image_file, "caption": mtranslate.translate(caption, lang, "en"), "url":image_url, "lang_id": lang})
        return lis_
    except Exception as e:
        print(caption, image_url, " skipped!")
        return

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)

_df = read_tsv_file(DATASET_PATH)
train_df, val_df = train_test_split(_df, test_size=VAL_SPLIT, random_state=1234)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print("train/val dataset created. beginning translation")

val_json = [arrange_data.remote(val_df["image_file"][i], val_df["caption"][i], val_df["url"][i]) for i in range(len(val_df))]
for x in tqdm(to_iterator(val_json), total=len(val_json)):
    pass
val_json = list(itertools.chain.from_iterable(ray.get(val_json)))

with open(os.path.join(SAVE_VAL, "val.json"), 'w', encoding='utf8') as json_file:
    json.dump(val_json, json_file, ensure_ascii=False)

train_json = [arrange_data.remote(train_df["image_file"][i], train_df["caption"][i], train_df["url"][i]) for i in range(len(train_df))]
for x in tqdm(to_iterator(train_json), total=len(train_json)):
    pass
train_json = list(itertools.chain.from_iterable(ray.get(train_json)))

with open(os.path.join(SAVE_TRAIN, "train.json"), 'w', encoding='utf8') as json_file:
    json.dump(train_json, json_file, ensure_ascii=False)


