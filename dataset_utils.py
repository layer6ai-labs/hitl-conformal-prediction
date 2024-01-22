from collections import Counter
import json
import os
from transformers import AutoTokenizer, AutoConfig
from datasets import DatasetDict, Dataset
import emoji
import torch
import torchvision
from PIL import Image
from itertools import chain, groupby
import random
from span_marker.tokenizer import SpanMarkerTokenizer
from span_marker.trainer import Trainer
from span_marker.data_collator import SpanMarkerDataCollator
import datasets
import html
import re

datasets.utils.logging.disable_progress_bar()

PUNCTUATION_TOKENS = [":",",",".",";","'","``","`",'"',"!", ")", "(", "%"]

### General Tools
def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_2 = df.groupby(col).apply(lambda x: x.sample(n))
    df_2.index = df_2.index.droplevel(0)
    return df_2


def bring_examples_to_top(df, m, min_num, repeat=3):
    # Sort dataframe, then put example instances from each class at the top
    df = df.sort_values("label", axis=0)
    sorted_order = df.index.to_list()
    temps = [[0] * m for i in range(repeat)]
    for i in range(m):
        for temp in temps:
            temp[i] = sorted_order.pop(i * min_num - i)
    for j, temp in enumerate(temps):
        for i, x in enumerate(temp):
            sorted_order.insert(i + m * j, x)
    df = df.reindex(sorted_order)
    return df

def prediction_set_text_fn(prediction_set, idlabels):
    out = ""
    for el in prediction_set.split():
        if el != " ":
            text = idlabels[int(el)].title()
            out += el + f". {text}  "
    return out[:-2]  # remove final spaces

def corr_ans_text_fn(label_text, reindexed_label):
    out = f"The best answer is {reindexed_label}. {label_text.title()}."
    return out

### fashion-mnist
def process_fmnist_dataframe(df, k):
    id2label_fm = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    min_num = min(203, df["label"].value_counts().min())
    m = len(id2label_fm)
    df = stratified_sample_df(df, "label", min_num * m)

    df["label_text"] = df["label"].apply(lambda label: id2label_fm[label])
    df["original_label"] = df["label"]

    def corr_ans_text_fn(label_text, label):
        out = f"The best answer is {label}. {label_text}."
        return out

    df["corr_ans_text"] = df.apply(
        lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
    )

    df["conformal_text"] = df.apply(
        lambda x: prediction_set_text_fn(x["conformal_set"], id2label_fm), axis=1
    )
    df[f"top{k}_text"] = df.apply(
        lambda x: prediction_set_text_fn(x[f"top{k}_set"], id2label_fm), axis=1
    )

    df = bring_examples_to_top(df, m, min_num)
    return df


### go-emotions
def flatten_labels(example):
    example["labels"] = [label[0] for label in example["labels"]]
    return example


def remap_labels(example, top_m_labels):
    example["labels"] = [top_m_labels.index(label) for label in example["labels"]]
    return example


def filter_go_emotions(dataset, m=10):
    # only keeps examples with a single label and without any emojis
    dataset = dataset.filter(lambda example: len(example["labels"]) == 1)
    dataset = dataset.filter(lambda example: emoji.emoji_count(example["text"]) == 0)

    # only keeps samples with the top m popular labels
    dataset = dataset.map(flatten_labels, batched=True)
    label_counts = Counter(dataset["validation"]["labels"])
    top_m_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:m]

    dataset = dataset.filter(lambda example: example["labels"] in top_m_labels)
    dataset = dataset.map(
        remap_labels, fn_kwargs={"top_m_labels": top_m_labels}, batched=True
    )

    # keep an equal number of each of the classes using stratified sampling
    for split in dataset:
        # get the number of examples in the least common class
        split_label_counts = Counter(dataset[split]["labels"])
        min_num = split_label_counts[
            min(split_label_counts, key=split_label_counts.get)
        ]
        dataset[split].set_format("pandas")
        pd_dataset = dataset[split][:]
        pd_dataset = stratified_sample_df(pd_dataset, col="labels", n_samples=min_num)
        dataset[split] = Dataset.from_pandas(pd_dataset)

    return dataset, top_m_labels


def tokenize_and_process_go_emotions(dataset):
    tokenizer = AutoTokenizer.from_pretrained(
        "SamLowe/roberta-base-go_emotions", use_fast=False
    )
    # Tokenize and truncate the dataset
    dataset_tokenized = dataset.map(
        lambda example: tokenizer(
            example["text"],
            truncation="longest_first",
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        ),
        batched=True,
    )

    dataset_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "text"]
    )
    return dataset_tokenized


def add_space_if_looks_like_list(val):
    if val.startswith("[") and val.endswith("]"):
        val = val + " "
    return val


def escape_dollar_signs_brackets(val):
    if val.startswith("$"):
        val = val.replace("$", "\$", 1)
    return val.replace("[", "{").replace("]", "}")


def process_go_emotions_dataframe(df, top_m_labels, k):
    # modify the input data to display on psychopy
    df["text_prompt"] = (
        df["text_prompt"]
        .apply(add_space_if_looks_like_list)
        .apply(escape_dollar_signs_brackets)
    )

    id2label = {
        0: "admiration",
        1: "amusement",
        2: "anger",
        3: "annoyance",
        4: "approval",
        5: "caring",
        6: "confusion",
        7: "curiosity",
        8: "desire",
        9: "disappointment",
        10: "disapproval",
        11: "disgust",
        12: "embarrassment",
        13: "excitement",
        14: "fear",
        15: "gratitude",
        16: "grief",
        17: "joy",
        18: "love",
        19: "nervousness",
        20: "optimism",
        21: "pride",
        22: "realization",
        23: "relief",
        24: "remorse",
        25: "sadness",
        26: "surprise",
        27: "neutral",
    }
    m = len(top_m_labels)
    id2label_ge = {x: id2label[top_m_labels[x]] for x in range(m)}

    df["label_text"] = df["label"].apply(lambda label: id2label_ge[label])
    df["original_label"] = df["label"].apply(lambda label: top_m_labels[label])

    df["corr_ans_text"] = df.apply(
        lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
    )

    df["conformal_text"] = df.apply(
        lambda x: prediction_set_text_fn(x["conformal_set"], id2label_ge), axis=1
    )
    df[f"top{k}_text"] = df.apply(
        lambda x: prediction_set_text_fn(x[f"top{k}_set"], id2label_ge), axis=1
    )

    # Sort dataframe, then put example instances from each class at the top
    min_num = df["label"].value_counts().min()
    df = bring_examples_to_top(df, m, min_num)

    return df


### object-net
def process_object_net_dataframe(df, classnames, k):
    id2label_obj = {i: classnames[i] for i in range(len(classnames))}
    m = len(classnames)
    df["original_label"] = df["label"]

    if m == 20: # Original label -> alphabetical order label with custom names
        label_reordering = {
            0: 16,
            1: 5,
            2: 19,
            3: 7,
            4: 18,
            5: 12,
            6: 17,
            7: 9,
            8: 3,
            9: 6,
            10: 20,
            11: 15,
            12: 10,
            13: 4,
            14: 8,
            15: 1,
            16: 2,
            17: 13,
            18: 11,
            19: 14,
        }
        df["label"] = df["label"].apply(lambda label: label_reordering[label])

        def relabel_set_obj(prediction_set):
            out = ""
            prediction_set = str(prediction_set)  # For single element int sets
            prediction_set = prediction_set.split()
            prediction_set = [label_reordering[int(el)] for el in prediction_set]
            prediction_set.sort()
            for el in prediction_set:
                out += str(el) + " "
            return out[:-1]  # remove final space

        df["conformal_set"] = df["conformal_set"].apply(relabel_set_obj)
        df[f"top{k}_set"] = df[f"top{k}_set"].apply(relabel_set_obj)
        df["top1"] = df["top1"].apply(relabel_set_obj)
        id2label_obj = {
            1: "Backpack",
            2: "Banana",
            3: "Bandage",
            4: "Battery",
            5: "Belt",
            6: "Blanket",
            7: "Book",
            8: "Bottle",
            9: "Bottle Cap",
            10: "Bottle Opener",
            11: "Broom",
            12: "Bucket",
            13: "Candle",
            14: "Cellphone",
            15: "Cellphone Charger",
            16: "Envelope",
            17: "Figurine",
            18: "Sandal",
            19: "Knife",
            20: "Trash Bin",
        }

    df["label_text"] = df["label"].apply(lambda label: id2label_obj[label])

    df["corr_ans_text"] = df.apply(
        lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
    )

    df["conformal_text"] = df.apply(
        lambda x: prediction_set_text_fn(x["conformal_set"], id2label_obj), axis=1
    )
    df[f"top{k}_text"] = df.apply(
        lambda x: prediction_set_text_fn(x[f"top{k}_set"], id2label_obj), axis=1
    )

    def get_last_two_file_path(file_path):
        root = "images/object-net"
        directory, file = os.path.split(file_path)
        directory_parts = directory.split(os.path.sep)
        last_dir = directory_parts[-1]
        result = os.path.join(root, last_dir, file)
        return result

    df["text_prompt"] = df["text_prompt"].apply(get_last_two_file_path)

    min_num = df["label"].value_counts().min()
    df = bring_examples_to_top(df, m, min_num)
    return df


def create_and_split_objectnet(vis_root, n_calib, m=20):
    vis_processor = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # torchvision.transforms.CenterCrop(224),
            # Above two steps have already been done
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    inner_dataset = torchvision.datasets.ImageFolder(os.path.join(vis_root, "images"))
    folder_to_objectnet_class = json.load(
        open(os.path.join(vis_root, "mappings", "folder_to_objectnet_label.json"), "r")
    )
    # Only keep top m most common classes
    label_counts = Counter(inner_dataset.targets)
    top_m_labels = list(dict(label_counts.most_common()).keys())[:m]
    min_num = label_counts[top_m_labels[-1]]
    size = m * min_num

    classnames = [
        folder_to_objectnet_class[inner_dataset.classes[class_idx]].lower()
        for class_idx in top_m_labels
    ]
    classes = [i for i in range(len(classnames))]

    if n_calib <= 0:
        raise ValueError(
            f"Invalid choice of n_calib: {n_calib}. Can't use the entire dataset for objectnet calibration."
        )
    elif n_calib > size:
        raise ValueError(
            f"Not enough data for requested calibration size. n_calib: {n_calib}, dataset_size: {size}"
        )
    elif 0 < n_calib < 1:
        # Take fraction of the dataset
        calib_frac = n_calib
        n_calib = int(n_calib * size)
        n_calib = n_calib - (n_calib % m)  # ensure that dataset will be class balanced
        print(
            f"Using {calib_frac * 100}% of the dataset for calibration, {n_calib} datapoints."
        )
    else:
        n_calib = n_calib - (n_calib % m)  # ensure that dataset will be class balanced

    min_num_calib = int(n_calib / m)  # Number of each class for calib
    min_num_test = min_num - min_num_calib

    # Ensure that both calibration and test sets are stratified the same way (exchangable)
    num_examples_per_class_calib = [0 for i in range(len(folder_to_objectnet_class))]
    num_examples_per_class_test = [0 for i in range(len(folder_to_objectnet_class))]
    calib_annotation = []
    test_annotation = []

    for path, class_idx in inner_dataset.imgs:
        # stratified sampling, e.g. load only min_num_calib examples per class for calib
        if class_idx in top_m_labels:
            if num_examples_per_class_calib[class_idx] < min_num_calib:
                num_examples_per_class_calib[class_idx] += 1
                calib_annotation.append(
                    {
                        "image": path,
                        "label": classnames.index(
                            folder_to_objectnet_class[
                                inner_dataset.classes[class_idx]
                            ].lower()
                        ),
                    }
                )
            elif num_examples_per_class_test[class_idx] < min_num_test:
                num_examples_per_class_test[class_idx] += 1
                test_annotation.append(
                    {
                        "image": path,
                        "label": classnames.index(
                            folder_to_objectnet_class[
                                inner_dataset.classes[class_idx]
                            ].lower()
                        ),
                    }
                )

    calib_dataset = ObjectNet(vis_processor, classnames, classes, calib_annotation)
    test_dataset = ObjectNet(vis_processor, classnames, classes, test_annotation)
    # Downstream processing expects a Subset object
    calib_subset = torch.utils.data.Subset(
        calib_dataset, torch.randperm(len(calib_dataset)).tolist()
    )
    test_subset = torch.utils.data.Subset(
        test_dataset, torch.randperm(len(test_dataset)).tolist()
    )

    return calib_subset, test_subset, classnames


class ObjectNet(Dataset):
    def __init__(self, vis_processor, classnames, classes, annotation):
        self.vis_processor = vis_processor
        self.classnames = classnames
        self.classes = classes
        self.annotation = annotation

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "label": ann["label"], "image_id": image_path}


### few-nerd
def unique_fine_ner_tags(example):
    example["unique_fine_ner_tags"] = [
        list(set(fine_tags)) for fine_tags in example["fine_ner_tags"]
    ]
    return example


def ner_tag_stratified_sampling(dataset, top_m_labels):
    for split in dataset:
        unique_tag_list = dataset[split]["unique_fine_ner_tags"]

        # Flatten the list of lists to get a single list of classes. Also note corresponding sublist indices
        flat_unique_tag_list = []
        flat_unique_tag_sublist_indices = []
        for i, sublist in enumerate(unique_tag_list):
            for tag in sublist:
                flat_unique_tag_list.append(tag)
                flat_unique_tag_sublist_indices.append(i)

        label_counts = {
            i: j for i, j in Counter(flat_unique_tag_list).items() if i in top_m_labels
        }
        min_label_count = min(label_counts.values())

        # Sample the same number of points for each class
        sampled_tags = []
        sampled_indices = []
        for tag in sorted(label_counts, key=lambda k: label_counts[k]):
            sampled_tags.extend([tag] * min_label_count)

            indices = [
                i
                for i, v in zip(flat_unique_tag_sublist_indices, flat_unique_tag_list)
                if (v == tag and i not in sampled_indices)
            ]
            sampled_indices.extend(random.sample(indices, min_label_count))

        dataset[split] = dataset[split].select(indices=sampled_indices)
        dataset[split] = dataset[split].add_column("sampled_fine_ner_tag", sampled_tags)

    return dataset


def unique_elements_shortlist(example, top_m_labels):
    input_list = example["fine_ner_tags"]
    example["non_repetitive_fine_ner_tags"] = [
        [
            key
            for key, group in groupby(ner_list)
            if ner_list.count(key) == len(list(group))
        ]
        for ner_list in input_list
    ]
    example["sampled_fine_ner_tag"] = [
        random.choice([x for x in tags if x in top_m_labels])
        if (len(tags) > 0 and len(set(tags) & set(top_m_labels)) > 0)
        else None
        for tags in example["non_repetitive_fine_ner_tags"]
    ]

    return example


# resolve issue of special token mapped to None
# 1. set -100 for special tokens as we do not want them to be trained on them
def tokenize_and_align_labels(example, tokenizer, label_all_tokens=True):
    tokenized_input = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )  # ,return_num_words=True, return_batch_encoding=True
    labels = []
    for i, label in enumerate(example["fine_ner_tags"]):
        word_ids = tokenized_input.word_ids(batch_index=i)
        prev_index = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_index:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            prev_index = word_idx
        labels.append(label_ids)
    tokenized_input["fine_ner_tags_adjusted"] = labels
    return tokenized_input


def tokenize_few_nerd_data(example, tokenizer):
    # Tokenize & add start/end markers
    tokenized_input = tokenizer(
        {"tokens": example["tokens"]}, return_num_words=True, return_batch_encoding=True
    )
    tokenized_input.pop("batch_encoding")
    return tokenized_input


def select_top_m_labels(m, label_counts, class_label_map):
    # drop default label 0 (not an entity)
    label_counts.pop(0)

    # drop subcategories that contain "other" (eg: person-other), and include main category "other" (eg: other-currency)
    for i, j in class_label_map.items():
        subcategory = j.split("-")[1] if len(j.split("-")) > 1 else None
        if "other" == subcategory:
            label_counts.pop(i)

    top_m_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:m]
    return top_m_labels


def filter_few_nerd(dataset_test, dataset_val, class_label_map, m=20):
    # filter non ASCII character samples
    dataset_test = dataset_test.filter(filter_non_ascii_characters)
    dataset_val = dataset_val.filter(filter_non_ascii_characters)

    # filter samples with entity span > max entity span length (8 tokens)
    dataset_test = dataset_test.filter(lambda x: filter_above_threshold_entity_span(x))
    dataset_val = dataset_val.filter(lambda x: filter_above_threshold_entity_span(x))

    # filter samples where all "non an entity" tokens are punctuations
    dataset_test = dataset_test.filter(lambda x: filter_punctuation_only_not_an_entity_samples(x))
    dataset_val = dataset_val.filter(lambda x: filter_punctuation_only_not_an_entity_samples(x))

    # filter samples with num_words > 60 -> ~99% test data samples
    # This is to make it easier to run the code within the codebase without much refactoring.
    # Else samples need to be broken down into sub samples (Trainer.spread_sample) and then combined again after inference
    dataset_test = dataset_test.filter(
        lambda x: filter_above_token_limit_threshold(x, num_token_threshold=60)
    )
    dataset_val = dataset_val.filter(
        lambda x: filter_above_token_limit_threshold(x, num_token_threshold=60)
    )

    dataset = DatasetDict({"test": dataset_test, "validation": dataset_val})

    # only keeps samples with the top m popular labels
    dataset = dataset.map(unique_fine_ner_tags, batched=True)
    label_counts = Counter(chain(*dataset["validation"]["unique_fine_ner_tags"]))

    # Select top m label categories
    top_m_labels = select_top_m_labels(m, label_counts, class_label_map)

    # filter dataset to include top m labels
    dataset = dataset.filter(
        lambda example: len(set(example["unique_fine_ner_tags"]) & set(top_m_labels))
        > 0
    )

    # Out of all the ner tags in each example, randomly pick label tag which is also part of top m labels
    # Also, perform stratified sampling to ensure same number of datapoints are sampled for each label
    dataset = ner_tag_stratified_sampling(dataset, top_m_labels)

    return dataset, top_m_labels


def spread_sample(dataset, tokenizer, model_config):
    dataset = dataset.map(
        Trainer.spread_sample,
        batched=True,
        desc="Spreading data between multiple samples",
        fn_kwargs={
            "model_max_length": tokenizer.model_max_length,
            "marker_max_length": model_config.marker_max_length,
        },
    )

    return dataset


def fetch_data_collator(tokenizer, model_config):
    return SpanMarkerDataCollator(
        tokenizer=tokenizer, marker_max_length=model_config.marker_max_length
    )


def get_ner_tag_indices_pairs(label_indices):
    tag_indices_pairs = []
    start_idx = None
    end_idx = None
    i = 0
    while i < len(label_indices):
        start_idx = i
        while (i < len(label_indices) - 1) and label_indices[i] + 1 == label_indices[
            i + 1
        ]:
            i += 1
        end_idx = i
        tag_indices_pairs.append((label_indices[start_idx], label_indices[end_idx]))
        i += 1
    return tag_indices_pairs


def identify_required_span_and_target(example, tokenizer, model_config, top_m_labels):
    valid_spans = list(
        tokenizer.get_all_valid_spans(
            len(example["tokens"]), model_config.entity_max_length
        )
    )
    label_indices = [
        index
        for index, value in enumerate(example["fine_ner_tags"])
        if value == example["sampled_fine_ner_tag"]
    ]

    # get (start_idx, end_idx) pairs for all the occurances of sampled ner tag in the example and randomly pick one
    # of them. If sampled ner tag is 0, then randomly pick any token idx and use the same to construct (start_idx,
    # end_idx) pair

    if int(example["sampled_fine_ner_tag"]) == 0:
        # ignore indices with punctuations
        filtered_label_indices = [idx for idx in label_indices if example["tokens"][idx].strip() not in PUNCTUATION_TOKENS]
        default_value_idx = random.choice(filtered_label_indices)
        label_index = (default_value_idx, default_value_idx)
    else:
        span_indices_pairs = get_ner_tag_indices_pairs(label_indices)
        label_index = random.choice(span_indices_pairs)

    # create columns to include start and end positions of NER tags (token start idx, token end idx)
    example["token_start_idx"] = label_index[0]
    example["token_end_idx"] = label_index[-1]  # inclusive

    label_indices_to_span = (label_index[0], label_index[-1] + 1)
    example["required_span_index"] = valid_spans.index(label_indices_to_span)

    # add target
    example["target"] = top_m_labels.index(example["sampled_fine_ner_tag"])
    return example


def subsample_few_nerd(dataset, n, top_m_labels):
    # num of unique samples before spreading based on num_spans
    n_full = len(set(dataset["id"]))
    if n > n_full:
        raise ValueError(
            f"Not enough data for requested dataset size. Requested: {n}, available: {n_full}"
        )
    if n <= 0:
        subset = dataset
    else:
        num_samples_per_group = int(n / len(top_m_labels))
        random_ids = []
        for i in top_m_labels:
            unique_ids = set(
                dataset.filter(
                    lambda x: x["sampled_fine_ner_tag"] == i
                )["id"]
            )
            random_ids.extend(random.sample(unique_ids, num_samples_per_group))
        random_indices = [
            index
            for index, element in enumerate(
                dataset["id"]
            )
            if element in random_ids
        ]
        subset = dataset.select(
            indices=random_indices
        )
    return subset


def tokenize_few_nerd(dataset):
    model_config = AutoConfig.from_pretrained(
        "tomaarsen/span-marker-roberta-large-fewnerd-fine-super"
    )

    # increase model max lenth to 512 and marker max length to 256
    model_config.model_max_length = 512
    model_config.marker_max_length = 256

    tokenizer = SpanMarkerTokenizer.from_pretrained(
        "tomaarsen/span-marker-roberta-large-fewnerd-fine-super", config=model_config
    )

    dataset_tokenized = dataset.map(
        lambda batch: tokenize_few_nerd_data(batch, tokenizer), batched=True
    )

    return dataset_tokenized, tokenizer, model_config


def filter_non_ascii_characters(example):
    tokens = example["tokens"]
    # Filter out examples where 'token' contains non-ASCII characters
    return all(all(ord(char) < 128 for char in token) for token in tokens)


def filter_above_threshold_entity_span(example):
    fine_ner_tags = example["fine_ner_tags"]
    if not fine_ner_tags:
        return False

    tag_freq_group_counter = groupby(fine_ner_tags)
    max_freq = max(
        [len(list(value)) for key, value in tag_freq_group_counter if key != 0],
        default=0,
    )  # ignore 0 as they dont represent any category
    return max_freq <= 8


def filter_punctuation_only_not_an_entity_samples(example):
    # filter samples where all "non an entity" tokens are punctuations
    punctuation_tokens_list = [i for i in example["tokens"] if i in PUNCTUATION_TOKENS]
    return not (len([x for x in example["fine_ner_tags"] if x==0]) == len(punctuation_tokens_list))


def filter_above_token_limit_threshold(example, num_token_threshold=30):
    return len(example["tokens"]) < num_token_threshold


def compute_span_marker_model_inputs(df, data_collator):
    df_collator = data_collator(df)

    for col in [
        "required_span_index",
        "target",
        "fine_ner_tags",
        "tokens",
        "token_start_idx",
        "token_end_idx",
    ]:
        df_collator[col] = df[col]

    # convert id and targets to integer and then tensors
    for col in ["id", "sampled_fine_ner_tag"]:
        df_collator[col] = torch.tensor([int(x) for x in df[col]])

    df_processed = Dataset.from_dict(df_collator)
    df_processed.set_format(
        type="torch",
        columns=[
            "input_ids",
            "position_ids",
            "attention_mask",
            "num_words",
            "start_marker_indices",
            "num_marker_pairs",
            "id",
            "sampled_fine_ner_tag",
            "required_span_index",
            "target",
            "token_start_idx",
            "token_end_idx",
        ],
    )
    return df_processed


def concat_text_with_ner_tags(id, data):
    index = torch.where(data["id"] == id)[0].item()
    example = data[index]
    # since tokens and fine ner tags are not in torch format, add them manually to example dict
    for col in ["tokens", "fine_ner_tags"]:
        example[col] = data[col][index]

    merged_list = []
    for token, fine_ner_tag in zip(example["tokens"], example["fine_ner_tags"]):
        merged_list.append(f"{token}->({fine_ner_tag})")
    return "|".join(merged_list)


def clean_text_prompt(input_text):
    # remove spaces before punctuation etc
    # input_text = input_text.replace("``",'')
    pattern = r"\s*([`.,;()\'\[\]])"
    result = re.sub(pattern, r"\1", input_text)

    # remove spaces after parenthesis and quotes
    pattern_quotes_parentheses = r'([`"\'\'(])\s*'
    result = re.sub(pattern_quotes_parentheses, r"\1", result)
    return result


def populate_tokens(id, data):
    index = torch.where(data["id"] == id)[0].item()
    example = data[index]
    # since tokens and fine ner tags are not in torch format, add them manually to example dict
    for col in ["tokens"]:
        example[col] = data[col][index]

    # Use double curly brackets ({{ ..... }}) to highlight the entity span
    example["tokens"].insert(example["token_end_idx"] + 1, "}}")
    example["tokens"].insert(example["token_start_idx"], "{{")
    example["tokens"] = " ".join(example["tokens"])

    # Remove HTML escape codes to make it human-readable
    example["tokens"] = (
        example["tokens"]
        .replace("& amp ;", "&amp;")
        .replace("& lt ;", "&lt;")
        .replace("& gt ;", "&gt;")
        .replace("& quot ;", "&quot;")
        .replace("& apos ;", "&apos;")
    )
    example["tokens"] = html.unescape(example["tokens"])

    # Clean up the text - remove space before comma, full stop etc
    example["tokens"] = clean_text_prompt(example["tokens"])

    return example["tokens"]


def remove_brackets(text_prompt):
    if "[[" in text_prompt and "]]" in text_prompt:
        # Remove double brackets from the string
        text_prompt = text_prompt.replace("[[", "").replace("]]", "")
    if "[" in text_prompt and "]" in text_prompt:
        # Remove brackets from the string
        text_prompt = text_prompt.replace("[", "").replace("]", "")
    return text_prompt


def process_few_nerd_dataframe(df, top_m_labels, k, id2label_few, dataset_test):
    # load the dataset and for each id, get corresponding text string
    df["text_prompt_with_original_fine_ner_tags"] = df["text_prompt"].apply(
        lambda x: concat_text_with_ner_tags(x, dataset_test)
    )

    df["text_prompt"] = df["text_prompt"].apply(
        lambda x: populate_tokens(x, dataset_test)
    )
    # JavaScript does unwanted parsing of list literals on brackets
    df["text_prompt"] = df["text_prompt"].apply(
        lambda x: remove_brackets(x)
    )
    df["original_label_text"] = df["label"].apply(
        lambda label: id2label_few[top_m_labels[label]]
    )
    df["original_label"] = df["label"].apply(lambda label: top_m_labels[label-1])

    # relabel dataset
    m = len(top_m_labels)
    if m == 20: # label -> alphabetical order label with custom names
        label_reordering = {
            0: 10,
            1: 7,
            2: 2,
            3: 15,
            4: 3,
            5: 9,
            6: 18,
            7: 11,
            8: 19,
            9: 16,
            10: 5,
            11: 1,
            12: 8,
            13: 12,
            14: 6,
            15: 13,
            16: 4,
            17: 14,
            18: 17,
            19: 20,
        }
        df["label"] = df["label"].apply(lambda label: label_reordering[label])

        def relabel_set_few(prediction_set):
            out = ""
            prediction_set = str(prediction_set)  # For single element int sets
            prediction_set = prediction_set.split()
            prediction_set = [label_reordering[int(el)] for el in prediction_set]
            prediction_set.sort()
            for el in prediction_set:
                out += str(el) + " "
            return out[:-1]  # remove final space

        df["conformal_set"] = df["conformal_set"].apply(relabel_set_few)
        df[f"top{k}_set"] = df[f"top{k}_set"].apply(relabel_set_few)
        df["top1"] = df["top1"].apply(relabel_set_few)
        id2label_few = {
            1: "Actor/Actress",
            2: "Artist/Author",
            3: "Athlete",
            4: "Award",
            5: "Body of Water",
            6: "Biology",
            7: "Company",
            8: "Military Conflict",
            9: "Education",
            10: "Geopolitics",
            11: "Government",
            12: "Media Organization",
            13: "Music",
            14: "Political Party",
            15: "Politician/Leader",
            16: "Sports Event",
            17: "Sports League",
            18: "Sports Team",
            19: "Transportation Route",
            20: "Written Art",
        }
        df["label_text"] = df["label"].apply(lambda label: id2label_few[label])

        df["corr_ans_text"] = df.apply(
            lambda x: corr_ans_text_fn(x["label_text"], x["label"]), axis=1
        )
        df["conformal_text"] = df.apply(
            lambda x: prediction_set_text_fn(x["conformal_set"], id2label_few), axis=1
        )
        df[f"top{k}_text"] = df.apply(
            lambda x: prediction_set_text_fn(x[f"top{k}_set"], id2label_few), axis=1
        )

    # Sort dataframe, then put example instances from each class at the top
    min_num = df["label"].value_counts().min()
    df = bring_examples_to_top(df, m, min_num, repeat=2)

    return df
