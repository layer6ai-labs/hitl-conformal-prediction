import torch
import torchvision

from torch.utils.data import DataLoader
from config import _VALID_DATASETS, _LOADER_KW
from dataset_utils import *
from datasets import load_dataset, DatasetDict


def get_loader(dset, batch_size, shuffle=True, drop_last=False, **loader_kwargs):
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )


def get_fmnist(
    data_root,
    train_batch_size=256,
    valid_batch_size=256,
    test_batch_size=256,
    calib_batch_size=256,
    n_calib=2000,
    n_test=8000,
    m=10,
):
    train_data = torchvision.datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    n_val = int(0.1 * len(train_data))
    val_subset = torch.utils.data.Subset(train_data, list(range(0, n_val)))
    train_subset = torch.utils.data.Subset(
        train_data, list(range(n_val, len(train_data)))
    )
    calib_subset = torch.utils.data.Subset(test_data, list(range(0, n_calib)))
    if n_test > len(test_data) - n_calib:
        raise ValueError(
            f"Not enough data for requested calibration size. n_calib: {n_calib}, n_test: {n_test}"
        )
    elif n_test <= 0:
        test_subset = torch.utils.data.Subset(
            test_data, list(range(n_calib, len(test_data)))
        )
    else:
        test_subset = torch.utils.data.Subset(
            test_data, list(range(n_calib, n_test + n_calib))
        )

    train_loader = get_loader(
        train_subset, train_batch_size, shuffle=True, drop_last=True
    )
    val_loader = get_loader(
        val_subset, valid_batch_size, shuffle=False, drop_last=False
    )
    calib_loader = get_loader(
        calib_subset, calib_batch_size, shuffle=False, drop_last=False
    )
    test_loader = get_loader(
        test_subset, test_batch_size, shuffle=False, drop_last=False
    )

    print(
        f"Dataset sizes: Train {len(train_loader.dataset)}, Val {len(val_loader.dataset)}, Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}."
    )

    return {
        "calib": calib_loader,
        "test": test_loader,
        "train": train_loader,
        "val": val_loader,
    }


def get_go_emotions(
    data_root,
    test_batch_size=256,
    calib_batch_size=256,
    n_calib=2000,
    n_test=-1,
    m=10,
):
    train_loader, val_loader = None, None  # Training not required
    dataset_test = load_dataset("go_emotions", "simplified", split="test")
    dataset_val = load_dataset("go_emotions", "simplified", split="validation")
    dataset_simplified = DatasetDict({"test": dataset_test, "validation": dataset_val})

    # filter on single label and most popular m:
    dataset_simplified, top_m_labels = filter_go_emotions(dataset_simplified, m=m)
    dataset_simplified_tokenized = tokenize_and_process_go_emotions(dataset_simplified)

    dataset_simplified_tokenized["validation"].classes = [
        i for i in range(m)
    ]  # all labels
    n_valid = len(dataset_simplified_tokenized["validation"])
    if n_calib > n_valid:
        raise ValueError(
            f"Not enough data for requested calibration size. n_calib: {n_calib}, validation_size: {n_valid}"
        )
    elif n_calib <= 0:
        calib_subset = dataset_simplified_tokenized["validation"]
    else:
        random_subset = torch.randperm(n_valid)[:n_calib].tolist()
        calib_subset = torch.utils.data.Subset(
            dataset_simplified_tokenized["validation"], random_subset
        )
    test_subset = dataset_simplified_tokenized["test"]

    calib_loader = get_loader(
        calib_subset, calib_batch_size, shuffle=False, drop_last=False
    )
    test_loader = get_loader(
        test_subset, test_batch_size, shuffle=False, drop_last=False
    )

    print(
        f"Dataset sizes: Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}."
    )

    return {
        "calib": calib_loader,
        "test": test_loader,
        "train": train_loader,
        "val": val_loader,
        "top_m_labels": top_m_labels,
    }


def get_object_net(
    data_root,
    test_batch_size=256,
    calib_batch_size=256,
    n_calib=2000,
    n_test=-1,
    m=20,
):
    train_loader, val_loader = None, None  # Training not required

    calib_subset, test_subset, classnames = create_and_split_objectnet(
        data_root, n_calib, m=m
    )
    calib_loader = get_loader(
        calib_subset, calib_batch_size, shuffle=False, drop_last=False
    )
    test_loader = get_loader(
        test_subset, test_batch_size, shuffle=False, drop_last=False
    )
    print(
        f"Dataset sizes: Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}."
    )

    return {
        "calib": calib_loader,
        "test": test_loader,
        "train": train_loader,
        "val": val_loader,
        "top_m_labels": classnames,
    }


def get_few_nerd(
    data_root,
    test_batch_size=50,
    calib_batch_size=50,
    n_calib=2000,
    n_test=2000,
    m=20,
):
    train_loader, val_loader = None, None  # Training not required
    dataset_test = load_dataset(
        "DFKI-SLT/few-nerd", "supervised", split="test"
    )
    dataset_val = load_dataset(
        "DFKI-SLT/few-nerd", "supervised", split="validation"
    )

    # class mappings
    class_labels = dataset_val.features["fine_ner_tags"].feature.names
    class_label_map = {i: j for i, j in enumerate(class_labels)}

    # filter top m most frequent labels and do stratified sampling to keep same number of samples per label
    dataset_simplified, top_m_labels = filter_few_nerd(
        dataset_test, dataset_val, m=m, class_label_map=class_label_map
    )

    # tokenize dataset
    dataset_simplified_tokenized, tokenizer, model_config = tokenize_few_nerd(
        dataset_simplified
    )

    # since tokens are divided into spans, identify the span ID of interest
    dataset_simplified_tokenized = dataset_simplified_tokenized.map(
        lambda example: identify_required_span_and_target(
            example, tokenizer, model_config, top_m_labels
        )
    )

    # remove columns
    dataset_simplified_tokenized = dataset_simplified_tokenized.remove_columns(
        ["unique_fine_ner_tags"]
    )

    # subsample to desired size
    calib_subset = subsample_few_nerd(dataset_simplified_tokenized["validation"], n_calib, top_m_labels)
    test_subset = subsample_few_nerd(dataset_simplified_tokenized["test"], n_test, top_m_labels)

    # instantiate data collator
    data_collator = fetch_data_collator(tokenizer, model_config)

    calib_subset = compute_span_marker_model_inputs(calib_subset, data_collator)
    test_subset = compute_span_marker_model_inputs(test_subset, data_collator)

    # for Conformal.py module
    calib_subset.classes = [i for i in range(m)]  # all labels

    # Loaders
    calib_loader = get_loader(
        calib_subset, calib_batch_size, shuffle=False, drop_last=False
    )
    test_loader = get_loader(
        test_subset, test_batch_size, shuffle=False, drop_last=False
    )

    print(
        f"Dataset sizes: Calib {len(calib_loader.dataset)}, Test {len(test_loader.dataset)}."
    )

    return {
        "calib": calib_loader,
        "test": test_loader,
        "train": train_loader,
        "val": val_loader,
        "top_m_labels": top_m_labels,
        "id2label_mappings": class_label_map,
    }


DATASET_FN_MAP = {
    "fashion-mnist": get_fmnist,
    "go-emotions": get_go_emotions,
    "object-net": get_object_net,
    "few-nerd": get_few_nerd,
}


def get_loaders(cfg):
    dataset = cfg["dataset"]
    data_root = cfg["data_root"]
    assert dataset in _VALID_DATASETS, f"Unknown dataset {dataset}"

    loader_kwargs = get_loader_kwargs(cfg)

    dataset_fn = DATASET_FN_MAP[dataset]
    output_dict = dataset_fn(data_root, **loader_kwargs)

    return output_dict


def get_loader_kwargs(cfg):
    # Create subdict of cfg for the keys relevant to dataloading
    loader_kw = _LOADER_KW
    loader_kwargs = {k: cfg[k] for k in cfg.keys() & loader_kw}

    return loader_kwargs
