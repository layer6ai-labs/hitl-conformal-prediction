import ast

_VALID_DATASETS = {"fashion-mnist", "go-emotions", "object-net", "few-nerd"}
_LOADER_KW = {
    "train_batch_size",
    "valid_batch_size",
    "test_batch_size",
    "calib_batch_size",
    "n_calib",
    "n_test",
    "m",
}
_MODEL_KW = {"optimizer", "lr", "epochs", "model_size"}


def get_config(dataset):
    dataset = dataset.lower()
    assert dataset in _VALID_DATASETS, f"Unknown dataset {dataset}"

    base_config = CFG_MAP["base"]()

    dataset_config = CFG_MAP[dataset]()

    return {
        "dataset": dataset,
        **base_config,
        **dataset_config,  # dataset unpacked last, so overwrites base if there are duplicates
    }


def get_base_config():
    # Shared config applicable to all datasets
    cfg = {
        "seed": 0,  # random seed for reproducibility
        "alpha": None,  # conformal error tolerance rate
        "kreg": None,
        "lamda": None,
        "T": 1.0,
        "test_batch_size": 256,
        "calib_batch_size": 256,
        "m": 10,
        "data_root": "data/",
        "logdir_root": "logs/",
        "k": 3,
    }

    return cfg


def get_fmnist_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "train_batch_size": 256,
        "valid_batch_size": 256,
        "n_calib": 2000,
        "n_test": 8000,
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 2,
    }

    return cfg


def get_go_emotions_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "kreg": 4,
        "lamda": 0.5,
        "T": 0.3,
        "n_calib": -1,  # use all calibration data
        "n_test": -1,  # use all test data
        "m": 10,
    }

    return cfg


def get_object_net_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "kreg": 5,
        "lamda": 0.5,
        "T": 0.002,
        "n_calib": 2000,
        "n_test": -1,  # use all test data
        "m": 20,
        "model_size": "ViT-L/14",
        "data_root": "data/ObjectNet/objectnet-1.0/",
    }

    return cfg


def get_few_nerd_config():
    # Add dataset-specific config parameters as required
    cfg = {
        "kreg": 5,
        "lamda": 0.5,
        "T": 0.3,
        "test_batch_size": 50,
        "calib_batch_size": 50,
        "n_calib": -1,  # use all calibration data
        "n_test": 2000,  # use all test data
        "m": 20,
    }
    return cfg


CFG_MAP = {
    "base": get_base_config,
    "fashion-mnist": get_fmnist_config,
    "go-emotions": get_go_emotions_config,
    "object-net": get_object_net_config,
    "few-nerd": get_few_nerd_config,
}


def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value`"

    k, v = key_value.split("=", maxsplit=1)

    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)

    return k, v
