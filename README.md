# README

This is the codebase accompanying the paper ["Conformal Prediction Sets Improve Human Decision Making"](add_link_later). Here we discuss how to generate the datasets of conformal prediction sets used in the paper.

## Setup

The main prerequisite is to set up the python environment.

    conda create --name conformal python=3.10
    conda activate conformal
    conda install pytorch=2.0.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    conda install scipy tqdm pandas
    conda install transformers datasets 
    conda install -c conda-forge emoji  # For go-emotions
    pip install git+https://github.com/openai/CLIP.git  # For object-net
    pip install span_marker  # for Few-NERD

## Usage - `main.py`

The main script for creating datasets is unsurprisingly `main.py`.
This script loads raw datasets, splits them, loads or trains a model, performs conformal calibration, and generates conformal prediction sets for the test set data points.

The basic usage is as follows:

    python main.py --dataset <dataset>

where `<dataset>` is the desired dataset. We have implemented `fashion-mnist`, `go-emotions`, `object-net`, and `few-nerd`.

### Dynamic Updating of Config Values

Dataset and calibration hyperparameters are loaded from the `config.py` file at runtime.
However, it is also possible to update the hyperparameters on the command line using the flag `--config`.
For each hyperparameter `<key>` that one wants to set to a new value `<value>`, add the following to the command line:

    --config <key>=<value>

This can be done multiple times for multiple keys. A full list of config values is visible in the `config.py` file.

### Two options for experiments

To create a fair comparison between conformal and top-k prediction sets, we ensure their sets have the same coverage on the calibration set. This can be done in two ways:
1. Default config: Select k of top-k first, and compute alpha as (1 - top_k_coverage) on the calibration set. Then use that alpha for conformal prediction.
2. Alternative: Pick an alpha value for conformal prediction first, and then select k for topk empirically to match the coverage from conformal prediction. Since k must be an integer, we are not guaranteed to find a value of k which produces a similar alpha. Example command:
    ```
    python main.py --dataset go-emotions --config alpha=0.05 --config k=None
    ```

### Run Directories

By default, the `main` command above will create a directory of the form `logs/<date>_<hh>-<mm>-<ss>`, e.g. `Apr26_09-38-37`, to store information about the run, including:

- Config files as `json`
- Experiment metrics / results as `json`
- `stderr` / `stdout` logs
- Output csvs containing conformal prediction sets for the test data

## BibTeX

    To be added
