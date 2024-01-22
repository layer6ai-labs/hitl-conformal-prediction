import random
import argparse
import pprint
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_config, parse_config_arg
from writer import get_writer
from dataloader_factory import get_loaders
from model_factory import get_model
from raps import ConformalModel, validate, validate_topk
from naive import compute_k_empirically
from dataset_utils import *

import datasets
import time

datasets.logging.set_verbosity_error()
datasets.utils.logging.disable_progress_bar()


def main():
    start_time = time.time()
    # Parse args and create config dict
    parser = argparse.ArgumentParser(description="Generate conformal prediction sets.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument(
        "--config",
        default=[],
        action="append",
        help="Override config entries. Specify as `key=value`.",
    )

    args = parser.parse_args()

    cfg = get_config(dataset=args.dataset)
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10 * "-" + "cfg" + 10 * "-")
    pp.pprint(cfg)

    writer = get_writer(args, cfg=cfg)

    # Set random seeds for reproducibility
    np.random.seed(seed=cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # Get specified dataset in the form of loaders
    loader_dict = get_loaders(cfg)

    # Get model specific for each dataset, trained from scratch or loaded from saved weights
    used_labels = (
        loader_dict["top_m_labels"]
        if args.dataset in ["go-emotions", "object-net", "few-nerd"]
        else None
    )
    model = get_model(
        cfg, device, loader_dict["train"], loader_dict["val"], used_labels
    )

    if cfg["k"]:
        # compute top k first, and then use the same coverage guarantee for comformal
        k = cfg["k"]
        print(
            f"Compute coverage on calibration of top{k}, then select alpha of conformal based on the empirical coverage of top{k}."
        )
        topk = validate_topk(
            loader_dict["test"], model, device=device, k=k, dataset=cfg["dataset"]
        )
        print(f"Empirical coverage of top {k} prediction sets on the test set: {topk}")
        topk = validate_topk(
            loader_dict["calib"], model, device=device, k=k, dataset=cfg["dataset"]
        )
        print(
            f"Empirical coverage of top {k} prediction sets on the calibration set: {topk}"
        )

        cfg["alpha"] = 1 - topk
        print(f"Using alpha {round(cfg['alpha'], 4)}")

        cmodel = ConformalModel(
            model,
            loader_dict["calib"],
            alpha=cfg["alpha"],
            device=device,
            lamda_criterion="size",
            kreg=cfg["kreg"],
            lamda=cfg["lamda"],
            batch_size=cfg["calib_batch_size"],
            T=cfg["T"],
            dataset=cfg["dataset"],
        )
    else:
        # compute conformal first, and then choose k empirically
        alpha = cfg["alpha"]
        print(
            f"Compute coverage of conformal calibration with alpha={alpha}, then select k of topk based on empirical coverage."
        )
        cmodel = ConformalModel(
            model,
            loader_dict["calib"],
            alpha=alpha,
            device=device,
            lamda_criterion="size",
            kreg=cfg["kreg"],
            lamda=cfg["lamda"],
            batch_size=cfg["calib_batch_size"],
            T=cfg["T"],
            dataset=cfg["dataset"],
        )
        k = compute_k_empirically(cmodel)

    top1, topk, coverage, size = validate(
        loader_dict["test"],
        cmodel,
        print_bool=True,
        device=device,
        dataset=cfg["dataset"],
        k=k,
    )
    metrics = {
        "alpha": cfg["alpha"],
        "top1": top1,
        f"top{k}": topk,
        "coverage": coverage,
        "size": size,
    }
    writer.write_json("metrics", metrics)

    # Produce output csv
    columns = ["text_prompt", "label", "top1", f"top{k}_set", "conformal_set"]
    df = pd.DataFrame(columns=columns)
    with torch.no_grad():
        for data in tqdm(
            loader_dict["test"], desc="Generating output csv", disable=True
        ):
            if args.dataset == "go-emotions":
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                x = (input, attn)
                y = data["labels"].cpu().numpy()
                input_data = data["text"]
            elif args.dataset == "object-net":
                input = data["image"].to(device)
                x = input
                y = data["label"].cpu().numpy()
                input_data = data["image_id"]
            elif args.dataset == "few-nerd":  # Few-Nerd
                #time.sleep(2)  # hack to fix gpu wattage surge during forward pass on PC
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                position_ids = data["position_ids"].to(device)
                start_marker_index = data["start_marker_indices"].to(device)
                num_marker_pairs = data["num_marker_pairs"].to(device)
                span_index = data["required_span_index"]

                x = (
                    input,
                    attn,
                    position_ids,
                    start_marker_index,
                    num_marker_pairs,
                    span_index,
                )
                y = data["target"].cpu().numpy()
                input_data = data["id"]
            else:  # fashion-mnist
                assert len(data) == 2
                x, y = data
                x = x.to(device)
                y = y.cpu().numpy()
                input_data = x.cpu().numpy()

            output, S = cmodel(x)
            values, indices = output.topk(k, 1, largest=True, sorted=True)
            indices = indices.cpu().numpy()
            top1 = indices[:, :1]
            # We are not providing information on logit values or ordering
            topk_set = indices.tolist()
            for lst in topk_set:
                lst.sort()
                if lst[0] == 0:
                    lst.append(lst.pop(0))  # Move 0 to end for display purposes
            conf_set = [arr.tolist() for arr in S]
            for lst in conf_set:
                lst.sort()
                if lst[0] == 0:
                    lst.append(lst.pop(0))  # Move 0 to end for display purposes

            batch_dict = {}
            for i, y_i in enumerate(y):
                topk_str = ""
                for idx in topk_set[i]:
                    topk_str += f"{idx} "
                cp_set_str = ""
                for lab in conf_set[i]:
                    cp_set_str += f"{lab} "
                batch_dict[i] = [input_data[i], y_i, top1[i][0], topk_str, cp_set_str]

            df_batch = pd.DataFrame.from_dict(
                batch_dict, orient="index", columns=columns
            )
            df = pd.concat([df, df_batch], axis=0, ignore_index=True)

    if args.dataset == "fashion-mnist":
        df = process_fmnist_dataframe(df, k)
    if args.dataset == "go-emotions":
        df = process_go_emotions_dataframe(df, loader_dict["top_m_labels"], k)
    if args.dataset == "object-net":
        df = process_object_net_dataframe(df, loader_dict["top_m_labels"], k)
    if args.dataset == "few-nerd":
        df = process_few_nerd_dataframe(
            df,
            loader_dict["top_m_labels"],
            k,
            loader_dict["id2label_mappings"],
            loader_dict["test"].dataset,
        )

    cols = [
        "text_prompt",
        "label_text",
        "label",
        "original_label",
        "top1",
        f"top{k}_set",
        "conformal_set",
        "corr_ans_text",
        "conformal_text",
        f"top{k}_text",
    ]

    if args.dataset == "fashion-mnist":
        cols.pop(0)
    if args.dataset == "few-nerd":
        cols.extend(["original_label_text","text_prompt_with_original_fine_ner_tags"])
    df = df[cols]
    dataset = cfg["dataset"]
    writer.write_pandas(f"{dataset}", df)
    print(f"total time to run {dataset} dataset : {time.time() - start_time}")


if __name__ == "__main__":
    main()
