# Code adapted from https://github.com/aangelopoulos/conformal_classification associated with the paper
# Angelopoulos et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction", ICLR 2021
# Published under the MIT License

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import time


def sort_sum(scores):
    I = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1)
    return I, ordered, cumsum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def validate(val_loader, model, print_bool, device, dataset, k=5):
    with torch.no_grad():
        batch_time = AverageMeter("batch_time")
        top1 = AverageMeter("top1")
        topk = AverageMeter(f"top{k}")
        coverage = AverageMeter("RAPS coverage")
        size = AverageMeter("RAPS size")
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, data in enumerate(val_loader):
            if dataset == "go-emotions":  # go emotions
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                target = data["labels"]
                x = (input, attn)
            elif dataset == "object-net":  # objectnet
                x = data["image"].to(device)
                target = data["label"]
            elif dataset == "few-nerd":  # Few-Nerd
                #time.sleep(2)  # hack to fix gpu wattage surge during forward pass on PC
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                position_ids = data["position_ids"].to(device)
                start_marker_index = data["start_marker_indices"].to(device)
                num_marker_pairs = data["num_marker_pairs"].to(device)
                id = data["id"]
                span_index = data["required_span_index"]

                x = (
                    input,
                    attn,
                    position_ids,
                    start_marker_index,
                    num_marker_pairs,
                    span_index,
                )
                target = data["target"]
            else:
                assert len(data) == 2
                x, target = data
                x = x.to(device)
            num_of_samples = len(target)
            target = target.to(device)
            # compute output
            output, S = model(x)
            # measure accuracy and record loss
            prec_1, prec_k = accuracy(output, target, topk=(1, k))
            cvg, sz = coverage_size(S, target)

            # Update meters
            top1.update(prec_1.item() / 100.0, n=num_of_samples)
            topk.update(prec_k.item() / 100.0, n=num_of_samples)
            coverage.update(cvg, n=num_of_samples)
            size.update(sz, n=num_of_samples)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + num_of_samples
            if print_bool:
                print(
                    f"\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@{k}: {topk.val:.3f} ({topk.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})",
                    end="",
                )
    if print_bool:
        print("")  # Endline

    return top1.avg, topk.avg, coverage.avg, size.avg


def validate_topk(val_loader, model, device, dataset, k=5):
    with torch.no_grad():
        topk = AverageMeter(f"top{k}")
        # switch to evaluate mode
        model.eval()
        for data in val_loader:
            if dataset == "go-emotions":  # go emotions
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                target = data["labels"]
                x = (input, attn)
            elif dataset == "object-net":  # objectnet
                x = data["image"].to(device)
                target = data["label"]
            elif dataset == "few-nerd":  # Few-Nerd
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                position_ids = data["position_ids"].to(device)
                start_marker_index = data["start_marker_indices"].to(device)
                num_marker_pairs = data["num_marker_pairs"].to(device)
                id = data["id"]
                span_index = data["required_span_index"]

                x = (
                    input,
                    attn,
                    position_ids,
                    start_marker_index,
                    num_marker_pairs,
                    span_index,
                )
                target = data["target"]
            else:
                assert len(data) == 2
                x, target = data
                x = x.to(device)
            num_of_samples = len(target)
            target = target.to(device)

            # compute output
            output = model(x)

            # measure accuracy and record loss
            prec_k = accuracy(output, target, topk=(k,))[0]

            # Update meters
            topk.update(prec_k.item() / 100.0, n=num_of_samples)

    return topk.avg


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if targets[i].item() in S[i]:
            covered += 1
        size = size + S[i].shape[0]
    return float(covered) / targets.shape[0], size / targets.shape[0]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets


def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(dataset, [n1, len(dataset) - n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset) - n1 - n2])
    return data1, data2


def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(
        dataset, [n1, dataset.tensors[0].shape[0] - n1]
    )
    data2, _ = torch.utils.data.random_split(
        temp, [n2, dataset.tensors[0].shape[0] - n1 - n2]
    )
    return data1, data2


def get_model(modelname):
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=True, progress=True)

    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=True, progress=True)

    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True, progress=True)

    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True, progress=True)

    elif modelname == "ResNeXt101":
        model = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)

    elif modelname == "VGG16":
        model = torchvision.models.vgg16(pretrained=True, progress=True)

    elif modelname == "ShuffleNet":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True, progress=True)

    elif modelname == "Inception":
        model = torchvision.models.inception_v3(pretrained=True, progress=True)

    elif modelname == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True, progress=True)

    else:
        raise NotImplementedError

    model.eval()
    model = torch.nn.DataParallel(model).cuda()

    return model


# Computes logits and targets from a model and loader
def get_logits_targets(model, loader, device, num_classes, dataset):
    logits = torch.zeros((len(loader.dataset), num_classes))
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print("Computing logits for model (only happens once).")
    with torch.no_grad():
        for data in tqdm(loader, disable=True):
            if dataset == "go-emotions":  # go emotions
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                targets = data["labels"]
                x = (input, attn)
            elif dataset == "object-net":  # objectnet
                x = data["image"].to(device)
                targets = data["label"]
            elif dataset == "few-nerd":  # Few-Nerd
                #time.sleep(2)  # hack to fix gpu wattage surge during forward pass on PC
                input = data["input_ids"].to(device)
                attn = data["attention_mask"].to(device)
                position_ids = data["position_ids"].to(device)
                start_marker_index = data["start_marker_indices"].to(device)
                num_marker_pairs = data["num_marker_pairs"].to(device)
                id = data["id"]
                span_index = data["required_span_index"]

                x = (
                    input,
                    attn,
                    position_ids,
                    start_marker_index,
                    num_marker_pairs,
                    span_index,
                )
                targets = data["target"]

            else:
                assert len(data) == 2
                x, targets = data
                x = x.to(device)
            num_samples = len(targets)
            batch_logits = model(x).detach().cpu()
            logits[i : (i + num_samples), :] = batch_logits
            labels[i : (i + num_samples)] = targets.cpu()
            i = i + num_samples

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())
    return dataset_logits


def get_logits_dataset(
    modelname,
    datasetname,
    datasetpath,
    cache=str(pathlib.Path(__file__).parent.absolute()) + "/experiments/.cache/",
):
    fname = cache + datasetname + "/" + modelname + ".pkl"

    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, "rb") as handle:
            return pickle.load(handle)

    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(modelname)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(datasetpath, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, pin_memory=True
    )

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader)

    # Save the dataset
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits
