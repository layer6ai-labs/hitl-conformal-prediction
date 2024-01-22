import torch
import torch.nn as nn

from networks import ConvNet, GoEmotionsModel, ObjectNetModel, FewNerdModel
from config import _VALID_DATASETS, _MODEL_KW


def get_fmnist_model(
    device, train_loader, val_loader, optimizer="adam", lr=0.001, epochs=2, **kwargs
):
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer.lower()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer} not implemented.")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}")
                running_loss = 0.0

    print("Finished Training")

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the validation images: {100 * correct // total} %"
    )

    return model


def get_go_emotions_model(device, train_loader, val_loader, **kwargs):
    used_labels = kwargs["used_labels"]
    model = GoEmotionsModel(used_labels).to(device)
    return model


def get_object_net_model(device, train_loader, val_loader, **kwargs):
    model = ObjectNetModel(device, kwargs["used_labels"], kwargs['model_size'])
    return model


def get_few_nerd_model(device, train_loader, val_loader, **kwargs):
    used_labels = kwargs["used_labels"]
    model = FewNerdModel(used_labels).to(device)
    return model


MODEL_FN_MAP = {
    "fashion-mnist": get_fmnist_model,
    "go-emotions": get_go_emotions_model,
    "object-net": get_object_net_model,
    "few-nerd": get_few_nerd_model,
}


def get_model(cfg, device, train_loader, val_loader, used_labels=None):
    dataset = cfg["dataset"]
    assert dataset in _VALID_DATASETS, f"Unknown dataset {dataset}"

    model_kwargs = get_model_kwargs(cfg)
    model_kwargs["used_labels"] = used_labels

    model_fn = MODEL_FN_MAP[dataset]
    model = model_fn(device, train_loader, val_loader, **model_kwargs)

    return model


def get_model_kwargs(cfg):
    # Create subdict of cfg for the keys relevant to models
    model_kw = _MODEL_KW
    model_kwargs = {k: cfg[k] for k in cfg.keys() & model_kw}

    return model_kwargs
