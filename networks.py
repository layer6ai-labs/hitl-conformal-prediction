import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
import clip
from span_marker import SpanMarkerModel


# CNN for Fashion-MNIST
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1, 2)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GoEmotionsModel(nn.Module):
    def __init__(self, used_labels):
        super(GoEmotionsModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "SamLowe/roberta-base-go_emotions",
            problem_type="multi_label_classification",
        )
        self.used_labels = used_labels

    def forward(self, x):
        input, attn = x
        out_logits = self.model(input, attn).logits
        # only take the used classes
        out_logits = out_logits[:, self.used_labels]
        return out_logits


class ObjectNetModel(nn.Module):
    def __init__(self, device, classnames, size="ViT-L/14"):
        super().__init__()
        self.device = device
        if size in ["ViT-B/32", "ViT-B/16", "ViT-L/14"]:
            self.clip = clip.load(size, device)[0]          
        else:
            raise ValueError(f"Invalid CLIP model size {size}")
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.classifier = self.zero_shot_classifier(
            classnames=classnames,
            templates=openai_imagenet_template,
        )

    def forward(self, images):
        image_features = self.clip.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)

        logits = image_features @ self.classifier

        return logits

    def zero_shot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [
                    template(classname) for template in templates
                ]  # format with class
                texts = clip.tokenize(texts, truncate=True).to(self.device)  # tokenize

                class_embeddings = self.clip.encode_text(texts).float()
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding = F.normalize(class_embedding, dim=0)
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights


openai_imagenet_template = [lambda c: f"{c}"]


class FewNerdModel(nn.Module):
    def __init__(self, used_labels):
        super(FewNerdModel, self).__init__()

        self.model = SpanMarkerModel.from_pretrained(
            "tomaarsen/span-marker-roberta-large-fewnerd-fine-super"
        )

        self.used_labels = used_labels

    def forward(self, x):
        input, attn, position_ids, start_marker_index, num_marker_pairs, span_index = x
        out_logits = self.model.forward(
            input, attn, position_ids, start_marker_index, num_marker_pairs
        ).logits

        # only take the required outputs
        result_tensors = []
        for i in range(len(span_index)):
            result_tensors.append(out_logits[i, span_index[i], self.used_labels])

        final_logits = torch.stack(result_tensors, dim=0)
        # output shape -> [num_batch, labels]
        return final_logits
