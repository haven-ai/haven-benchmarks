import torch
import tqdm
from torch.utils.data import DataLoader


def get_metric_function(metric_name):
    if metric_name == "softmax_loss":
        return softmax_loss

    if metric_name == "softmax_accuracy":
        return softmax_accuracy


@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_name, batch_size=128):
    device = next(model.parameters()).device
    metric_function = get_metric_function(metric_name)

    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=batch_size)
    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    for batch in tqdm.tqdm(loader):
        images, labels = batch[0].to(
            device=device), batch[1].to(device=device)

        score_sum += metric_function(model, images,
                                     labels).item() * images.shape[0]

    score = float(score_sum / len(loader.dataset))

    return score


def softmax_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    loss = criterion(logits, labels.long().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc
