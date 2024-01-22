import torch

from raps import AverageMeter, accuracy


def compute_k_empirically(cmodel, sequential=False):
    print(
        "Computing which k to use for topk based on scaled logits, calibration data, and alpha"
    )
    with torch.no_grad():
        alpha = cmodel.alpha
        calib_loader = cmodel.calib_loader
        logits, _ = next(iter(calib_loader))
        K = logits.shape[1]

        if not sequential:
            # Compute accuracy for all k's at once
            # May be slow if K is extremely large and k is expected to be small
            ks = [i + 1 for i in range(K)]
            meters = []
            for i in range(K):
                meters.append(AverageMeter(f"top{i+1}"))
            for logits, targets in calib_loader:
                precs = accuracy(logits, targets, topk=ks)
                num_of_samples = len(targets)
                for meter, prec in zip(meters, precs):
                    meter.update(prec.item() / 100.0, n=num_of_samples)

            accs = []
            for meter in meters:
                acc = meter.avg
                accs.append(acc)
            print(f"Accuracies for increasing k: {accs}")
            for i in range(K):
                if accs[i] > 1 - alpha:
                    print(f"Using k={i+1}.")
                    return i + 1

        else:
            # Compute accuracy for each k sequentially,
            # and stop when desired error rate is reached
            for i in range(K):
                meter = AverageMeter(f"top{i+1}")
                for logits, targets in calib_loader:
                    prec = accuracy(logits, targets, topk=(i + 1,))[0]
                    num_of_samples = len(targets)
                    meter.update(prec.item() / 100.0, n=num_of_samples)

                acc = meter.avg
                print(f"At top {i+1}, accuracy is {acc}.")
                if acc > 1 - alpha:
                    print(f"Using k={i+1}.")
                    return i + 1

        print(f"No k found with accuracy better than {1-alpha}. Last was {acc}.")
        return K
