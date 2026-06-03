"""Train the official CUPY LIF readout configuration for MI EEG experiments."""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset

from tools import functional
from src import models
from tools.augmentations import EEGAugmentedDataset
from src.data import EEGset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser(description="Official-loop runner for CUPY_SNN_LIF_READOUT")
parser.add_argument("--dataset", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--lr2", type=float, default=1e-4)
parser.add_argument("--epoch", type=int, default=1500)
parser.add_argument("--epoch2", type=int, default=600)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--trial_num", type=int, default=1)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--patience", type=int, default=200)
parser.add_argument("--stage2_select", type=str, default="train_loss", choices=["train_loss", "val_acc"])
parser.add_argument("--stage2_patience", type=int, default=200)
parser.add_argument("--loo", type=bool, default=False)
parser.add_argument("--EA", type=bool, default=False)
parser.add_argument("--model", type=str, default="CUPY_SNN_LIF_READOUT")
parser.add_argument("--T", type=int, default=250 * 4)
parser.add_argument("--prep", type=str, default="250Hz_preprocess_eeg/")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--beta", type=float, default=2)
parser.add_argument("--subject_id", type=int, default=0)
parser.add_argument("--run_name", type=str, default="official_lif_readout")

parser.add_argument("--readout_v_threshold", type=float, default=0.2)
parser.add_argument("--readout_adapt_scale", type=float, default=0.02)
parser.add_argument("--readout_tau_adp_scale", type=float, default=6.0)
parser.add_argument("--readout_input_scale", type=float, default=2.5)
parser.add_argument("--encoder_threshold", type=float, default=0.5)
parser.add_argument("--encoder_scale", type=float, default=1.0)
parser.add_argument("--hidden_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lif_dropout", type=float, default=0.0)
parser.add_argument("--lif_v_threshold", type=float, default=0.5)
parser.add_argument("--lif_input_scale", type=float, default=2.5)

parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
parser.add_argument("--loss", type=str, default="ce", choices=["ce", "weighted_ce"])
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--zscore", action="store_true")
parser.add_argument("--train_stat_norm", action="store_true")
parser.add_argument("--train_stat_eps", type=float, default=1e-6)

parser.add_argument("--augment_train", action="store_true")
parser.add_argument("--aug_copies", type=int, default=1)
parser.add_argument("--aug_reverse_prob", type=float, default=0.0)
parser.add_argument("--aug_gaussian_std_min", type=float, default=0.005)
parser.add_argument("--aug_gaussian_std_max", type=float, default=0.02)
parser.add_argument("--aug_scale_min", type=float, default=0.95)
parser.add_argument("--aug_scale_max", type=float, default=1.05)
parser.add_argument("--aug_shift_max", type=int, default=5)


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, patience=10, path=None):
        self.patience = patience
        self.counter = 0
        self.val_min_acc = 0.0
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):
        if val_acc < self.val_min_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.val_min_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def build_model(prms):
    model_kwargs = {
        "in_channels": [22, 15, 60][prms["dataset"]],
        "out_num": [4, 2, 6][prms["dataset"]],
        "time_step": prms["T"],
        "beta": prms["beta"],
        "readout_v_threshold": prms["readout_v_threshold"],
        "readout_adapt_scale": prms["readout_adapt_scale"],
        "readout_tau_adp_scale": prms["readout_tau_adp_scale"],
        "readout_input_scale": prms["readout_input_scale"],
    }
    if prms["model"] in ["CUPY_SNN_SIGNED_LIF_MLP_READOUT", "CUPY_SNN_SPIKING_CONV_LIF_READOUT"]:
        model_kwargs.update(
            {
                "encoder_threshold": prms["encoder_threshold"],
                "encoder_scale": prms["encoder_scale"],
                "dropout": prms["dropout"],
                "lif_v_threshold": prms["lif_v_threshold"],
            }
        )
        if prms["model"] == "CUPY_SNN_SIGNED_LIF_MLP_READOUT":
            model_kwargs["hidden_layers"] = prms["hidden_layers"]
        if prms["model"] == "CUPY_SNN_SPIKING_CONV_LIF_READOUT":
            model_kwargs["lif_input_scale"] = prms["lif_input_scale"]
    if prms["model"] == "CUPY_SNN_LIF_PLIF_LIF_READOUT":
        model_kwargs.update(
            {
                "lif_v_threshold": prms["lif_v_threshold"],
                "lif_input_scale": prms["lif_input_scale"],
            }
        )
    if prms["model"] == "CUPY_SNN_LIF_READOUT":
        model_kwargs["dropout"] = prms["lif_dropout"]
    return getattr(models, prms["model"])(**model_kwargs).cuda()


def build_optimizer(parameters, lr, prms):
    if prms["optimizer"] == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=prms["weight_decay"],
        )
    return torch.optim.Adam(
        parameters,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=prms["weight_decay"],
    )


def build_scheduler(optimizer, epochs, prms):
    if prms["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=0.0,
        )
    return None


def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset:
        counts[int(label)] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_function(train_set, prms):
    if prms["loss"] == "weighted_ce":
        weights = compute_class_weights(train_set, [4, 2, 6][prms["dataset"]]).cuda()
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=prms["label_smoothing"]).cuda()
    return nn.CrossEntropyLoss(label_smoothing=prms["label_smoothing"]).cuda()


def maybe_augment(train_set, prms, seed):
    if not prms["augment_train"]:
        return train_set
    return EEGAugmentedDataset(
        train_set,
        copies_per_sample=prms["aug_copies"],
        reverse_prob=prms["aug_reverse_prob"],
        gaussian_std_min=prms["aug_gaussian_std_min"],
        gaussian_std_max=prms["aug_gaussian_std_max"],
        scale_min=prms["aug_scale_min"],
        scale_max=prms["aug_scale_max"],
        shift_max=prms["aug_shift_max"],
        seed=seed,
    )


class TrainStatNormalizedDataset(Dataset):
    def __init__(self, base_dataset, mean, std, eps=1e-6):
        self.base_dataset = base_dataset
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.eps = float(eps)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        x = np.asarray(x, dtype=np.float32)
        x = (x - self.mean) / (self.std + self.eps)
        return x.astype(np.float32), y


def train_channel_stats(train_set):
    count = 0
    sum_x = None
    sum_x2 = None
    for x, _ in train_set:
        x = np.asarray(x, dtype=np.float64)
        if sum_x is None:
            sum_x = np.zeros((x.shape[0], 1), dtype=np.float64)
            sum_x2 = np.zeros((x.shape[0], 1), dtype=np.float64)
        sum_x += x.sum(axis=1, keepdims=True)
        sum_x2 += (x * x).sum(axis=1, keepdims=True)
        count += x.shape[1]
    mean = sum_x / max(1, count)
    var = np.maximum(sum_x2 / max(1, count) - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def stage_one_train(net, train_set, validate_set, model_path, prms, loss_function):
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=prms["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        dataset=validate_set,
        batch_size=prms["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    optimizer = build_optimizer(net.parameters(), prms["lr"], prms)
    scheduler = build_scheduler(optimizer, prms["epoch"], prms)
    early_stop = EarlyStopping(patience=prms["patience"], path=model_path)
    train_loss = 0.0

    for epoch in range(prms["epoch"]):
        net.train()
        loss0 = 0.0
        train_num = 0
        num = 0
        for frame, label in train_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            loss0 += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_num += label.numel()
            num += 1
            functional.reset_net(net)
        train_loss = loss0 / max(1, num)
        if scheduler is not None:
            scheduler.step()

        net.eval()
        accuracy = 0.0
        val_num = 0
        with torch.no_grad():
            for frame, label in validate_data_loader:
                frame = frame.cuda()
                label = label.reshape(-1).cuda()
                out_fr = net(frame)
                accuracy += (out_fr.argmax(dim=1) == label).float().sum().item()
                val_num += label.numel()
                functional.reset_net(net)
        accuracy /= val_num
        early_stop(accuracy, net)
        if early_stop.early_stop:
            print("Early stopping at %d epoch" % (epoch))
            break

    net.load_state_dict(torch.load(model_path))
    return net, train_loss


def evaluate_accuracy(net, data_loader):
    net.eval()
    accuracy = 0.0
    total = 0
    with torch.no_grad():
        for frame, label in data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            accuracy += (out_fr.argmax(dim=1) == label).float().sum().item()
            total += label.numel()
            functional.reset_net(net)
    return accuracy / max(1, total)


def stage_two_train(net, train_set, validate_set, model_path, train_loss, prms, loss_function):
    if prms.get("stage2_select", "train_loss") == "val_acc":
        return stage_two_train_val_select(net, train_set, validate_set, model_path, prms, loss_function)
    combined_dataset = ConcatDataset([train_set, validate_set])
    combined_data_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=prms["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    net.load_state_dict(torch.load(model_path))
    optimizer = build_optimizer(net.parameters(), prms["lr2"], prms)
    scheduler = build_scheduler(optimizer, prms["epoch2"], prms)

    for epoch in range(prms["epoch2"]):
        net.train()
        loss0 = 0.0
        num = 0
        for frame, label in combined_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            loss0 += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num += 1
            functional.reset_net(net)
        loss0 /= max(1, num)
        if scheduler is not None:
            scheduler.step()
        torch.save(net.state_dict(), model_path)
        if loss0 < train_loss:
            print("current loss < stage 1 train loss at %d epoch" % (epoch))
            break

    net.load_state_dict(torch.load(model_path))
    return net


def stage_two_train_val_select(net, train_set, validate_set, model_path, prms, loss_function):
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=prms["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        dataset=validate_set,
        batch_size=prms["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    net.load_state_dict(torch.load(model_path))
    optimizer = build_optimizer(net.parameters(), prms["lr2"], prms)
    scheduler = build_scheduler(optimizer, prms["epoch2"], prms)
    stage2_patience = prms.get("stage2_patience", prms["patience"])
    best_val_acc = evaluate_accuracy(net, validate_data_loader)
    epochs_without_improvement = 0

    for epoch in range(prms["epoch2"]):
        net.train()
        for frame, label in train_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(net)
        if scheduler is not None:
            scheduler.step()

        accuracy = evaluate_accuracy(net, validate_data_loader)
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            epochs_without_improvement = 0
            torch.save(net.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= stage2_patience:
            print("Stage2 early stopping at %d epoch" % (epoch))
            break

    net.load_state_dict(torch.load(model_path))
    return net


def test(net, test_set, model_path, prms):
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=prms["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    net.load_state_dict(torch.load(model_path))
    test_acc = 0.0
    test_num = 0
    net.eval()
    with torch.no_grad():
        for frame, label in test_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            test_acc += (out_fr.argmax(dim=1) == label).float().sum().item()
            test_num += label.numel()
            functional.reset_net(net)
    return round(test_acc / test_num, 4)


def main():
    prms = vars(parser.parse_args())
    data_path = os.path.join(PROJECT_ROOT, "data", ["BNCI2014001", "BNCI2014002", "Weibo2014"][prms["dataset"]], prms["prep"])
    if prms["dataset"] == 2:
        torch.set_num_threads(2)
    torch.cuda.set_device(prms["device"])
    print(prms)

    subject_ids = list([range(1, 10), range(1, 15), range(1, 11)][prms["dataset"]])
    if prms["subject_id"]:
        subject_ids = [prms["subject_id"]]

    trial_acc = []
    run_records = []
    for trial_num in range(prms["trial_num"]):
        seedval = prms["seed"] + trial_num
        seed_torch(seedval)
        id_test_acc = []
        for ids in subject_ids:
            model_path = os.path.join(
                data_path,
                prms["model"]
                + f"_{prms['run_name']}_id{ids}_seed{seedval}"
                + ("_loo" if prms["loo"] else "")
                + ("_EA" if prms["EA"] else "")
                + ".pth",
            )
            net = build_model(prms)
            if trial_num == 0 and ids == 1:
                print(net)
            print(prms["model"] + f"_trial_num{trial_num}_id{ids}")

            train_base = EEGset(
                root_path=data_path,
                pick_id=(ids,),
                settup="train",
                T=prms["T"],
                loo=prms["loo"],
                all_id=[range(1, 10), range(1, 15), range(1, 11)][prms["dataset"]],
                EA=prms["EA"],
                zscore=prms["zscore"],
            )
            validate_set = EEGset(
                root_path=data_path,
                pick_id=(ids,),
                settup="validate",
                T=prms["T"],
                loo=prms["loo"],
                all_id=[range(1, 10), range(1, 15), range(1, 11)][prms["dataset"]],
                EA=prms["EA"],
                zscore=prms["zscore"],
            )
            test_set = EEGset(
                root_path=data_path,
                pick_id=(ids,),
                settup="test",
                T=prms["T"],
                loo=prms["loo"],
                all_id=[range(1, 10), range(1, 15), range(1, 11)][prms["dataset"]],
                EA=prms["EA"],
                zscore=prms["zscore"],
            )
            if prms["train_stat_norm"]:
                stat_mean, stat_std = train_channel_stats(train_base)
                train_base = TrainStatNormalizedDataset(train_base, stat_mean, stat_std, prms["train_stat_eps"])
                validate_set = TrainStatNormalizedDataset(validate_set, stat_mean, stat_std, prms["train_stat_eps"])
                test_set = TrainStatNormalizedDataset(test_set, stat_mean, stat_std, prms["train_stat_eps"])
            train_set = maybe_augment(train_base, prms, seedval)
            loss_function = build_loss_function(train_set, prms)
            net, train_loss = stage_one_train(net, train_set, validate_set, model_path, prms, loss_function)
            net = stage_two_train(net, train_set, validate_set, model_path, train_loss, prms, loss_function)
            test_acc = test(net, test_set, model_path, prms)
            print(f"the test accuracy is {test_acc}\n")
            id_test_acc.append(100 * test_acc)
            run_records.append({"seed": seedval, "subject": ids, "accuracy": 100 * test_acc})

        seed_mean = float(np.mean(np.array(id_test_acc)))
        print("seed{} mean is {}".format(seedval, seed_mean))
        trial_acc.append(id_test_acc)

    trial_acc = np.array(trial_acc)
    print(trial_acc)
    trial_mean = np.mean(trial_acc, axis=1).reshape(-1)
    result_mean = float(np.mean(trial_mean))
    result_std = float(math.sqrt(np.var(trial_mean)))
    summary = {
        "run_name": prms["run_name"],
        "model": prms["model"],
        "result_mean": result_mean,
        "result_std": result_std,
        "trial_means": trial_mean.tolist(),
        "records": run_records,
        "params": prms,
    }
    print("JSON_SUMMARY " + json.dumps(summary, sort_keys=True))
    print(f"results is: {result_mean}+-{result_std}")


if __name__ == "__main__":
    main()
