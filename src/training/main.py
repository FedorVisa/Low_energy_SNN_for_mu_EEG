"""Main training entry point for within-subject and cross-subject MI EEG experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import functional, surrogate
from src.data import EEGset
import argparse
import numpy as np
import random
import os
from torch.utils.data import ConcatDataset
import math
from src import models
from tools.augmentations import EEGAugmentedDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser(description='PyTorch Spiking Neural Network for MI')
parser.add_argument('--dataset', type=int, default=0, help='Choose Dataset')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning Rate')
parser.add_argument('--lr2', type=float, default=1e-4, help='Stage two learning rate')
parser.add_argument('--epoch', type=int, default=1500, help='Number of epochs for stage one')
parser.add_argument('--epoch2', type=int, default=600, help='Number of epochs for stage two')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--trial_num', type=int, default=8, help='Number of repeated experiments')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--patience', type=int, default=200, help='Early Stop Tolerance')
parser.add_argument('--loo', type=bool, default=False, help='whether cross subject')
parser.add_argument('--EA', type=bool, default=False, help='whether EA')
parser.add_argument('--model', type=str, default='CUPY_SNN_2ALIF', help='Choose the model')
parser.add_argument('--T', type=int, default=250 * 4, help='Time step')
parser.add_argument('--prep', type=str, default='250Hz_preprocess_eeg/', help='Choose preprocess method')
parser.add_argument('--device', type=int, default=0, help='Choose GPU index')
parser.add_argument('--beta', type=float, default=2, help='Choose model hyper-parameter')
parser.add_argument('--readout_v_threshold', type=float, default=0.2, help='ALIF readout threshold')
parser.add_argument('--readout_adapt_scale', type=float, default=0.02, help='ALIF readout adaptation scale')
parser.add_argument('--readout_tau_adp_scale', type=float, default=6.0, help='ALIF readout adaptation tau multiplier')
parser.add_argument('--readout_input_scale', type=float, default=2.5, help='Scale factor before ALIF readout')
parser.add_argument('--head_tau_spread', type=float, default=0.2, help='Relative tau spread for parallel PLIF heads')
parser.add_argument('--head_vth_spread', type=float, default=0.1, help='Threshold spread for parallel PLIF heads')
parser.add_argument('--head_dropout', type=float, default=0.1, help='Dropout probability for merged parallel heads')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer: adam or adamw')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
parser.add_argument('--augment_train', action='store_true', help='Apply data augmentation to the training set only')
parser.add_argument('--aug_copies', type=int, default=2, help='Number of augmented copies per original training sample')
parser.add_argument('--aug_reverse_prob', type=float, default=0.5, help='Probability of time-reversing an augmented sample')
parser.add_argument('--aug_gaussian_std_min', type=float, default=0.01, help='Minimum gaussian noise std for augmentation')
parser.add_argument('--aug_gaussian_std_max', type=float, default=0.05, help='Maximum gaussian noise std for augmentation')
parser.add_argument('--aug_scale_min', type=float, default=0.9, help='Minimum amplitude scale for augmentation')
parser.add_argument('--aug_scale_max', type=float, default=1.1, help='Maximum amplitude scale for augmentation')
parser.add_argument('--aug_shift_max', type=int, default=20, help='Maximum circular time shift for augmentation')
parser.add_argument('--preset', type=str, default='baseline', choices=['baseline', 'augmented_smoke', 'augmented_long'], help='Training preset for the pipeline')
parser.add_argument('--subject_id', type=int, default=0, help='Run a single subject id (0 for all subjects)')
parser.add_argument('--zscore', action='store_true', help='Apply per-trial z-score normalization')
parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'weighted_ce', 'focal'], help='Loss function')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Cross entropy label smoothing')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
parser.add_argument('--focal_alpha', type=float, default=-1.0, help='Focal loss alpha (set >=0 to enable)')
parser.add_argument('--aug_boost_subjects', type=str, default='', help='Comma-separated subject ids to boost augmentation')
parser.add_argument('--aug_boost_multiplier', type=float, default=1.5, help='Boost multiplier for augmentation strength')
prms = vars(parser.parse_args())


def apply_preset(prms):
    preset = prms['preset']
    if preset == 'baseline':
        return prms

    if preset in ['augmented_smoke', 'augmented_long']:
        prms['augment_train'] = True
        prms['aug_copies'] = 1
        prms['aug_reverse_prob'] = 0.25
        prms['aug_gaussian_std_min'] = 0.005
        prms['aug_gaussian_std_max'] = 0.02
        prms['aug_scale_min'] = 0.95
        prms['aug_scale_max'] = 1.05
        prms['aug_shift_max'] = 5

    if preset == 'augmented_smoke':
        prms['epoch'] = min(prms['epoch'], 40)
        prms['epoch2'] = min(prms['epoch2'], 20)
        prms['patience'] = min(prms['patience'], 20)
        prms['trial_num'] = min(prms['trial_num'], 1)
    elif preset == 'augmented_long':
        prms['epoch'] = min(prms['epoch'], 40)
        prms['epoch2'] = min(prms['epoch2'], 20)
        prms['patience'] = min(prms['patience'], 20)
        prms['trial_num'] = min(prms['trial_num'], 1)

    return prms


prms = apply_preset(prms)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        logp_t = logp.gather(1, targets.view(-1, 1)).squeeze(1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            if self.alpha.numel() == 1:
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
        loss = -alpha_t * ((1.0 - p_t) ** self.gamma) * logp_t
        return loss.mean()


def parse_subject_list(value):
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(',') if x.strip()}


def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset:
        counts[int(label)] += 1
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_function(train_set, prms, num_classes):
    loss_name = prms['loss']
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=prms['label_smoothing']).cuda()

    if loss_name == 'weighted_ce':
        weights = compute_class_weights(train_set, num_classes).cuda()
        return nn.CrossEntropyLoss(weight=weights).cuda()

    if loss_name == 'focal':
        alpha_value = prms['focal_alpha']
        alpha = None
        if alpha_value >= 0.0:
            alpha = torch.tensor([alpha_value], device='cuda', dtype=torch.float32)
        return FocalLoss(gamma=prms['focal_gamma'], alpha=alpha).cuda()

    return nn.CrossEntropyLoss().cuda()


def get_aug_params(prms, subject_id):
    params = {
        'copies_per_sample': prms['aug_copies'],
        'reverse_prob': prms['aug_reverse_prob'],
        'gaussian_std_min': prms['aug_gaussian_std_min'],
        'gaussian_std_max': prms['aug_gaussian_std_max'],
        'scale_min': prms['aug_scale_min'],
        'scale_max': prms['aug_scale_max'],
        'shift_max': prms['aug_shift_max'],
    }
    boost_subjects = parse_subject_list(prms['aug_boost_subjects'])
    if boost_subjects and subject_id in boost_subjects:
        mult = float(prms['aug_boost_multiplier'])
        params['copies_per_sample'] = max(1, int(round(params['copies_per_sample'] * mult)))
        params['reverse_prob'] = min(1.0, params['reverse_prob'] * mult)
        params['gaussian_std_min'] = params['gaussian_std_min'] * mult
        params['gaussian_std_max'] = params['gaussian_std_max'] * mult
        params['shift_max'] = int(round(params['shift_max'] * mult))
    return params


def setup_cuda_device(requested_device):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, but this training script requires CUDA.')
    device_count = torch.cuda.device_count()
    if requested_device < 0 or requested_device >= device_count:
        print(f"[Warning] Requested --device {requested_device} is out of range for {device_count} GPU(s). Falling back to cuda:0.")
        requested_device = 0
    torch.cuda.set_device(requested_device)
    return requested_device

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        self.best_val_acc = -1.0
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):
        # "No increase" means plateau also increases patience counter.
        if val_acc <= self.best_val_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def build_optimizer(parameters, lr, prms):
    optimizer_name = prms['optimizer'].lower()
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=prms['weight_decay'])
    return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=prms['weight_decay'])


def evaluate_average_loss(net, data_loader, loss_function):
    net.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for frame, label in data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            total_loss += float(loss.item())
            num_batches += 1
            functional.reset_net(net)
    return total_loss / max(1, num_batches)

def stage_one_train(net, train_set, validate_set, model_path, prms, loss_function):
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=prms['batch_size'],
        shuffle=True,
        drop_last=True
    )
    validate_data_loader = torch.utils.data.DataLoader(
        dataset=validate_set,
        batch_size=prms['batch_size'],
        shuffle=False,
        drop_last=False
    )
    optimizer = build_optimizer(net.parameters(), prms['lr'], prms)
    EarlyStop = EarlyStopping(patience=prms['patience'], path=model_path)
    # stage1
    for epoch in range(prms['epoch']):
        net.train()
        accuracy = 0
        loss0 = 0
        train_num = 0
        num = 0
        for frame, label in train_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            loss0 += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy += (out_fr.argmax(dim=1) == label.cuda()).float().sum().item()
            train_num += label.numel()
            num += 1
            functional.reset_net(net)
        accuracy /= train_num
        loss0 /= num
        net.eval()
        accuracy = 0
        loss0 = 0
        val_num = 0
        num = 0
        with torch.no_grad():
            for frame, label in validate_data_loader:
                frame = frame.cuda()
                label = label.reshape(-1).cuda()
                out_fr = net(frame)
                loss = loss_function(out_fr, label)
                loss0 += loss
                accuracy += (out_fr.argmax(dim=1) == label.cuda()).float().sum().item()
                val_num += label.numel()
                num += 1
                functional.reset_net(net)
            accuracy /= val_num
            loss0 /= num
        EarlyStop(accuracy, net)
        if EarlyStop.early_stop:
            print("Early stopping at %d epoch" % (epoch))
            break

    # Restore best stage1 checkpoint by validation accuracy.
    net.load_state_dict(torch.load(model_path))
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=prms['batch_size'],
        shuffle=False,
        drop_last=False
    )
    stage1_train_loss = evaluate_average_loss(net, train_eval_loader, loss_function)
    return net, stage1_train_loss

def stage_two_train(net, train_set, validate_set, model_path, train_loss, prms, loss_function):
    combined_dataset = ConcatDataset([train_set, validate_set])
    combined_data_loader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=prms['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    validate_data_loader = torch.utils.data.DataLoader(
        dataset=validate_set,
        batch_size=prms['batch_size'],
        shuffle=False,
        drop_last=False
    )
    net.load_state_dict(torch.load(model_path))
    optimizer = build_optimizer(net.parameters(), prms['lr2'], prms)
    # Keep stage1 checkpoint unless stage2 reaches lower validation loss than stage1 train loss.
    best_stage2_val_loss = float(train_loss)
    stage2_improved = False
    for epoch in range(prms['epoch2']):
        net.train()
        loss0 = 0
        train_num = 0
        num = 0
        for frame, label in combined_data_loader:
            frame = frame.cuda()
            label = label.reshape(-1).cuda()
            out_fr = net(frame)
            loss = loss_function(out_fr, label)
            loss0 += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_num += label.numel()
            num += 1
            functional.reset_net(net)

        current_stage2_val_loss = evaluate_average_loss(net, validate_data_loader, loss_function)
        if current_stage2_val_loss < best_stage2_val_loss:
            best_stage2_val_loss = current_stage2_val_loss
            stage2_improved = True
            torch.save(net.state_dict(), model_path)

        if current_stage2_val_loss < train_loss:
            print("current loss < stage 1 train loss at %d epoch" % (epoch))
            break

    if not stage2_improved:
        print("stage 2 did not improve over initialized checkpoint; keep stage 1 best checkpoint")
    net.load_state_dict(torch.load(model_path))
    return net

def test(net, test_set, model_path, prms):
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=prms['batch_size'],
        shuffle=False,
        drop_last=False
    )
    net.load_state_dict(torch.load(model_path))
    test_acc = 0
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
        test_acc /= test_num
        test_acc = round(test_acc, 4)
    return test_acc

if __name__ == '__main__':
    data_path = os.path.join(PROJECT_ROOT, 'data', ['BNCI2014001', 'BNCI2014002', 'Weibo2014'][prms['dataset']], prms['prep'])
    if prms['model'] in ['FBCNet']:
        data_path = os.path.join(data_path, 'filterbank')
    if prms['dataset'] == 2: torch.set_num_threads(2)
    prms['device'] = setup_cuda_device(prms['device'])
    print(prms)
    patience = prms['patience']
    trial_acc = []
    for trial_num in range(prms['trial_num']):
        seedval = prms['seed'] + trial_num
        seed_torch(seedval)
        id_test_acc = []
        subject_ids = list([range(1, 10), range(1, 15), range(1, 11)][prms['dataset']])
        if prms['subject_id']:
            if prms['subject_id'] not in subject_ids:
                raise ValueError(f"subject_id {prms['subject_id']} is not valid for dataset {prms['dataset']}")
            subject_ids = [prms['subject_id']]
        for ids in subject_ids:
            model_path = os.path.join(
                data_path,
                prms['model'] + f'_id{ids}_seed{seedval}' + ('_loo' if prms['loo'] else '') + ('_EA' if prms['EA'] else '') + '.pth',
            )
            if prms['model'] in ['ShallowConvNet', 'deepconv', 'EEGNet']:
                net = getattr(models, prms['model'])(in_channels=[22, 15, 60][prms['dataset']], time_step=prms['T'], classes_num=[4, 2, 6][prms['dataset']]).cuda()
            elif prms['model'] in ['FBCNet']:
                net = getattr(models, prms['model'])(nChan=[22, 15, 60][prms['dataset']], nTime=prms['T'], nClass=[4, 2, 6][prms['dataset']]).cuda()
            elif prms['model'] in ['CUPY_SNN_PLIF', 'CUPY_SNN_2ALIF', 'CUPY_SNN_2PLIF', 'CUPY_SNN_ALIF_READOUT', 'CUPY_SNN_LIF_READOUT', 'CUPY_SNN_SIGNED_LIF_MLP_READOUT', 'CUPY_SNN_LIF_PLIF_LIF_READOUT', 'CUPY_SNN_PLIF_DUAL_READOUT', 'CUPY_SNN_3PLIF_LN_ALIF_READOUT', 'CUPY_SNN_3PLIF_PARALLEL', 'CUPY_SNN_3PLIF_PARALLEL_ALIF_READOUT']:
                model_kwargs = {
                    'in_channels': [22, 15, 60][prms['dataset']],
                    'out_num': [4, 2, 6][prms['dataset']],
                    'time_step': prms['T'],
                    'beta': prms['beta'],
                }
                if prms['model'] in ['CUPY_SNN_ALIF_READOUT', 'CUPY_SNN_LIF_READOUT', 'CUPY_SNN_SIGNED_LIF_MLP_READOUT', 'CUPY_SNN_LIF_PLIF_LIF_READOUT', 'CUPY_SNN_PLIF_DUAL_READOUT', 'CUPY_SNN_3PLIF_LN_ALIF_READOUT', 'CUPY_SNN_3PLIF_PARALLEL_ALIF_READOUT']:
                    model_kwargs.update({
                        'readout_v_threshold': prms['readout_v_threshold'],
                        'readout_adapt_scale': prms['readout_adapt_scale'],
                        'readout_tau_adp_scale': prms['readout_tau_adp_scale'],
                        'readout_input_scale': prms['readout_input_scale'],
                    })
                if prms['model'] in ['CUPY_SNN_3PLIF_PARALLEL', 'CUPY_SNN_3PLIF_PARALLEL_ALIF_READOUT']:
                    model_kwargs.update({
                        'head_tau_spread': prms['head_tau_spread'],
                        'head_vth_spread': prms['head_vth_spread'],
                        'head_dropout': prms['head_dropout'],
                    })
                net = getattr(models, prms['model'])(**model_kwargs).cuda()
            if trial_num == 0 and ids == 1: print(net)
            print(prms['model'] + f'_trial_num{trial_num}_id{ids}')
            train_set = EEGset(root_path=data_path, pick_id=(ids,), settup='train', T=prms['T'], loo=prms['loo'], all_id=[range(1, 10), range(1, 15), range(1, 11)][prms['dataset']], EA=prms['EA'], zscore=prms['zscore'])
            if prms['augment_train']:
                aug_params = get_aug_params(prms, ids)
                train_set = EEGAugmentedDataset(
                    train_set,
                    copies_per_sample=aug_params['copies_per_sample'],
                    reverse_prob=aug_params['reverse_prob'],
                    gaussian_std_min=aug_params['gaussian_std_min'],
                    gaussian_std_max=aug_params['gaussian_std_max'],
                    scale_min=aug_params['scale_min'],
                    scale_max=aug_params['scale_max'],
                    shift_max=aug_params['shift_max'],
                    seed=seedval,
                )
            validate_set = EEGset(root_path=data_path, pick_id=(ids,), settup='validate', T=prms['T'], loo=prms['loo'], all_id=[range(1, 10), range(1, 15), range(1, 11)][prms['dataset']], EA=prms['EA'], zscore=prms['zscore'])
            test_set = EEGset(root_path=data_path, pick_id=(ids,), settup='test', T=prms['T'], loo=prms['loo'], all_id=[range(1, 10), range(1, 15), range(1, 11)][prms['dataset']], EA=prms['EA'], zscore=prms['zscore'])
            loss_function = build_loss_function(train_set, prms, [4, 2, 6][prms['dataset']])
            net, train_loss = stage_one_train(net, train_set, validate_set, model_path, prms, loss_function)
            net = stage_two_train(net, train_set, validate_set, model_path, train_loss, prms, loss_function)
            test_acc = test(net, test_set, model_path, prms)
            print(f'the test accuracy is {test_acc}\n')
            id_test_acc.append(100 * test_acc)
        print('seed{} mean is {}'.format(seedval, np.mean(np.array(id_test_acc))))
        trial_acc.append(id_test_acc)
    trial_acc = np.array(trial_acc)
    print(trial_acc)
    id_mean = np.mean(trial_acc, axis=0).reshape(-1)
    id_var = np.var(trial_acc, axis=0).reshape(-1)
    trial_mean = np.mean(trial_acc, axis=1).reshape(-1)
    result_mean = np.mean(trial_mean)
    result_var = np.var(trial_mean)
    print(f'results is: {result_mean}±{np.sqrt(result_var)}')
