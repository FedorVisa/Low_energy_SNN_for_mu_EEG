"""Benchmark inference latency and throughput for selected EEG classification models."""

import argparse
import os
from statistics import mean, pstdev

import torch
from torch.utils.data import DataLoader

from tools import functional
from src import models
from src.data import EEGset


DATASET_CHANNELS = [22, 15, 60]
DATASET_CLASSES = [4, 2, 6]
DATASET_SUBJECTS = [range(1, 10), range(1, 15), range(1, 11)]
DEFAULT_MODELS = [
    'ShallowConvNet',
    'deepconv',
    'EEGNet',
    'CUPY_SNN_PLIF',
    'CUPY_SNN_LIF_READOUT',
    'CUPY_SNN_SPIKING_CONV_LIF_READOUT',
]


def setup_cuda_device(requested_device):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, but this benchmark requires CUDA.')
    device_count = torch.cuda.device_count()
    if requested_device < 0 or requested_device >= device_count:
        print(f"[Warning] Requested --device {requested_device} is out of range for {device_count} GPU(s). Falling back to cuda:0.")
        requested_device = 0
    torch.cuda.set_device(requested_device)
    return requested_device


def build_data_path(dataset, prep):
    root = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = ['BNCI2014001/', 'BNCI2014002/', 'Weibo2014/'][dataset]
    return os.path.join(root, 'data', dataset_folder, prep)


def build_model(model_name, dataset, time_step, beta):
    in_channels = DATASET_CHANNELS[dataset]
    out_num = DATASET_CLASSES[dataset]

    if model_name in ['ShallowConvNet', 'deepconv', 'EEGNet']:
        return getattr(models, model_name)(
            classes_num=out_num,
            in_channels=in_channels,
            time_step=time_step,
        ).cuda()

    if model_name in ['CUPY_SNN_PLIF', 'CUPY_SNN_LIF_READOUT', 'CUPY_SNN_SPIKING_CONV_LIF_READOUT']:
        return getattr(models, model_name)(
            in_channels=in_channels,
            out_num=out_num,
            time_step=time_step,
            beta=beta,
        ).cuda()

    raise ValueError(f'Unsupported model: {model_name}')


def load_batch(dataset, prep, batch_size, time_step, loo, ea):
    data_path = build_data_path(dataset, prep)
    subject_ids = DATASET_SUBJECTS[dataset]
    benchmark_set = EEGset(
        root_path=data_path,
        pick_id=(1,),
        settup='test',
        T=time_step,
        loo=loo,
        all_id=subject_ids,
        EA=ea,
    )
    loader = DataLoader(
        dataset=benchmark_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    frame, label = next(iter(loader))
    return frame.cuda(non_blocking=True), label.reshape(-1).cuda(non_blocking=True)


def benchmark_model(net, frame, warmup, repeat):
    net.eval()
    times_ms = []

    with torch.inference_mode():
        for _ in range(warmup):
            _ = net(frame)
            torch.cuda.synchronize()
            functional.reset_net(net)

        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()
            _ = net(frame)
            end.record()
            torch.cuda.synchronize()

            times_ms.append(start.elapsed_time(end))
            functional.reset_net(net)

    return times_ms


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference time for EEG models')
    parser.add_argument('--dataset', type=int, default=0, help='Choose Dataset')
    parser.add_argument('--prep', type=str, default='250Hz_preprocess_eeg/', help='Choose preprocess method')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for timing')
    parser.add_argument('--time_step', type=int, default=250 * 4, help='Time step')
    parser.add_argument('--beta', type=float, default=2, help='Choose model hyper-parameter for SNNs')
    parser.add_argument('--device', type=int, default=0, help='Choose GPU index')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations before timing')
    parser.add_argument('--repeat', type=int, default=30, help='Timed iterations')
    parser.add_argument('--loo', action='store_true', help='Whether cross subject')
    parser.add_argument('--EA', action='store_true', help='Whether EA')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Models to benchmark')
    args = parser.parse_args()

    args.device = setup_cuda_device(args.device)
    frame, label = load_batch(args.dataset, args.prep, args.batch_size, args.time_step, args.loo, args.EA)
    actual_batch_size = frame.shape[0]

    print({
        'dataset': args.dataset,
        'prep': args.prep,
        'time_step': args.time_step,
        'batch_size': actual_batch_size,
        'warmup': args.warmup,
        'repeat': args.repeat,
        'models': args.models,
        'device': args.device,
    })

    results = []
    for model_name in args.models:
        net = build_model(model_name, args.dataset, args.time_step, args.beta)
        times_ms = benchmark_model(net, frame, args.warmup, args.repeat)
        avg_ms = mean(times_ms)
        std_ms = pstdev(times_ms) if len(times_ms) > 1 else 0.0
        per_sample_ms = avg_ms / actual_batch_size
        throughput = actual_batch_size / (avg_ms / 1000.0)

        results.append((model_name, avg_ms, std_ms, per_sample_ms, throughput))
        print(
            f'{model_name:24s} | '
            f'{avg_ms:8.3f} ms/batch | '
            f'{std_ms:7.3f} ms std | '
            f'{per_sample_ms:8.4f} ms/sample | '
            f'{throughput:8.2f} samples/s'
        )

        del net
        torch.cuda.empty_cache()

    results.sort(key=lambda item: item[1])
    print('\nRanked by average batch latency:')
    for rank, (model_name, avg_ms, std_ms, per_sample_ms, throughput) in enumerate(results, start=1):
        print(
            f'{rank}. {model_name}: {avg_ms:.3f} ms/batch, '
            f'{per_sample_ms:.4f} ms/sample, {throughput:.2f} samples/s'
        )


if __name__ == '__main__':
    main()
