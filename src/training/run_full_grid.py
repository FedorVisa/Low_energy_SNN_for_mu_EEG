"""Run grid-search experiments across subjects, seeds, and training variants."""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_list(value):
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]


def parse_int_list(value):
    if not value:
        return []
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def best_known_variant_args(subject_id):
    best_known = {
        1: 'aug_base',
        2: 'aug_base',
        3: 'aug_boost',
        4: 'aug_boost',
        5: 'aug_weighted_ce',
        6: 'aug_base',
        7: 'aug_zscore',
        8: 'aug_boost',
        9: 'aug_base',
    }
    variant_name = best_known.get(subject_id, 'aug_base')
    return build_variants()[variant_name](subject_id)


def best_seed23_variant_args(subject_id):
    best_seed23 = {
        1: ['--aug_boost_subjects', str(subject_id), '--aug_boost_multiplier', '1.5'],
        2: ['--loss', 'ce', '--label_smoothing', '0.05'],
        3: ['--loss', 'weighted_ce'],
        4: ['--aug_boost_subjects', str(subject_id), '--aug_boost_multiplier', '1.5'],
        5: ['--loss', 'ce', '--label_smoothing', '0.05', '--EA', 'True'],
        6: ['--loss', 'focal', '--focal_gamma', '2.0', '--focal_alpha', '0.25'],
        7: [],
        8: ['--loss', 'weighted_ce'],
        9: ['--aug_boost_subjects', str(subject_id), '--aug_boost_multiplier', '1.5'],
    }
    return best_seed23.get(subject_id, [])


def build_variants():
    return {
        'aug_base': lambda sid: [],
        'aug_zscore': lambda sid: ['--zscore'],
        'aug_weighted_ce': lambda sid: ['--loss', 'weighted_ce'],
        'aug_boost': lambda sid: ['--aug_boost_subjects', str(sid), '--aug_boost_multiplier', '1.5'],
        'aug_focal': lambda sid: ['--loss', 'focal', '--focal_gamma', '2.0', '--focal_alpha', '0.25'],
        'aug_focal_g1': lambda sid: ['--loss', 'focal', '--focal_gamma', '1.0'],
        'aug_focal_g15': lambda sid: ['--loss', 'focal', '--focal_gamma', '1.5'],
        'aug_focal_g25': lambda sid: ['--loss', 'focal', '--focal_gamma', '2.5'],
        'aug_label_smooth_005': lambda sid: ['--loss', 'ce', '--label_smoothing', '0.05'],
        'aug_label_smooth_010': lambda sid: ['--loss', 'ce', '--label_smoothing', '0.10'],
        'best_known': best_known_variant_args,
        'best_seed23': best_seed23_variant_args,
    }


def run_one(cmd, log_path):
    with open(log_path, 'w', encoding='utf-8') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def parse_accuracy_from_log(log_path):
    pattern = re.compile(r"the test accuracy is ([0-9.]+)")
    last = None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last = float(match.group(1))
    return last


def main():
    parser = argparse.ArgumentParser(description='Run full grid over subjects/variants')
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--model', type=str, default='CUPY_SNN_LIF_READOUT')
    parser.add_argument('--epoch', type=int, default=1500)
    parser.add_argument('--epoch2', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--trial_num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--EA', action='store_true')
    parser.add_argument('--augment_train', action='store_true', default=True)
    parser.add_argument('--aug_copies', type=int, default=1)
    parser.add_argument('--aug_reverse_prob', type=float, default=0.25)
    parser.add_argument('--aug_gaussian_std_min', type=float, default=0.005)
    parser.add_argument('--aug_gaussian_std_max', type=float, default=0.02)
    parser.add_argument('--aug_scale_min', type=float, default=0.95)
    parser.add_argument('--aug_scale_max', type=float, default=1.05)
    parser.add_argument('--aug_shift_max', type=int, default=5)
    parser.add_argument('--subjects', type=str, default='')
    parser.add_argument('--variants', type=str, default='aug_base,aug_zscore,aug_weighted_ce,aug_boost')
    parser.add_argument('--log_dir', type=str, default=os.path.join('benchmarks', 'grid', 'logs'))
    parser.add_argument('--out_json', type=str, default=os.path.join('benchmarks', 'grid', 'grid_results.json'))
    parser.add_argument('--smoke', action='store_true', help='Run a quick smoke test')
    args = parser.parse_args()

    subjects = parse_int_list(args.subjects)
    if not subjects:
        subjects = list(range(1, 10))

    variants = parse_list(args.variants)
    variant_map = build_variants()

    if args.smoke:
        args.epoch = min(args.epoch, 2)
        args.epoch2 = min(args.epoch2, 1)
        args.trial_num = 1
        subjects = subjects[:1]
        variants = variants[:1]

    ensure_dir(args.log_dir)
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        ensure_dir(out_dir)

    results = {
        'dataset': args.dataset,
        'model': args.model,
        'epoch': args.epoch,
        'epoch2': args.epoch2,
        'batch_size': args.batch_size,
        'trial_num': args.trial_num,
        'seed': args.seed,
        'subjects': subjects,
        'variants': variants,
        'runs': [],
    }

    for sid in subjects:
        for variant in variants:
            if variant not in variant_map:
                raise ValueError(f'Unknown variant: {variant}')
            extra_args = variant_map[variant](sid)

            cmd = [
                sys.executable,
                'main.py',
                '--dataset', str(args.dataset),
                '--model', args.model,
                '--subject_id', str(sid),
                '--trial_num', str(args.trial_num),
                '--seed', str(args.seed),
                '--epoch', str(args.epoch),
                '--epoch2', str(args.epoch2),
                '--batch_size', str(args.batch_size),
                '--device', str(args.device),
                '--augment_train',
                '--aug_copies', str(args.aug_copies),
                '--aug_reverse_prob', str(args.aug_reverse_prob),
                '--aug_gaussian_std_min', str(args.aug_gaussian_std_min),
                '--aug_gaussian_std_max', str(args.aug_gaussian_std_max),
                '--aug_scale_min', str(args.aug_scale_min),
                '--aug_scale_max', str(args.aug_scale_max),
                '--aug_shift_max', str(args.aug_shift_max),
            ] + extra_args
            if args.EA:
                cmd += ['--EA', 'True']

            stamp = time.strftime('%Y%m%d_%H%M%S')
            log_name = f'subject{sid}_{variant}_{stamp}.log'
            log_path = os.path.join(args.log_dir, log_name)

            print('Running:', ' '.join(cmd))
            ret = run_one(cmd, log_path)
            acc = parse_accuracy_from_log(log_path)

            results['runs'].append({
                'subject': sid,
                'variant': variant,
                'return_code': ret,
                'accuracy': acc,
                'log_path': log_path,
            })

            with open(args.out_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            if ret != 0:
                print(f'Run failed for subject {sid} variant {variant} (code {ret}).')

    print('Saved', args.out_json)


if __name__ == '__main__':
    main()
