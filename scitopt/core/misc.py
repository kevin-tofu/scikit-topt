import os
import argparse
import re


def find_latest_iter_file(dst_path: str):
    pattern = re.compile(r"(\d{6})-rho\.npz")
    max_iter = -1
    latest_file = None

    for fname in os.listdir(dst_path):
        match = pattern.match(fname)
        if match:
            iter_num = int(match.group(1))
            if iter_num > max_iter:
                max_iter = iter_num
                latest_file = fname

    return max_iter, os.path.join(dst_path, latest_file) if latest_file else None


def str2bool(value):
    if isinstance(value, bool):
        return value
    elif value.lower() in ('true', 't'):
        return True
    elif value.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean values is expeted')
