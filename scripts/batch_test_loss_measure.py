#!/usr/bin/env python
# coding: utf-8
"""
Test the generative model with own settings.

usage: batch_test_loss_measure.py  [options]

options:
    --ref_spec_pth=<ref_spec path>
    --loss_type=<"psnr", "l1", "mean_l1">
    --output_pth=<output path>
"""
from docopt import docopt
from pathlib import Path
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt
import os.path as op
import cv2
import os
import math


def l1_loss(x, y):
    error = abs(x - y)
    return sum(sum(error))


def mean_l1_loss(x, y):
     error = abs(x - y)
     return np.mean(error)


def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    ref_spec_pth = args['--ref_spec_pth']
    loss_type = args['--loss_type'] # psnr, l1
    output_pth = args['--output_pth']

    if output_pth is None:
        # output_pth = op.join(proj_pth, 'examples/esc50/mag_origin_npy/loss_test_output')
        raise ValueError("Please set the path for model's output.")
    if ref_spec_pth is None:
        # data_pth = op.join(proj_pth, 'examples/esc50/mag_origin_npy')
        raise ValueError("Path of reference spec not set!")
    if loss_type is None:
        loss_type = 'mean_l1'

    if op.isdir(output_pth) is False:
        # os.mkdir(output_pth)
        raise ValueError("Output path should exist with batch test flist.")

    # Measure loss of output
    print("Measure the loss by each category...")
    dict_category_list = dict()
    for i, filename in tqdm(enumerate(sorted(Path(output_pth).glob('*.npy')))):
        # print(i, filename)
        file_basename = op.basename(filename)
        category = file_basename.split('.')[0].split('_')[0].split('-')[-1]
        if int(category) < 10:
            category = '0' + category
        if dict_category_list.get(category) is None:
            dict_category_list[category] = []

        ref_spec_filename = op.join(ref_spec_pth, file_basename)
        ref_spec = np.load(ref_spec_filename)
        h, w, _ = ref_spec.shape
        ref_spec = ref_spec.reshape((h, w))

        inpaint_spec = np.load(filename)
        h, w, _ = inpaint_spec.shape
        inpaint_spec = inpaint_spec.reshape((h, w))

        if ref_spec.shape != inpaint_spec.shape:
            raise ValueError("Mismatch of spec shape:", ref_spec.shape, inpaint_spec.shape)

        if loss_type == 'l1':
            cur_loss = l1_loss(ref_spec, inpaint_spec)
        elif loss_type == 'mean_l1':
            cur_loss = mean_l1_loss(ref_spec, inpaint_spec)
        elif loss_type == 'psnr':
            cur_loss = psnr(ref_spec, inpaint_spec)
        else:
            raise ValueError("Wrong type of loss type:", loss_type)
        dict_category_list[category].append(cur_loss)

        # If the diretory of loss not exist, create it.
    loss_pth = op.join(output_pth, 'loss')
    mask_type = op.basename(output_pth).split('_')[-1]
    if op.isdir(loss_pth) is False:
        os.mkdir(loss_pth)
    loss_filename = op.join(loss_pth, f'{mask_type}_{loss_type}_loss.txt')
    with open(loss_filename, 'w') as f:
        # f.write("#list_id, min, max, mean, percentile")
        for a_list in sorted(dict_category_list):
            f.write(f'{dict_category_list[a_list]}\n')
            #arr = np.array(sorted(dict_category_list[a_list]))
            #s1, s2, s3 = np.percentile(arr, [25, 50, 75])
            #f.write("%s %s %s %s %s %s %s\n" %(a_list, arr.min(), arr.max(), arr.mean(), s1, s2, s3))
    print("Save loss file to:", loss_filename)
