#!/usr/bin/env python
# coding: utf-8
"""
Test the generative model with own settings.

usage: batch_test_genList.py  [options]

options:
    --file_list_pth=<file list path>
    --output_pth=<output path>
    --mask_pth=<the mask file path>
"""
from docopt import docopt
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt
import os.path as op
import os
import math


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    file_list_pth = args['--file_list_pth']
    output_pth = args['--output_pth']
    mask_pth = args['--mask_pth']

    if output_pth is None:
        # output_pth = op.join(proj_pth, 'examples/esc50/mag_origin_npy/loss_test_output')
        raise ValueError("Please set the path for model's output.")
    if file_list_pth is None:
        # data_pth = op.join(proj_pth, 'examples/esc50/mag_origin_npy')
        raise ValueError("File list not set!")
    if mask_pth is None:
        # mask_pth = op.join(proj_pth, 'examples/esc50/mask_shape_1')
        raise ValueError("No mask output path be selected.")

    if op.isdir(output_pth) is False:
        os.mkdir(output_pth)

    with open(file_list_pth, 'r') as f:
        file_list = f.read().splitlines()

    out_batch_test_flist = op.join(output_pth, 'batch_test_list.txt')
    with open(out_batch_test_flist, 'w') as f:
        for filename in tqdm(sorted(file_list)):
            file_basename = op.basename(filename)
            image = filename
            mask = mask_pth
            output = op.join(output_pth, file_basename)
            f.write("%s %s %s\n" %(image, mask, output))
