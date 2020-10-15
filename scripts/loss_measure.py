#!/usr/bin/env python
# coding: utf-8
"""
Test the generative model with own settings.

usage: loss_measure.py  [options]

options:
    --project_pth=<test file directory>
    --data_pth=<data path>
    --loss_type=<"psnr", "l1", "mean_l1">
    --output_pth=<output path>
    --checkpoint=<model path>
    --mask_pth=<mask path>
    --mask_type=<mask type>
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


def create_mask(length, mask_pth, mask_type='square'):
    # For sorted.
    if length < 100:
        output_pth = op.join(mask_pth, f'mask_{mask_type}_0{length}')
    else:
        output_pth = op.join(mask_pth, f'mask_{mask_type}_{length}')

    if op.isfile(output_pth):
        return output_pth

    h, w = spec.shape
    mid_point = h // 2, w // 2
    # print(mid_point, length)

    mask = np.zeros((h, w, 3), np.uint8)
    if mask_type == 'square':
        t = mid_point[0] - (length // 2)
        b = mid_point[0] + (length // 2)
        l = mid_point[1] - (length // 2)
        r = mid_point[1] + (length // 2)
        # print(t, b, l, r)
        mask = cv2.rectangle(mask, (l, t), (r, b), (255, 255, 255), -1)
    elif mask_type == 'time':
        area = length * length
        constant = 10
        if area < h * constant:
            height = area // constant
            t = mid_point[0] - (height // 2)
            b = mid_point[0] + (height // 2)
            l = mid_point[1] - (constant // 2)
            r = mid_point[1] + (constant // 2)
        else:
            width = area // h
            t = 0
            b = h
            l = mid_point[1] - (width // 2)
            r = mid_point[1] + (width // 2)
        # print(t, b, l, r)
        mask = cv2.rectangle(mask, (l, t), (r, b), (255, 255, 255), -1)
    else:
        raise ValueError("Wrong type of mask:", mask_type)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # print(gray[110][:])

    # This way is for showing the visualized results then convert to 1-channel.
    mask = mask[:, :, np.newaxis]
    # print(square_mask.shape)

    np.save(output_pth, mask)
    return output_pth + '.npy'


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    proj_pth = args['--project_pth']
    data_pth = args['--data_pth']
    loss_type = args['--loss_type'] # psnr, l1
    output_pth = args['--output_pth']
    checkpoint = args['--checkpoint']
    mask_pth = args['--mask_pth']
    mask_type = args['--mask_type']

    if proj_pth is None:
         proj_pth = '/work/r08922a13/generative_inpainting'
    if output_pth is None:
        output_pth = op.join(proj_pth, 'examples/esc50/mag_origin_npy/loss_test_output')
    if checkpoint is None:
        checkpoint = op.join(proj_pth, 'logs/full_model_esc50_origin_npy_time_mask_training')
    if data_pth is None:
        data_pth = op.join(proj_pth, 'examples/esc50/mag_origin_npy')
    if loss_type is None:
        loss_type = 'l1'
    if mask_pth is None:
        mask_pth = op.join(proj_pth, 'examples/esc50/mask_shape_1')
    if mask_type is None:
        mask_type = 'square'
    os.chdir(proj_pth)

    for i, filename in enumerate(sorted(Path(data_pth).glob('*.npy'))):
        print(i, filename)

        file_basename = op.basename(filename)
        spec = np.load(filename)
        #print(spec.shape)
        h, w, _ = spec.shape
        spec = spec.reshape((h, w))

        square_l1_loss = []
        # Set 10 as progress unix for mask length. From 10 -> 400.
        for i in tqdm(range(40)):
            image = filename

            # mask_type = 'square'
            mask_length = (i + 1) * 10
            mask = create_mask(mask_length, mask_pth, mask_type)

            # print(op.basename(filename))
            if i < 10:
                output_filename = file_basename.split('.')[0] + f'_{mask_type}_00{i}' + '.' + file_basename.split('.')[1]
            elif i < 100:
                output_filename = file_basename.split('.')[0] + f'_{mask_type}_0{i}' + '.' + file_basename.split('.')[1]
            else:
                output_filename = file_basename.split('.')[0] + f'_{mask_type}_{i}' + '.' + file_basename.split('.')[1]
            output = op.join(output_pth, output_filename)
            # print(output)
            if op.isfile(output) is False:
                cmd = f'python {proj_pth}/test.py --image {image} --mask {mask} --output {output} --checkpoint {checkpoint}'
                # print(cmd)
                os.system(cmd)

            inpaint_spec = np.load(output)
            h, w, _ = inpaint_spec.shape
            inpaint_spec = inpaint_spec.reshape((h, w))

            if spec.shape != inpaint_spec.shape:
                print(spec.shape, inpaint_spec.shape)
            if loss_type == 'l1':
                square_l1_loss.append(l1_loss(spec, inpaint_spec))
            elif loss_type == 'mean_l1':
                square_l1_loss.append(mean_l1_loss(spec, inpaint_spec))
            elif loss_type == 'psnr':
                square_l1_loss.append(psnr(spec, inpaint_spec))
            else:
                raise ValueError("Wrong type of loss type:", loss_type)

        # If the diretory of loss not exist, create it.
        loss_pth = op.join(output_pth, 'loss')
        if op.isdir(loss_pth) is False:
            os.mkdir(loss_pth)
        sound_name = file_basename.split('.')[0]
        loss_filename = op.join(loss_pth, f'{sound_name}_{loss_type}_loss_{mask_type}.txt')
        with open(loss_filename, 'w') as f:
            for index in square_l1_loss:
                f.write(str(index) + '\n')
