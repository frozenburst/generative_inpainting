#!/usr/bin/env python
# coding: utf-8
"""
Transfer the output spectrogram to audio.

usage: spec_diff.py  [options]

options:
    --spec_pth=<spectrogram file directory>
    --spec_origin_pth=<original spectrogram file directory>
"""
from docopt import docopt
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os
import imageio


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    spec_pth = args['--spec_pth']
    spec_origin_pth = args['--spec_origin_pth']

    if spec_pth is None:
        raise ValueError("Please set the path for spectrogram directory.")

    diff_pth = op.join(spec_pth, 'diff_output')
    if op.isdir(diff_pth) is False:
        os.mkdir(diff_pth)
    shape = ['square', 'time']
    print("Start to process the diff of spectrogram...")
    for filename in sorted(Path(spec_origin_pth).glob('*.npy')):
        origin = np.load(filename)
        file = op.basename(filename).split('.')[0]
        for anyshape in shape:
            file_shape = file + '_' + anyshape
            for output in tqdm(sorted(Path(spec_pth).glob(f'{file_shape}*.npy'))):
                inpainted = np.load(output)
                diff = abs(inpainted - origin)
                diff_output = op.join(diff_pth, op.basename(output))
                np.save(diff_output, diff)

    print("Load npy spec of diff output and save as png file...")
    png_pth = op.join(diff_pth, 'png')
    if op.isdir(png_pth) is False:
        os.mkdir(png_pth)
    for filename in tqdm(sorted(Path(diff_pth).glob('*.npy'))):
        spec = np.load(filename)
        h, w, _ = spec.shape
        spec = spec.reshape((h, w))

        png_filename = op.basename(filename).split('.')[0] + '.png'
        png_filename = op.join(png_pth, png_filename)

        # First save the spec to png with force try.
        plt.imsave(png_filename, spec, cmap='gray')

    print("Make the png files to gif...")
    for filename in sorted(Path(spec_origin_pth).glob('*.npy')):
        file_basename = op.basename(filename).split('.')[0]
        for anyshape in shape:
            file_shape = file_basename + '_' + anyshape
            with imageio.get_writer(op.join(png_pth, f'{file_shape}.gif'), mode='I', duration=0.2) as writer:
                for png in tqdm(sorted(Path(png_pth).glob(f'{file_shape}*.png'))):
                    image = imageio.imread(png)
                    writer.append_data(image)
