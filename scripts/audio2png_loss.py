#!/usr/bin/env python
# coding: utf-8
"""
Transfer the output spectrogram to audio.

usage: audio2png_loss.py  [options]

options:
    --data_pth=<original audio file directory>
    --audio_pth=<output audio file directory>
"""
from docopt import docopt
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import os
import torch
import torchaudio
import imageio


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    data_pth = args['--data_pth']
    audio_pth = args['--audio_pth']

    if data_pth is None:
        raise ValueError("Please set the path for original audio directory.")
    if audio_pth is None:
        raise ValueError("Please set the path for output audio directory.")

    output_pth = op.join(audio_pth, 'loss')
    if op.isdir(output_pth) is False:
        os.mkdir(output_pth)
    png_pth = op.join(audio_pth, 'png')
    if op.isdir(png_pth) is False:
        os.mkdir(png_pth)
    shape = ['square', 'time']
    for filename in sorted(Path(data_pth).glob('*_res.wav')):
        origin_waveform, sr = torchaudio.load(filename, normalization=True)
        file_basename = op.basename(filename).split('.')[0].split('_origin')[0]
        for anyshape in shape:
            file_shape = file_basename + '_' + anyshape
            error_list = []
            with imageio.get_writer(op.join(png_pth, f'{file_shape}.gif'), mode='I', duration=0.2) as writer:
                for output in tqdm(sorted(Path(audio_pth).glob(f'{file_shape}*.wav'))):
                    inpainted_waveform, sr = torchaudio.load(output, normalization=True)
                    diff = torch.mean(abs(inpainted_waveform - origin_waveform))
                    err_value = diff.item()
                    error_list.append(err_value)

                    wave_pack = np.concatenate((origin_waveform, inpainted_waveform), axis=0)

                    plt.figure()
                    plt.ylim(-0.5, 0.5)
                    plt.plot(wave_pack[0], 'b', wave_pack[1], 'r')
                    png_name = op.basename(output).split('.')[0] + '.png'
                    png_name = op.join(png_pth, png_name)
                    plt.savefig(png_name)

                    image = imageio.imread(png_name)
                    writer.append_data(image)
            error_filename = op.join(output_pth, f'{file_shape}_mean-l1.txt')
            with open(error_filename, 'w') as f:
                for index in error_list:
                    f.write(str(index)+'\n')
