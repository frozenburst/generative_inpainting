#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from tqdm import tqdm

import numpy as np
import os.path as op
import os
import torchaudio
import torchaudio.transforms as transforms


audio_pth = '/work/r08922a13/download_audioset/audiosetdata'
audio_seg_pth = '/work/r08922a13/datasets/audioset/audio_15seg'
spec_seg_pth = '/work/r08922a13/datasets/audioset/spec_15seg'


class hp:
    sr = 44100  # Sampling rate.
    n_fft = 510
    win_length = n_fft
    hop_length = win_length // 2
    power = 2
    max_db = 100
    ref_db = 20
    least_amp = 1e-5


def toSpec_db_norm(waveform):
    spec = transforms.Spectrogram(
        n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length,
        power=hp.power, normalized=False)(waveform)
    # To decible
    db_spec = 20 * np.log10(np.maximum(hp.least_amp, spec))
    # Normalize
    db_spec_norm = np.clip(db_spec / hp.max_db, -1, 1)
    db_spec_norm = (db_spec_norm + 1.) / 2.
    return db_spec_norm

unexpected_audio_num = 0
for dirname in tqdm(sorted(Path(audio_pth).glob('*'))):
    dirname_base = op.basename(dirname)
    new_audio_seg_pth = op.join(audio_seg_pth, dirname_base)
    new_spec_seg_pth = op.join(spec_seg_pth, dirname_base)
    if op.isdir(new_audio_seg_pth) is False:
        os.mkdir(new_audio_seg_pth)
    if op.isdir(new_spec_seg_pth) is False:
        os.mkdir(new_spec_seg_pth)

    for filename in sorted(Path(dirname).glob('*.wav')):
        wav1_name = op.basename(filename).split('.')[0] + '_1.' + op.basename(filename).split('.')[1]
        wav1_name = op.join(new_audio_seg_pth, wav1_name)
        if op.isfile(wav1_name): continue

        try:
            waveform, sr = torchaudio.load(filename, normalization=True)
        except RuntimeError:
            print("Remove the file not available for loaded:", filename)
            os.remove(filename)
            continue

        if sr != 44100:
            # raise ValueError("Unexpected sample rate:", sr)
            unexpected_audio_num += 1
            print("skip unexpected sample rate:", sr, unexpected_audio_num)
            continue

        t = int(sr * 1.5)
        # 0 ~ 1.5 sec
        if len(waveform[0]) >= sr * 1.5:
            start = 0
            end = int(sr * 1.5)  # 66150
            waveform_1 = waveform[:, start:end]

            #wav1_name = op.basename(filename).split('.')[0] + '_1.' + op.basename(filename).split('.')[1]
            #wav1_name = op.join(new_audio_seg_pth, wav1_name)
            #if op.isfile(wav1_name): continue
            torchaudio.save(wav1_name, src=waveform_1, sample_rate=sr)

            # Transform to spec and save as npy
            spec = toSpec_db_norm(waveform_1)
            # [1, 256, 260] -> [256, 256, 1]
            spec = spec[0, :, 2:-2, np.newaxis]

            spec_name = op.basename(filename).split('.')[0] + '_1'
            spec_name = op.join(new_spec_seg_pth, spec_name)
            np.save(spec_name, spec)

        # 1.75 ~ 3.25 sec
        if len(waveform[0]) >= sr * 3.25:
            start = int(sr * 1.75)
            end = int(sr * 3.25)
            waveform_2 = waveform[:, start:end]

            wav2_name = op.basename(filename).split('.')[0] + '_2.' + op.basename(filename).split('.')[1]
            wav2_name = op.join(new_audio_seg_pth, wav2_name)
            torchaudio.save(wav2_name, src=waveform_2, sample_rate=sr)

            # Transform to spec and save as npy
            spec = toSpec_db_norm(waveform_2)
            # [1, 256, 260] -> [256, 256, 1]
            spec = spec[0, :, 2:-2, np.newaxis]

            spec_name = op.basename(filename).split('.')[0] + '_2'
            spec_name = op.join(new_spec_seg_pth, spec_name)
            np.save(spec_name, spec)
        # 3.5 ~ 5 sec
        if len(waveform[0]) >= sr * 5:
            start = int(sr * 3.5)
            end = int(sr * 5)
            waveform_3 = waveform[:, start:end]

            wav3_name = op.basename(filename).split('.')[0] + '_3.' + op.basename(filename).split('.')[1]
            wav3_name = op.join(new_audio_seg_pth, wav3_name)
            torchaudio.save(wav3_name, src=waveform_3, sample_rate=sr)

            # Transform to spec and save as npy
            spec = toSpec_db_norm(waveform_3)
            # [1, 256, 260] -> [256, 256, 1]
            spec = spec[0, :, 2:-2, np.newaxis]

            spec_name = op.basename(filename).split('.')[0] + '_3'
            spec_name = op.join(new_spec_seg_pth, spec_name)
            np.save(spec_name, spec)
        # 5.25 ~ 6.75 sec
        if len(waveform[0]) >= sr * 6.75:
            start = int(sr * 5.25)
            end = int(sr * 6.75)
            waveform_4 = waveform[:, start:end]

            wav4_name = op.basename(filename).split('.')[0] + '_4.' + op.basename(filename).split('.')[1]
            wav4_name = op.join(new_audio_seg_pth, wav4_name)
            torchaudio.save(wav4_name, src=waveform_4, sample_rate=sr)

            # Transform to spec and save as npy
            spec = toSpec_db_norm(waveform_4)
            # [1, 256, 260] -> [256, 256, 1]
            spec = spec[0, :, 2:-2, np.newaxis]

            spec_name = op.basename(filename).split('.')[0] + '_4'
            spec_name = op.join(new_spec_seg_pth, spec_name)
            np.save(spec_name, spec)
        # 7 ~ 8.5 sec
        if len(waveform[0]) >= sr * 8.5:
            start = int(sr * 7)
            end = int(sr * 8.5)
            waveform_5 = waveform[:, start:end]

            wav5_name = op.basename(filename).split('.')[0] + '_5.' + op.basename(filename).split('.')[1]
            wav5_name = op.join(new_audio_seg_pth, wav5_name)
            torchaudio.save(wav5_name, src=waveform_5, sample_rate=sr)

            # Transform to spec and save as npy
            spec = toSpec_db_norm(waveform_5)
            # [1, 256, 260] -> [256, 256, 1]
            spec = spec[0, :, 2:-2, np.newaxis]

            spec_name = op.basename(filename).split('.')[0] + '_5'
            spec_name = op.join(new_spec_seg_pth, spec_name)
            np.save(spec_name, spec)
