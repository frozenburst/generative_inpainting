# 将原数据集分为training ,validation  by gavin
from pathlib import Path
from tqdm import tqdm

import numpy as np
import glob
import os
import os.path as op
import random

import argparse

# 划分验证集训练集
_NUM_TEST = 0.1
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/work/r08922a13/Waveform-auto-encoder/datasets/ESC-50-master/audio', type=str,
                    help='The reference path')
parser.add_argument('--data_path', default='/work/r08922a13/Waveform-auto-encoder/datasets/ESC-50-master/spectrogram_large', type=str,
                    help='The data path')
parser.add_argument('--train_filename', default='./data/esc50/spec_large/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data/esc50/spec_large/validation_static_view.flist', type=str,
                    help='The validation filename.')

if __name__ == "__main__":

    args = parser.parse_args()
    folder_dir = args.folder_path
    data_dir = args.data_path

    if op.isdir(op.dirname(args.train_filename)) is False:
        os.mkdir(op.dirname(args.train_filename))
    if op.isdir(op.dirname(args.validation_filename)) is False:
        os.mkdir(op.dirname(args.validation_filename))

    # get all file names
    #photo_filenames = _get_filenames(data_dir)
    audio_filenames = glob.glob(f'{folder_dir}/*.wav')
    num_files = len(audio_filenames)
    print(f'size of esc50 is {num_files}')

    # 切分数据为测试训练集
    random.seed(0)
    random.shuffle(audio_filenames)
    num_test = int(_NUM_TEST * num_files)
    training_file_names = audio_filenames[num_test:]
    validation_file_names = audio_filenames[:num_test]

    t_file_all = []
    v_file_all = []
    for name in tqdm(training_file_names):
        name = op.join(data_dir, op.basename(name).split('.')[0])
        t_file_all.append(glob.glob(f'{name}*.npy'))
    training_file_names = t_file_all
    training_file_names = list(np.array(training_file_names).flat)
    random.shuffle(training_file_names)

    for name in tqdm(validation_file_names):
        name = op.join(data_dir, op.basename(name).split('.')[0])
        v_file_all.append(glob.glob(f'{name}*.npy'))
    validation_file_names = v_file_all
    validation_file_names = list(np.array(validation_file_names).flat)
    random.shuffle(validation_file_names)

    print("training file size:",len(training_file_names))
    print("validation file size:", len(validation_file_names))

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename)
