#!/bin/bash

#####
# After the training of genrative model.

# Use the "2audio.py" transfer all the outout spectrogram to waveform.

# The last one cannot work in twcc node, which is without -X forward for plt.
# Use the "audio2png_loss.py" to visualize the transformation of mask field from min to max.
#####

echo "Total argument: $#"
echo "Script name: $0"

# directory settings.
proj_pth="/work/r08922a13/generative_inpainting"
checkpoint="/work/r08922a13/generative_inpainting/logs/cur/full_model_esc50_15seg_time_mask_only_with_brush"
output_pth="/work/r08922a13/generative_inpainting/examples/esc50/mag_seg15_256_all/train_m52_with_brush"
data_pth="/work/r08922a13/Waveform-auto-encoder/datasets/ESC-50-master/spectrogram_15seg"
mask_pth="/work/r08922a13/generative_inpainting/examples/esc50/mask_256/mask_time_052.npy"
# file_list_pth="/work/r08922a13/generative_inpainting/data/esc50/spec_15seg/train_shuffled.flist"
# train_shuffled.flist(train), validation_static_view.flist(test)

# other settings
#mask_type="time"  # [time, square]
#loss_type="mean_l1"    # [l1, mean_l1, psnr]
imageH=256
imageW=256


# Processing parts.
echo "Generate without mask output"



echo "Start testing..."
cd ${proj_pth}
python test.py \
    --image "${data_pth}" \
    --mask "${mask_pth}" \
    --output "${output_pth}" \
    --checkpoint "${checkpoint}"
cd "${proj_pth}/scripts"

echo "Save npy spec as png file..."
python npy2png.py \
    --spec_pth "${output_pth}"
echo "Saving png file complete."

#echo "Count the diff of spectrogram..."
#python spec_diff.py \
#    --spec_pth="${output_pth}" \
#    --spec_origin_pth="${data_pth}"
#echo "Diff process finish."

echo "Transfer the spectrogram to audio..."
python 2audio.py \
    --spec_pth="${output_pth}"
echo "Finish!"

#echo "Anamated the diff of audio transformation"
#python audio2png_loss.py \
#    --data_pth="${data_pth}"
#    --audio_pth="${output_pth}/audio"
echo "All Finish!"
