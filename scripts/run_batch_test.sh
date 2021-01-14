#!/bin/bash

#####
# After the training of genrative model.

# Use the "batch_test_genList.py" to generate the file list for batch test.
# Use the "batch_test_loss_measure.py" to measure the loss of batch test output.
# Use the "2audio.py" transfer all the outout spectrogram to waveform.

# The last one cannot work in twcc node, which is without -X forward for plt.
# Use the "audio2png_loss.py" to visualize the transformation of mask field from min to max.
#####

echo "Total argument: $#"
echo "Script name: $0"

# directory settings.
proj_pth="/work/r08922a13/generative_inpainting"
checkpoint="/work/r08922a13/generative_inpainting/logs/past/full_model_esc50_15seg_time_mask_only"
output_pth="/work/r08922a13/generative_inpainting/examples/esc50/mag_seg15_256_all/train_m52"
data_pth="/work/r08922a13/Waveform-auto-encoder/datasets/ESC-50-master/spectrogram_15seg"
mask_pth="/work/r08922a13/generative_inpainting/examples/esc50/mask_256/mask_square_052.npy"
file_list_pth="/work/r08922a13/generative_inpainting/data/esc50/spec_15seg/train_shuffled.flist"

# other settings
#mask_type="time"  # [time, square]
loss_type="mean_l1"    # [l1, mean_l1, psnr]
imageH=256
imageW=256


# Processing parts.
echo "Generate the file list of batch test."
python batch_test_genList.py \
    --file_list_pth "${file_list_pth}" \
    --output_pth "${output_pth}" \
    --mask_pth "${mask_pth}"
echo "Generation Finish."

echo "Start batch test..."
cd ${proj_pth}
python batch_test.py \
    --flist "${output_pth}/batch_test_list.txt" \
    --image_height ${imageH} \
    --image_width ${imageW} \
    --checkpoint_dir "${checkpoint}"
cd "${proj_pth}/scripts"

echo "Strat to test the model... with loss measure..."
python batch_test_loss_measure.py \
    --output_pth "${output_pth}" \
    --loss_type "${loss_type}" \
    --ref_spec_pth "${data_pth}"
echo "Test of time mask finish."

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
