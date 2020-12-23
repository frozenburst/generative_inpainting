#!/bin/bash

#####
# After the training of genrative model.

# Use the "loss_measure.py" to test with own test data.
# Use the "spec_diff.py" to get the difference with original test data.
# Use the "2audio.py" transfer all the outout spectrogram to waveform.

# The last one cannot work in twcc node, which is without -X forward for plt.
# Use the "audio2png_loss.py" to visualize the transformation of mask field from min to max.
#####

echo "Total argument: $#"
echo "Script name: $0"

# directory settings.
checkpoint="/work/r08922a13/generative_inpainting/logs/cur/full_model_esc50_15seg_time_mask_only_without_brush"
output_pth="/work/r08922a13/generative_inpainting/examples/esc50/mag_seg15_256/loss_test_without_brush"
data_pth="/work/r08922a13/generative_inpainting/examples/esc50/mag_seg15_256"
mask_pth="/work/r08922a13/generative_inpainting/examples/esc50/mask_256"


# other settings
#mask_type="time"  # [time, square]
#loss_type="l1"    # [l1, mean_l1, psnr]


# Record the command of lose_measure.py, might develop to ez shell script.
echo "Strat to test the model... with mask type: time"
python loss_measure.py \
    --checkpoint "${checkpoint}" \
    --output_pth "${output_pth}" \
    --mask_type "time" \
    --loss_type "l1" \
    --data_pth "${data_pth}" \
    --mask_pth "${mask_pth}"
echo "Test of time mask finish."

echo "Strat to test the model... with mask type: square"
python loss_measure.py \
    --checkpoint "${checkpoint}" \
    --output_pth "${output_pth}" \
    --mask_type "square" \
    --loss_type "l1" \
    --data_pth "${data_pth}" \
    --mask_pth "${mask_pth}"
echo "Test finish."

echo "Count the diff of spectrogram..."
python spec_diff.py \
    --spec_pth="${output_pth}" \
    --spec_origin_pth="${data_pth}"
echo "Diff process finish."

echo "Transfer the spectrogram to audio..."
python 2audio.py \
    --spec_pth="${output_pth}"
echo "Finish!"

echo "Anamated the diff of audio transformation"
#python audio2png_loss.py \
#    --data_pth="${data_pth}"
#    --audio_pth="${output_pth}/audio"
echo "All Finish!"
