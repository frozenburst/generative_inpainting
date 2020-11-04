#!/bin/bash
# Record the command of lose_measure.py, might develop to ez shell script.
python loss_measure.py --checkpoint /work/r08922a13/generative_inpainting/logs/full_model_esc50_15seg --output_pth /work/r08922a13/generative_inpainting/examples/esc50/mag_seg15_256/loss_test_output --mask_type time --loss_type l1 --data_pth /work/r08922a13/generative_inpainting/examples/esc50/mag_seg15_256 --mask_pth /work/r08922a13/generative_inpainting/examples/esc50/mask_256
