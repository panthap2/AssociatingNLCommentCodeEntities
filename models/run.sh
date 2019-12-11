#!/bin/bash

# Binary classification
python run_model.py -model feedforward -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name feedforward_annotated -delete_size 0
python run_model.py -model feedforward -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name feedforward_oracle -delete_size 0 -oracle
python run_model.py -model more_data_feedforward -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name more_data_feedforward_annotated -delete_size 867
python run_model.py -model more_data_feedforward -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name more_data_feedforward_oracle -delete_size 867 -oracle

# CRF
python run_model.py -model crf -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name crf_annotated -delete_size 0
python run_model.py -model crf -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name crf_oracle -delete_size 0 -oracle
python run_model.py -model more_data_crf -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name more_data_crf_annotated -delete_size 867
python run_model.py -model more_data_crf -dropout 0.8 -lr 0.001 -decay 0.99 -decay_steps 500 -num_layers 4 -layer_units 512,384,256,128 -model_name more_data_crf_oracle -delete_size 867 -oracle

# Baselines
python run_model.py -model subtoken_matching_baseline -model_name subtoken_matching_baseline_annotated
python run_model.py -model subtoken_matching_baseline -model_name subtoken_matching_baseline_oracle -oracle

python run_model.py -model return_line_baseline -model_name return_line_annotated
python run_model.py -model return_line_baseline -model_name return_line_oracle -oracle

python run_model.py -model majority_class_random_baseline -model_name random_baseline_annotated
python run_model.py -model majority_class_random_baseline -model_name random_baseline_oracle -oracle

python run_model.py -model random_baseline -model_name random_baseline_annotated
python run_model.py -model random_baseline -model_name random_baseline_oracle -oracle