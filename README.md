# Associating Natural Language Comment and Source Code Entities
**Code and datasets for our AAAI-2020 paper "Associating Natural Language Comment and Source Code Entities", which can be found [here](https://arxiv.org/abs/1912.06728).**

If you find this work useful, please consider citing our paper:

```
@inproceedings{panthaplackel2020associating,
  author={Sheena Panthaplackel and Milos Gligoric and Raymond J. Mooney and Junyi Jessy Li},
  title={Associating Natural Language Comment and Source Code Entities},
  booktitle={AAAI},
  pages = {8592-8599},
  year={2020},
}
```

Data is available in `model_data/`. It can be parsed using the `load_data` method in `models/model_utils.py`.

Download embeddings.tar.gz continaining pretrained embeddings from [here](https://drive.google.com/open?id=1pPaNIsVx4zftY0-AFA48A6Uj-evYsJYy).
Unzip the file in the root directory:

```
tar zxvf embeddings.tar.gz
```

A directory with the name `embeddings` should appear, in the root directory, with 3 json files.

You will need to create a `checkpoints` directory under the root directory.

Run models from within the `models` directory. Commands to train models are structured as below:

```
python run_model.py -model [MODEL_TYPE] -dropout [DROPOUT_KEEP_PROBABILITY] -lr [LEARNING_RATE] -decay [DECAY_RATE] -decay_steps [NUM_DECAY_STEPS] -num_layers [NUM_LAYERS] -layer_units [LAYER_DIMENSIONS] -model_name [MOEL_NAME] -delete_size [NUM_EXAMPLES_FROM_DELETIONS]
```

Insert one of the following model types in place of MODEL_TYPE:
* feedforward
* more_data_feedforward
* crf
* more_data_crf
* subtoken_matching_baseline
* return_line_baseline
* random_baseline
* majority_class_random_baseline

Matching model types with those in the paper:

**Learned models:**
* feedforward = binary classifier
* more_data_feedforward = binary classifier w/ data from deletions set
* crf = CRF for joint classification
* more_data_crf = CRF for joint classication w/ data from deletions set

**Baselines:**
* subtoken_matching_baseline = subtoken matching
* return_line_baseline = presence in return line
* random_baseline = random
* majority_class_random_baseline = weighted random

Sample commands can be found in `models/run.sh`.

Please email spantha@cs.utexas.edu for any questions.
