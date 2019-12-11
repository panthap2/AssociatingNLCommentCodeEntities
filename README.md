# AssociatingNLCommentCodeEntities
**Dataset and code corresponding to Associating Natural Language Comment and Source Code Entities (AAAI 2020)**

Data is available in `model_data/`. It can be parsed using the `load_data` method in `models/model_utils.py`.

Download embeddings.tar.gz continaining pretrained embeddings from [here](https://drive.google.com/open?id=1pPaNIsVx4zftY0-AFA48A6Uj-evYsJYy).
Unzip the file in the root directory:

```
tar zxvf embeddings.tar.gz
```

A directory with the name `embeddings` should appear, in the root directory, with 3 json files.

Commands to train models are structured as below:

```
python run_model.py -model [MODEL_TYPE] -dropout [DROPOUT_KEEP_PROBABILITY] -lr [LEARNING_RATE] -decay [DECAY_RATE] -decay_steps [NUM_DECAY_STEPS] -num_layers [NUM_LAYERS] -layer_units [LAYER_DIMENSIONS] -model_name [MOEL_NAME] -delete_size [NUM_EXAMPLES_FROM_DELETIONS]
```

MODEL_TYPES = [feedforward, more_data_feedforward, crf, more_data_crf,
               subtoken_matching_baseline, return_line_baseline,
               random_baseline', 'majority_class_random_baseline]


Matching model types with those in the paper:

**Learned models:**
* feedforward = binary classifier
* more_data_feedforward = binary classifier w/ data from deletions set
* crf = CRF for joint classification
* mode_data_crf = CRF for joint classication w/ data from deletions set

**Baselines:**
* subtoken_matching_baseline = subtoken matching
* return_line_baseline = presence in return line
* random_baseline = random
* majority_class_random_baseline = weighted random

Sample commands can be found in `models/run.sh`.

Please email spantha@cs.utexas.edu for any questions.