import argparse
import os
import sys
import json
import math

from crf_model import CRFModel
from data_utils import raw_tokenize_code
from feedforward_nn import FeedForwardNN
from random_baseline import RandomBaseline
from majority_class_random_baseline import MajorityClassRandomBaseline
from return_line_baseline import ReturnLineBaseline
from subtoken_matching_baseline import SubtokenMatchingBaseline
from model_utils import load_data, load_embeddings, set_features, load_data_from_object

MODEL_DATA_PATH = '../model_data/'
OUTPUT_FILE = '../output.txt'
FULL_DATASET = '../model_data/full_dataset.json'
FULL_ANNOTATIONS = '../model_data/full_annotations.json'

MODEL_TYPES = ['feedforward', 'more_data_feedforward', 'crf', 'more_data_crf',
               'subtoken_matching_baseline', 'return_line_baseline',
               'random_baseline', 'majority_class_random_baseline']

# Matching model types with those in the paper:

# Learned models:
# feedforward = binary classifier
# more_data_feedforward = binary classifier w/ data from deletions set
# crf = CRF for joint classification
# mode_data_crf = CRF for joint classication w/ data from deletions set

# Baselines:
# subtoken_matching_baseline = subtoken matching
# return_line_baseline = presence in return line
# random_baseline = random
# majority_class_random_baseline = weighted random

parser = argparse.ArgumentParser()
parser.add_argument('-model', help=str(MODEL_TYPES))
parser.add_argument('-v', help='verbose', action='store_true')
parser.add_argument('-dropout', help='dropout keep probability')
parser.add_argument('-lr', help='learning rate')
parser.add_argument('-decay', help='decay percentage')
parser.add_argument('-decay_steps', help='decay steps')
parser.add_argument('-num_layers', help='number of layers')
parser.add_argument('-layer_units', help='comma separated list of layer units')
parser.add_argument('-model_name', help='model name')
parser.add_argument('-delete_size', help='amount of deleted data to include')
parser.add_argument('-oracle', help='use oracle annotations', action='store_true')

def main(args=None):
    args = parser.parse_args(args)

    if args.model not in MODEL_TYPES:
        raise ValueError('Invalid model type')

    if not args.model_name:
        raise ValueError('Model name must be provided')

    embeddings = load_embeddings('../embeddings/')

    with open('../ontology/java_utils_main.json') as json_file:
        java_utils = json.load(json_file)

    with open('../ontology/java_utils_methods.json') as json_file:
        java_utils_methods = json.load(json_file)

    with open(FULL_DATASET) as json_file:
        data = json.load(json_file)

    train_associations = load_data_from_object(data['train'])
    test_associations = load_data_from_object(data['test'])
    valid_associations = load_data_from_object(data['valid'])
    deleted_associations = load_data_from_object(data['deleted'])

    if args.model == 'more_data_feedforward' or args.model == 'more_data_crf':
        if not args.delete_size or int(args.delete_size) > len(deleted_associations):
            raise ValueError('Delete size must be provided. Max size is ' + str(len(deleted_associations)))

        train_associations = train_associations + deleted_associations[0:int(args.delete_size)]

    with open(FULL_ANNOTATIONS) as json_file:
        annotations = json.load(json_file)

    if not args.oracle:
        # Use annotated data for evaluation.
        for association in test_associations:
            annotation_id = association.annotation_id
            for candidate in association.full_code_representation:
                token = candidate['token']
                line_number = candidate['line_idx']
                position = candidate['pos_idx']
                key = token + '-' + str(line_number) + '-' + str(position)
                if key in annotations[str(annotation_id)]:
                    if annotations[str(annotation_id)][key] == 'True':
                        candidate['is_associated'] = True
                    else:
                        candidate['is_associated'] = False

    if 'crf' in args.model:
        process_crf_dataset(train_associations, embeddings, java_utils, java_utils_methods)
        process_crf_dataset(test_associations, embeddings, java_utils, java_utils_methods)
        process_crf_dataset(valid_associations, embeddings, java_utils, java_utils_methods)

    elif 'baseline' not in args.model:
        process_dataset(train_associations, embeddings, java_utils, java_utils_methods)
        process_dataset(test_associations, embeddings, java_utils, java_utils_methods)
        process_dataset(valid_associations, embeddings, java_utils, java_utils_methods)

    print "Train: " + str(len(train_associations))
    print "Test: " + str(len(test_associations))
    print "Valid: " + str(len(valid_associations))

    if args.model == 'crf' or args.model == 'more_data_crf':
        model = CRFModel(train_associations, test_associations, valid_associations, args)
    elif args.model == 'feedforward' or args.model == 'more_data_feedforward':
        model = FeedForwardNN(train_associations, test_associations, valid_associations, args)
    elif args.model == 'subtoken_matching_baseline':
        model = SubtokenMatchingBaseline(train_associations, test_associations, valid_associations)
    elif args.model == 'return_line_baseline':
        model = ReturnLineBaseline(train_associations, test_associations, valid_associations)
    elif args.model == 'random_baseline':
        model = RandomBaseline(train_associations, test_associations, valid_associations)
    elif args.model == 'majority_class_random_baseline':
        model = MajorityClassRandomBaseline(train_associations, test_associations, valid_associations)
    else:
        raise ValueError('Unable to identify model type')

    print("Evaluation:")
    print("------------------")
    print("Train:")
    model.classify(train_associations)
    print("------------------")
    print("Valid:")
    model.classify(valid_associations)
    print("------------------")
    print("Test:")
    model.classify(test_associations)
    print("------------------")
    sys.stdout.flush()

    if args.v:
        with open(OUTPUT_FILE, 'w+') as f:
            for association in test_associations:
                f.write("NP: " + association.np_chunks[0] + '\n')
                f.write("Comment line: " + association.comment_line + '\n\n')
                f.write(association.full_code.encode('utf-8') + '\n\n')

                predicted = [str(c['token']) for c in association.full_code_representation if c['prediction'] == 1]
                gold = [str(c['token']) for c in association.full_code_representation if c['is_associated']]

                f.write("Predicted: " + str(predicted) + '\n\n')
                f.write("Gold: " + str(gold) + '\n\n')
                f.write("Candidates: " + str([str(c) for c in association.candidate_code_tokens]) + '\n')
                f.write('***************************\n\n')

def process_crf_dataset(dataset, embeddings, java_utils, java_utils_methods):
    for association in dataset:
        candidate_lookup = dict()
        for c in association.full_code_representation:
            pos = c['pos_idx']
            line = c['line_idx']
            key = str(pos) + '-' + str(line)
            candidate_lookup[key] = c['is_associated']

        raw_tokens, all_values = raw_tokenize_code(association.full_code)
        line_dict = dict()
        
        modified_candidates = []
        for token in raw_tokens:
            val = token.value
            l = token.position[0]
            p = token.position[1]

            if l not in line_dict:
                line_dict[l] = []
            line_dict[l].append(val)

            modified_candidate = dict()
            modified_candidate['token'] = val
            modified_candidate['line_idx'] = l
            modified_candidate['pos_idx'] = p

            key = str(p) + '-' + str(l)
            if key in candidate_lookup:
                modified_candidate['is_associated'] = candidate_lookup[key]
                modified_candidate['pseudo_class'] = str(candidate_lookup[key])
            else:
                modified_candidate['is_associated'] = False
                modified_candidate['pseudo_class'] = 'JAVA'

            modified_candidates.append(modified_candidate)

        for m in modified_candidates:
            m['tokenized_line'] = line_dict[m['line_idx']]

        association.full_code_representation = modified_candidates
        set_features(association, embeddings, java_utils, java_utils_methods)

def process_dataset(dataset, embeddings, java_utils, java_utils_methods):
    for association in dataset:
        set_features(association, embeddings, java_utils, java_utils_methods)

if __name__ == '__main__':
    main()