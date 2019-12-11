from model_utils import *
import os
import numpy as np
import sys
import tensorflow as tf

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import utils
import random
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid
        

        self.initialize()

    def initialize(self):
        self.train_inp, self.train_out = self.process_dataset(self.train)
        self.valid_inp, self.valid_out = self.process_dataset(self.valid)

        self.model = LogisticRegression(solver='liblinear').fit(self.train_inp, self.train_out)

    def process_dataset(self, dataset):
        temp = []
        for association in dataset:
            for candidate in association.full_code_representation:
                c = dict()
                c['feature_vector'] = candidate['features']
                if candidate['is_associated']:
                    c['class'] = 1
                else:
                    c['class'] = 0
                temp.append(c)

        random.shuffle(temp)
        inp = []
        out = []

        for t in temp:
            inp.append(t['feature_vector'])
            out.append(t['class'])

        return np.asarray(inp), np.asarray(out)

    def classify(self, associations):
        for association in associations:
            candidates = association.full_code_representation
            inputs = [c['features'] for c in candidates]
            pred_this_instance = self.model.predict(inputs)
            for i in range(len(pred_this_instance)):
                candidates[i]['prediction'] = pred_this_instance[i]
        compute_metrics(associations)