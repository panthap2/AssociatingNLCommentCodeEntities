import os
import numpy as np
import random
import sys

import model_utils

ITERATIONS = 10

class MajorityClassRandomBaseline:
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid

        total = 0
        pos = 0
        neg = 0

        for association in self.train:
            for candidate in association.full_code_representation:
                if candidate['is_associated']:
                    pos += 1
                else:
                    neg += 1
                total += 1

        self.positive_rate = float(pos)/total
        self.negative_rate = float(neg)/total

        print "Positive rate: " + str(self.positive_rate)
        print "Negative rate: " + str(self.negative_rate)

    def average(self, vals):
        return float(sum(vals))/len(vals)
    
    def classify(self, associations):
        p_vals = []
        r_vals = []
        f_vals = []
        a_vals = []
        mp_vals = []
        mr_vals = []
        mf_vals = []
        ma_vals = []

        i = 0
        while i < ITERATIONS:
            for association in associations:
                for candidate in association.full_code_representation:
                    candidate['prediction'] = np.random.choice([0, 1], p=[self.negative_rate, self.positive_rate])
            precision, recall, f1, accuracy, macro_precision, macro_recall, macro_f1, macro_accuracy = model_utils.compute_metrics(associations)
            p_vals.append(precision)
            r_vals.append(recall)
            f_vals.append(f1)
            a_vals.append(accuracy)

            mp_vals.append(macro_precision)
            mr_vals.append(macro_recall)
            mf_vals.append(macro_f1)
            ma_vals.append(macro_accuracy)

            i += 1

        print "Final: "
        print "Average Precision: " + str(self.average(p_vals))
        print "Average Recall: " + str(self.average(r_vals))
        print "Average F1: " + str(self.average(f_vals))
        print "Average accuracy: " + str(self.average(a_vals))
        print "Average Macro Precision: " + str(self.average(mp_vals))
        print "Average Macro Recall: " + str(self.average(mr_vals))
        print "Average Macro F1: " + str(self.average(mf_vals))
        print "Average Macro Accuracy: " + str(self.average(ma_vals))

    def classify_candidates(self, candidate_pairs):
        f1_vals = []
        i = 0
        while i < ITERATIONS:
            candidates = []
            for candidate, _ in candidate_pairs:
                candidate['prediction'] = np.random.choice([0, 1], p=[self.negative_rate, self.positive_rate])
                candidates.append(candidate)
            f1_vals.append(model_utils.compute_candidate_metrics(candidates))
            i += 1
        return float(sum(f1_vals))/len(f1_vals)
    
    def get_candidate_predictions(self, candidate_pairs):
        candidates = []
        for candidate, _ in candidate_pairs:
            candidate['prediction'] = np.random.choice([0, 1], p=[self.negative_rate, self.positive_rate])
            candidates.append(candidate)
        return [c['prediction'] for c in candidates]
    
    def get_candidate_predictions_list(self, candidate_pairs):
        predictions = []
        i = 0
        while i < ITERATIONS:
            predictions.append(self.get_candidate_predictions(candidate_pairs))
            i += 1
        return predictions