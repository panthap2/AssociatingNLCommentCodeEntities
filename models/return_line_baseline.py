import os
import sys

import model_utils

class ReturnLineBaseline:
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid
    
    def classify(self, associations):
        for association in associations:
            code_lines = association.full_code.split('\n')
            return_lines = [l for l in code_lines if 'return' in l]
            for candidate in association.full_code_representation:
                # TODO: Could use the tokenized return line
                if 'return' in code_lines[candidate['line_idx']-1]:
                    prediction = 1
                else:
                    prediction = 0
                    for line in return_lines:
                        if candidate['token'] in line:
                            prediction = 1
                            break
                candidate['prediction'] = prediction

        model_utils.compute_metrics(associations)

    @classmethod
    def classify_candidates(cls, candidate_pairs):
        candidates = []
        for candidate, association in candidate_pairs:
            code_lines = association.full_code.split('\n')
            return_lines = [l for l in code_lines if 'return' in l]
            if 'return' in code_lines[candidate['line_idx']-1]:
                prediction = 1
            else:
                prediction = 0
                for line in return_lines:
                    if candidate['token'] in line:
                        prediction = 1
                        break
            candidate['prediction'] = prediction
            candidates.append(candidate)
        return model_utils.compute_candidate_metrics(candidates)

    @classmethod
    def get_candidate_predictions(cls, candidate_pairs):
        candidates = []
        for candidate, association in candidate_pairs:
            code_lines = association.full_code.split('\n')
            return_lines = [l for l in code_lines if 'return' in l]
            if 'return' in code_lines[candidate['line_idx']-1]:
                prediction = 1
            else:
                prediction = 0
                for line in return_lines:
                    if candidate['token'] in line:
                        prediction = 1
                        break
            candidate['prediction'] = prediction
            candidates.append(candidate)
        return [c['prediction'] for c in candidates]
    
    @classmethod
    def get_candidate_predictions_list(cls, candidate_pairs):
        return [ReturnLineBaseline.get_candidate_predictions(candidate_pairs)]