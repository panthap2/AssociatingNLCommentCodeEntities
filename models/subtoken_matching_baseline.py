import os
import sys

import data_utils as utils
import model_utils

class SubtokenMatchingBaseline:
    def __init__(self, train, test, valid):
        self.train = train
        self.test = test
        self.valid = valid
    
    def classify(self, associations):
        for association in associations:
            np_subtokens = utils.subtokenize_comment_line(association.np_chunks[0])
            for candidate in association.full_code_representation:
                 # Although it isn't a comment, can use the same function
                subtokens = utils.subtokenize_comment_line(candidate['token'])
                prediction = 0
                for sub in subtokens:
                    if sub in np_subtokens:
                        prediction = 1
                candidate['prediction'] = prediction

        model_utils.compute_metrics(associations)

    @classmethod
    def classify_candidates(cls, candidate_pairs):
        candidates = []
        for candidate, association in candidate_pairs:
            np_subtokens = utils.subtokenize_comment_line(association.np_chunks[0])
            subtokens = utils.subtokenize_comment_line(candidate['token'])
            prediction = 0
            for sub in subtokens:
                if sub in np_subtokens:
                    prediction = 1
            candidate['prediction'] = prediction
            candidates.append(candidate)
        return model_utils.compute_candidate_metrics(candidates)
    
    @classmethod
    def get_candidate_predictions(cls, candidate_pairs):
        candidates = []
        for candidate, association in candidate_pairs:
            np_subtokens = utils.subtokenize_comment_line(association.np_chunks[0])
            subtokens = utils.subtokenize_comment_line(candidate['token'])
            prediction = 0
            for sub in subtokens:
                if sub in np_subtokens:
                    prediction = 1
            candidate['prediction'] = prediction
            candidates.append(candidate)
        return [c['prediction'] for c in candidates]
    
    @classmethod
    def get_candidate_predictions_list(cls, candidate_pairs):
        return [SubtokenMatchingBaseline.get_candidate_predictions(candidate_pairs)]


