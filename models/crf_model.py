from model_utils import *
import os
import numpy as np
import sys
import tensorflow as tf

import random
from sklearn.metrics import precision_recall_fscore_support

PAD_IDX = 2

class CRFModel:
    def __init__(self, train, test, valid, parameters, mode='train'):
        self.train = train
        self.test = test
        self.valid = valid
        self.model_name = parameters.model_name
        self.mode = mode
        
        self.num_classes = 3
        # self.first_num_hidden_units = 128
        # self.second_num_hidden_units = 256

        if parameters.decay_steps:
            self.decay_steps = int(parameters.decay_steps)
        else:
            self.decay_steps = 500 #500 #1000

        if parameters.decay:
            self.learning_rate_decay_factor = float(parameters.decay)
        else:
            self.learning_rate_decay_factor = 0.99

        if parameters.lr:
            self.initial_learning_rate = float(parameters.lr)
        else:
            self.initial_learning_rate = 0.001

        if parameters.dropout:
            self.dropout = float(parameters.dropout)
        else:
            self.dropout = 0.5

        self.save_path = "../checkpoints/" + self.model_name + ".ckpt"

        if parameters.num_layers and parameters.layer_units:
            layer_units = [int(l) for l in parameters.layer_units.split(',')]
            if len(layer_units) != int(parameters.num_layers):
                raise ValueError('Invalid layer parameters')
            self.num_layers = int(parameters.num_layers)
            self.layer_units = layer_units
        else:
            self.num_layers = 2
            self.layer_units = [128,256]
        

        self.num_epochs = 50#200
        self.batch_size = 16 #32
        self.early_stopping_tolerance = 5
        self.early_stopping_counter = 0
        self.last_validation_score = 0

        self.initialize()

    def initialize(self):
        self.train_inp, self.train_out, self.train_sequence_lengths = self.process_dataset(self.train)
        self.valid_inp, self.valid_out, self.valid_sequence_lengths = self.process_dataset(self.valid)

        self.num_features = self.train_inp.shape[-1]
        self.build_network()

    def build_network(self):
        # Based on https://www.kaggle.com/hbaderts/simple-feed-forward-neural-network-with-tensorflow
        self.inputs = tf.placeholder(tf.float32, [None, None, self.num_features])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.sequence_lengths = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # First layer
        w1 = tf.Variable(tf.random_normal([self.layer_units[0], self.num_features], stddev=0.01), name='w1')
        b1 = tf.Variable(tf.constant(0.1, shape=(self.layer_units[0], 1)), name='b1')
        
        flattend_inputs = tf.reshape(self.inputs, [tf.shape(self.inputs)[0]*tf.shape(self.inputs)[1], -1])
        y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(flattend_inputs)), b1)), keep_prob=self.dropout)
        final_y = y1

        if self.num_layers > 1:
            # Second layer
            w2 = tf.Variable(tf.random_normal([self.layer_units[1], self.layer_units[0]], stddev=0.01), name='w2')
            b2 = tf.Variable(tf.constant(0.1, shape=(self.layer_units[1], 1)), name='b2')
            y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=self.dropout_keep_prob)
            final_y = y2

        if self.num_layers > 2: 
            # Third layer
            w3 = tf.Variable(tf.random_normal([self.layer_units[2], self.layer_units[1]], stddev=0.01), name='w3')
            b3 = tf.Variable(tf.constant(0.1, shape=(self.layer_units[2], 1)), name='b3')
            y3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w3, y2), b3)), keep_prob=self.dropout_keep_prob)
            final_y = y3

        if self.num_layers > 3: 
            # Third layer
            w4 = tf.Variable(tf.random_normal([self.layer_units[3], self.layer_units[2]], stddev=0.01), name='w4')
            b4 = tf.Variable(tf.constant(0.1, shape=(self.layer_units[3], 1)), name='b4')
            y4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w4, y3), b4)), keep_prob=self.dropout_keep_prob)
            final_y = y4

        if self.num_layers > 4: 
            # Third layer
            w5 = tf.Variable(tf.random_normal([self.layer_units[4], self.layer_units[3]], stddev=0.01), name='w5')
            b5 = tf.Variable(tf.constant(0.1, shape=(self.layer_units[4], 1)), name='b5')
            y5 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w5, y4), b5)), keep_prob=self.dropout_keep_prob)
            final_y = y5

        # Output layer
        wo = tf.Variable(tf.random_normal([self.num_classes, self.layer_units[-1]], stddev=0.01), name='wo')
        bo = tf.Variable(tf.random_normal([self.num_classes, 1]), name='bo')
        yo = tf.transpose(tf.add(tf.matmul(wo, final_y), bo))


        # Second layer
        # w2 = tf.Variable(tf.random_normal([self.second_num_hidden_units, self.first_num_hidden_units], stddev=0.01), name='w2')
        # b2 = tf.Variable(tf.constant(0.1, shape=(self.second_num_hidden_units, 1)), name='b2')
        # y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=self.dropout)

        # # Output layer
        # wo = tf.Variable(tf.random_normal([self.num_classes, self.second_num_hidden_units], stddev=0.01), name='wo')
        # bo = tf.Variable(tf.random_normal([self.num_classes, 1]), name='bo')
        # yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

        # TODO: Verify the shape
        self.scores = tf.reshape(yo, [tf.shape(self.inputs)[0], tf.shape(self.inputs)[1], self.num_classes])
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.scores, self.label, self.sequence_lengths)

        loss = tf.reduce_mean(-log_likelihood)

        global_step = tf.contrib.framework.get_or_create_global_step() 
        lr = tf.train.exponential_decay(self.initial_learning_rate,
                                        global_step,
                                        self.decay_steps,
                                        self.learning_rate_decay_factor,
                                        staircase=True)

        # Logging with Tensorboard
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('loss', loss)
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

        if self.mode == 'train':
            init = tf.global_variables_initializer()
            merged = tf.summary.merge_all()

            batch_cutoffs = get_batch_cutoffs(self.train_inp, self.batch_size)
            validation_score = 0.0
            with tf.Session() as sess:
                train_writer = tf.summary.FileWriter('logs/', sess.graph)
                tf.set_random_seed(0)
                sess.run(init)
                step_idx = 0
                try:
                    for i in range(0, self.num_epochs):
                        loss_this_iter = 0
                        for s_idx, e_idx, batch_size in batch_cutoffs:
                            [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {self.inputs: self.train_inp[s_idx:e_idx],
                                self.label: self.train_out[s_idx:e_idx],
                                self.sequence_lengths: self.train_sequence_lengths[s_idx:e_idx],
                                self.dropout_keep_prob:self.dropout})
                            train_writer.add_summary(summary, step_idx)
                            step_idx += 1
                            loss_this_iter += loss_this_instance
                        print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
                        
                        if i % 1 == 0:
                            validation_score, stop = self.run_intermediate_validation(sess, validation_score, i)
                            if stop:
                                return
                except KeyboardInterrupt:
                    print "Terminating"
                    validation_score, _ = self.run_intermediate_validation(sess, validation_score, i)


    def run_intermediate_validation(self, sess, prev_validation_score, i):
        stop = False
        validation_score = self.validate(self.valid_inp, self.valid_out, self.valid_sequence_lengths, sess)
        print "Validation: " + str(validation_score)

        if validation_score > prev_validation_score:
            saver = tf.train.Saver()
            save_path = saver.save(sess, self.save_path)
            print("Model saved in path: %s" % save_path)
            self.early_stopping_counter = 0
            # self.last_validation_score = validation_score
            return validation_score, stop
        
        if self.early_stopping_counter >= self.early_stopping_tolerance and i >= 10:
            stop = True
        else:
            self.early_stopping_counter += 1

        return prev_validation_score, stop


    def validate(self, inp, out, sequence_lengths, sess):
        predictions = []
        gold = []
        batch_cutoffs = get_batch_cutoffs(inp, self.batch_size)
        for s_idx, e_idx, batch_size in batch_cutoffs:
            [scores, transition_params] = sess.run([self.scores, self.transition_params], feed_dict={self.inputs: inp[s_idx:e_idx],
                self.sequence_lengths: sequence_lengths[s_idx:e_idx],
                self.dropout_keep_prob:1.0})
            for i in range(len(scores)):
                score = scores[i][0:sequence_lengths[s_idx+i]]
                labels = out[s_idx+i][0:sequence_lengths[s_idx+i]]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                score, transition_params)
                for j in range(len(viterbi_sequence)):
                    predictions.append(viterbi_sequence[j])
                    gold.append(labels[j])
                   
        precision, recall, f1, _ = precision_recall_fscore_support(np.asarray(gold), np.asarray(predictions),
            average='micro', labels=[1])
        validation_score = f1
        return f1
        
    def process_dataset(self, dataset):
        # TODO: If need to shuffle, need to do before this

        num_features = len(dataset[0].full_code_representation[0]['features'])
        max_candidates = 0

        sequence_lengths = []

        for association in dataset:
            num_candidates = len(association.full_code_representation)
            sequence_lengths.append(num_candidates)
            
            if num_candidates > max_candidates:
                max_candidates = num_candidates

        inp = np.full((len(dataset), max_candidates, num_features), PAD_IDX)
        out = np.full((len(dataset), max_candidates), PAD_IDX)

        for i in range(len(dataset)):
            association = dataset[i]
            for j in range(len(association.full_code_representation)):
                candidate = association.full_code_representation[j]
                inp[i][j] = candidate['features']
                
                if candidate['pseudo_class'] == 'False':
                    out[i][j] = 0
                elif candidate['pseudo_class'] == 'True':
                    out[i][j] = 1
                elif candidate['pseudo_class'] == 'JAVA':
                    out[i][j] = 2
                else:
                    raise ValueError('Invalid class')

        return inp, out, np.asarray(sequence_lengths)

    def classify(self, associations):
        inp, out, sequence_lengths = self.process_dataset(associations)
        num_predictions = 0
        num_invalid_predictions = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)

            batch_cutoffs = get_batch_cutoffs(inp, self.batch_size)
            for s_idx, e_idx, batch_size in batch_cutoffs:
                [scores, transition_params] = sess.run([self.scores, self.transition_params],
                    feed_dict={self.inputs: inp[s_idx:e_idx],
                    self.sequence_lengths: sequence_lengths[s_idx:e_idx], self.dropout_keep_prob:1.0})
                for i in range(len(scores)):
                    score = scores[i][0:sequence_lengths[s_idx+i]]
                    labels = out[s_idx+i][0:sequence_lengths[s_idx+i]]
                    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    score, transition_params)
                    for j in range(len(viterbi_sequence)):
                        if associations[s_idx+i].full_code_representation[j]['pseudo_class'] != 'JAVA':
                            num_predictions += 1
                            if viterbi_sequence[j] not in [0,1]:
                                num_invalid_predictions += 1
                        if viterbi_sequence[j] not in [0,1]:
                            associations[s_idx+i].full_code_representation[j]['prediction'] = 0
                        else:
                            associations[s_idx+i].full_code_representation[j]['prediction'] = viterbi_sequence[j]
            
            print('Number of predictions: ' + str(num_predictions))
            print('Number of invalid predictions: ' + str(num_invalid_predictions))
            compute_metrics(associations)

    def classify_candidates(self, candidate_pairs):
        associations = [a for _, a in candidate_pairs]
        inp, out, sequence_lengths = self.process_dataset(associations)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)

            batch_cutoffs = get_batch_cutoffs(inp, self.batch_size)
            for s_idx, e_idx, batch_size in batch_cutoffs:
                [scores, transition_params] = sess.run([self.scores, self.transition_params],
                    feed_dict={self.inputs: inp[s_idx:e_idx],
                    self.sequence_lengths: sequence_lengths[s_idx:e_idx], self.dropout_keep_prob:1.0})
                for i in range(len(scores)):
                    score = scores[i][0:sequence_lengths[s_idx+i]]
                    labels = out[s_idx+i][0:sequence_lengths[s_idx+i]]
                    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    score, transition_params)
                    for j in range(len(viterbi_sequence)):
                        if viterbi_sequence[j] not in [0,1]:
                            associations[s_idx+i].full_code_representation[j]['prediction'] = 0
                        else:
                            associations[s_idx+i].full_code_representation[j]['prediction'] = viterbi_sequence[j]
            
            candidates = []
            for idx,(c_idx,_) in enumerate(candidate_pairs):
                candidates.append(associations[idx].full_code_representation[c_idx])
            return compute_candidate_metrics(candidates)
    
    def get_candidate_predictions(self, candidate_pairs):
        associations = [a for _, a in candidate_pairs]
        inp, out, sequence_lengths = self.process_dataset(associations)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)

            batch_cutoffs = get_batch_cutoffs(inp, self.batch_size)
            for s_idx, e_idx, batch_size in batch_cutoffs:
                [scores, transition_params] = sess.run([self.scores, self.transition_params],
                    feed_dict={self.inputs: inp[s_idx:e_idx],
                    self.sequence_lengths: sequence_lengths[s_idx:e_idx], self.dropout_keep_prob:1.0})
                for i in range(len(scores)):
                    score = scores[i][0:sequence_lengths[s_idx+i]]
                    labels = out[s_idx+i][0:sequence_lengths[s_idx+i]]
                    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                    score, transition_params)
                    for j in range(len(viterbi_sequence)):
                        if viterbi_sequence[j] not in [0,1]:
                            associations[s_idx+i].full_code_representation[j]['prediction'] = 0
                        else:
                            associations[s_idx+i].full_code_representation[j]['prediction'] = viterbi_sequence[j]
            
            candidates = []
            for idx,(c_idx,_) in enumerate(candidate_pairs):
                if associations[idx].full_code_representation[c_idx]['pseudo_class'] != 'JAVA':
                    candidates.append(associations[idx].full_code_representation[c_idx])
        return [c['prediction'] for c in candidates]
