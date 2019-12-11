from model_utils import *
import os
import numpy as np
import sys
import tensorflow as tf

import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

class FeedForwardNN:
    '''Binary classifier implemented as a feedforward network.'''
    def __init__(self, train, test, valid, parameters, pu_learning=False, aux=False,
        train_inp=None, train_out=None, valid_inp=None, valid_out=None, mode='train'):
        self.train = train
        self.test = test
        self.valid = valid
        self.pu_learning = pu_learning
        self.model_name = parameters.model_name

        self.aux = aux
        self.mode = mode


        if aux:
            self.train_inp = train_inp
            self.train_out = train_out
            self.valid_inp = valid_inp
            self.valid_out = valid_out
        else:
            if self.model_name == 'more_data_feedforward':
                self.delete_weight = float(parameters.delete_weight)
                self.train_inp, self.train_out, self.train_weights = self.process_dataset(self.train)
                self.valid_inp, self.valid_out, self.valid_weights = self.process_dataset(self.valid)
            else:
                self.train_inp, self.train_out = self.process_dataset(self.train)
                self.valid_inp, self.valid_out = self.process_dataset(self.valid)

        self.num_classes = 2

        if parameters.decay_steps:
            self.decay_steps = int(parameters.decay_steps)
        else:
            self.decay_steps = 500

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
        
        self.num_epochs = 50
        self.batch_size = 16
        self.early_stopping_tolerance = 5
        self.early_stopping_counter = 0
        self.last_validation_score = 0

        self.parameters = parameters

        self.initialize()

    def initialize(self):
        self.num_features = self.train_inp.shape[1]
        if not self.model_name == 'more_data_feedforward':
            self.train_weights = np.ones_like(self.train_out, dtype=float)

        self.build_network()

    def build_network(self):
        # Based on https://www.kaggle.com/hbaderts/simple-feed-forward-neural-network-with-tensorflow
        self.inputs = tf.placeholder(tf.float32, [None, self.num_features])
        self.label = tf.placeholder(tf.int32, [None, 1])
        self.weights = tf.placeholder(tf.float32, [None, 1])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # First layer
        w1 = tf.Variable(tf.random_normal([self.layer_units[0], self.num_features], stddev=0.01), name='w1')
        b1 = tf.Variable(tf.constant(0.1, shape=(self.layer_units[0], 1)), name='b1')
        y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(self.inputs)), b1)), keep_prob=self.dropout_keep_prob)
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

        self.pred = tf.nn.softmax(yo)
        self.one_best = tf.argmax(self.pred, 1)

        label_onehot = tf.one_hot(self.label, self.num_classes)

        loss = tf.reduce_mean(self.weights * tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=label_onehot))
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

        if not self.aux and self.mode == 'train':
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
                                self.label: np.transpose(np.array([self.train_out[s_idx:e_idx]])),
                                self.weights: np.transpose(np.array([self.train_weights[s_idx:e_idx]])),
                                self.dropout_keep_prob: self.dropout})
                            train_writer.add_summary(summary, step_idx)
                            step_idx += 1
                            loss_this_iter += loss_this_instance
                        print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
                        sys.stdout.flush()
                        
                        if i % 1 == 0:
                            validation_score, stop = self.run_intermediate_validation(sess, validation_score, i)
                            if stop:
                                return
                except KeyboardInterrupt:
                    print "Terminating"
                    validation_score = self.run_intermediate_validation(sess, validation_score, i)

    def run_intermediate_validation(self, sess, prev_validation_score, i):
        stop = False
        validation_score = self.validate(self.valid_inp, self.valid_out, sess)

        print "Validation: " + str(validation_score)
        sys.stdout.flush()

        if validation_score > prev_validation_score:
            saver = tf.train.Saver()
            save_path = saver.save(sess, self.save_path)
            print("Model saved in path: %s" % save_path)
            sys.stdout.flush()
            self.early_stopping_counter = 0
            return validation_score, stop
            
        if self.early_stopping_counter >= self.early_stopping_tolerance and i >= 10:
            stop = True
        else:
            self.early_stopping_counter += 1

        return prev_validation_score, stop

    def validate(self, inp, out, sess):
        predictions = []
        gold = []
        batch_cutoffs = get_batch_cutoffs(inp, self.batch_size)
        for s_idx, e_idx, batch_size in batch_cutoffs:
            [pred_this_instance] = sess.run([self.one_best], feed_dict={self.inputs: inp[s_idx:e_idx], self.dropout_keep_prob: 1.0})
            for i in range(len(pred_this_instance)):
                predictions.append(pred_this_instance[i])
                gold.append(out[s_idx+i])

        precision, recall, f1, _ = precision_recall_fscore_support(np.asarray(gold), np.asarray(predictions),
            average='binary', pos_label=1)
        validation_score = f1
        return f1
  
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

    def process_weighted_dataset(self, dataset):
        temp = []
        for association in dataset:
            for candidate in association.full_code_representation:
                c = dict()
                c['feature_vector'] = candidate['features']
                if candidate['is_associated']:
                    c['class'] = 1
                else:
                    c['class'] = 0

                if association.src == 'added':
                    c['weight'] = 1.0
                elif association.src == 'deleted':
                    c['weight'] = self.delete_weight
                else:
                    raise ValueError('Invalid data source')
                temp.append(c)

        random.shuffle(temp)
        inp = []
        out = []
        weights = []

        for t in temp:
            inp.append(t['feature_vector'])
            out.append(t['class'])
            weights.append(t['weight'])

        return np.asarray(inp), np.asarray(out), np.asarray(weights)

    def classify_candidates(self, candidate_pairs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)
            candidates = [c for c, _ in candidate_pairs]
            inputs = [c['features'] for c in candidates]
            [pred_this_instance] = sess.run([self.one_best], feed_dict={self.inputs: inputs, self.dropout_keep_prob: 1.0})
            for i in range(len(pred_this_instance)):
                candidates[i]['prediction'] = pred_this_instance[i]
            return compute_candidate_metrics(candidates)
    
    def get_candidate_predictions(self, candidate_pairs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)
            candidates = [c for c, _ in candidate_pairs]
            inputs = [c['features'] for c in candidates]
            [pred_this_instance] = sess.run([self.one_best], feed_dict={self.inputs: inputs, self.dropout_keep_prob: 1.0})
            for i in range(len(pred_this_instance)):
                candidates[i]['prediction'] = pred_this_instance[i]
        return [c['prediction'] for c in candidates]

    def classify(self, associations):
        total_correct = 0
        total_candidates = 0
        per_example_accumulation = 0.0
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)
            for association in associations:
                correct = 0
                candidates = association.full_code_representation
                inputs = [c['features'] for c in candidates]
                [pred_this_instance] = sess.run([self.one_best], feed_dict={self.inputs: inputs, self.dropout_keep_prob: 1.0})
                for i in range(len(pred_this_instance)):
                    candidates[i]['prediction'] = pred_this_instance[i]

            compute_metrics(associations)

    def predict_probs(self, inputs, path_to_use=None):
        if not path_to_use:
            path_to_use = self.save_path
        total_correct = 0
        total_candidates = 0
        per_example_accumulation = 0.0
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=path_to_use)
            [probs] = sess.run([self.pred], feed_dict={self.inputs: inputs, self.dropout_keep_prob:1.0})
        return probs