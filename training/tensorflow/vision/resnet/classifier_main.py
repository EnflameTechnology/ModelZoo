#!/usr/bin/env python
#
# Copyright 2018-2021 Enflame. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import tensorflow as tf
import numpy as np
import traceback
import tops_models.common_utils as common_utils
import tops_models.tf_utils as tf_utils
from tops_models.datasets import dataset_mapping
from data_utils.data_processing import DataProcessing, CIFARDataProcessing
from utils.network import build_model
from utils.learning_rate import LearningRate
from utils.optimizer import get_optimizer
from utils.loss import LossFunc


class Benchmark(object):
    def __init__(self, params, logger):
        self.params = params
        self.logger = logger
        self.dtype = tf_utils.str2tf[params['dtype']]
        if params['is_training']:
            self.training_step_num_per_epoch = self._training_step_num_per_epoch()
            self.logger.info("training_step_per_epoch: {}".format(
                self.training_step_num_per_epoch))
        if params['enable_evaluate']:
            self.evaluate_step_num_per_epoch = self._evaluate_step_num_per_epoch()
            self.logger.info("evaluate_step_per_epoch: {}".format(
                self.evaluate_step_num_per_epoch))
        self.cpu_device = '/cpu:0'
        self.default_device = common_utils.device_mapping[self.params['device']]

    def _training_step_num_per_epoch(self):
        """calculate number of training steps of one epoch"""
        train_image_number = dataset_mapping[self.params['dataset']][
            'train_images_number']
        self.logger.info("image number for training in {} is {}".format(
            self.params['dataset'], train_image_number))
        one_step_number = self.params['batch_size'] * self.params[
            'num_cluster'] * \
                          self.params['hvd_size']
        self.logger.info(
            "image number of one-step training is {}".format(one_step_number))
        assert one_step_number <= train_image_number, 'Batch size {} too large for {}'.format(
            self.params['batch_size'], self.params['dataset'])
        step_num_per_epoch = int(np.floor(train_image_number / one_step_number))
        if self.params['training_step_per_epoch'] >= 0:
            assert one_step_number * self.params[
                'training_step_per_epoch'] < train_image_number, \
                'training_step_per_epoch must be less than {}'.format(
                    step_num_per_epoch)
            step_num_per_epoch = self.params['training_step_per_epoch']

        return step_num_per_epoch

    def _evaluate_step_num_per_epoch(self):
        """calculate number of evaluate steps of one epoch"""
        eval_image_number = dataset_mapping[self.params['dataset']][
            'val_images_number']
        self.logger.info("image number for evaluate in {} is {}".format(
            self.params['dataset'], eval_image_number))
        one_step_number = self.params['batch_size']
        self.logger.info(
            "image number of one-step evaluate is {}".format(one_step_number))
        assert one_step_number <= eval_image_number, 'Batch size {} too large for {}'.format(
            self.params['batch_size'], self.params['dataset'])
        step_num_per_epoch = int(np.floor(eval_image_number / one_step_number))
        if self.params['evaluate_step_per_epoch'] >= 0:
            assert one_step_number * self.params[
                'evaluate_step_per_epoch'] < eval_image_number, \
                'evaluate_step_per_epoch must be less than {}'.format(
                    step_num_per_epoch)
            step_num_per_epoch = self.params['evaluate_step_per_epoch']

        return step_num_per_epoch

    def _exclude_batch_norm(self, name):
        """exclude variables of BN from l2loss calculation"""
        return 'BatchNorm' not in name \
               and 'preact' not in name \
               and 'postnorm' not in name \
               and 'batch_normalization' not in name

    def _generate_data(self, data_dir, is_training):
        """
            Generate input data for training/evaluate/inference.
            Both real dataset and synthetic data are supported.
            Args:
                data_dir: The filepath of dataset, only tfrecords supported now.
                is_training: A bool
                    if training: dataset will be shuffled.
                    if not: batch size only for one cluster.
            Return:
                An uninitialized dataset iterator.
        """
        if self.params['dataset'] == 'cifar10':
          dataset = CIFARDataProcessing(is_training=is_training,
                                        data_dir=data_dir,
                                        dtype=self.dtype,
                                        params=self.params,
                                        target_device=self.default_device)
        else:
          dataset = DataProcessing(is_training=is_training,
                                   data_dir=data_dir,
                                   dtype=self.dtype,
                                   params=self.params,
                                   target_device=self.default_device)

        batch_size = self.params['batch_size']

        if self.params['use_synthetic_data']:
            self.logger.info("Use synthetic data instead of real images.")
            data = dataset.get_synth_input_fn(batch_size=batch_size)
        else:
            data = dataset.input_fn(batch_size=batch_size)

        iterator = data.make_initializable_iterator()

        return iterator

    def build_training_graph(self, iterator, global_step):
        with tf.device(self.cpu_device):
            weight_init = tf.glorot_uniform_initializer(dtype=tf.float32)
            try:
                (image_train, label_train) = iterator.get_next()
            except Exception as ex:
                traceback.print_exc()
            assert image_train.dtype == self.dtype
            if self.params['debug_mode']:
                tf.summary.image('images', tf.cast(image_train, tf.float32),
                                 max_outputs=1)

        with tf.device(self.default_device):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE,
                                   use_resource=self.params['use_resource']):
                logits, _ = build_model(image_train, self.params,
                                        is_training=True,
                                        weight_init=weight_init)
                logits = tf.cast(logits, tf.float32)
                loss_func = LossFunc(self.params, self.logger)
                cross_entropy = loss_func(logits=logits, labels=label_train)
                tf.identity(cross_entropy, name='cross_entropy')
                tf.summary.scalar('cross_entropy', cross_entropy,
                                  family='cross_entropy')
                params = tf.trainable_variables()
                l2_loss = self.params['weight_decay'] * tf.add_n(
                    [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in params
                        if self._exclude_batch_norm(v.name)])
                tf.identity(l2_loss, name='l2_loss')
                tf.summary.scalar('l2_loss', l2_loss)
                learning_rate_func = LearningRate(self.params,
                                                  self.training_step_num_per_epoch,
                                                  global_step,
                                                  self.logger)
                if self.params['optimizer'] == "lars":
                    loss_op = cross_entropy
                    learning_rate = learning_rate_func.poly_learning_rate()
                else:
                    loss_op = cross_entropy + l2_loss
                    learning_rate = learning_rate_func.decay_learning_rate()
                tf.identity(learning_rate, name='learning_rate')
                tf.summary.scalar('learning_rate', learning_rate,
                                  family='learning_rate')

                opt = get_optimizer(self.params, learning_rate)
                # Automatic Mixed Precision Training
                enable_amp = self.params['enable_amp']
                if enable_amp:
                    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

                if self.params['enable_horovod']:
                    import horovod.tensorflow as hvd
                    if self.params['dtype'] == 'fp16' or\
                        self.params['hvd_compression'] == 'fp16':
                        compression = hvd.Compression.fp16
                    else:
                        compression = hvd.Compression.none

                    opt = hvd.DistributedOptimizer(opt, compression=compression)

                accuracy = tf.metrics.accuracy(label_train,
                                               tf.argmax(logits, axis=1),
                                               name="metric")
                tf.identity(accuracy[1], name='train_accuracy')
                tf.summary.scalar('train_accuracy', accuracy[1],
                                  family='train_accuracy')

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_op = opt.minimize(loss_op, global_step=global_step,
                                        name='ApplyEF')

                train_op = tf.group(update_ops, train_op,
                                    name='train_ops_group')
        return train_op, accuracy, loss_op

    def build_evaluate_graph(self, iterator):
        with tf.device(self.cpu_device):
            weight_init = tf.glorot_uniform_initializer(dtype=tf.float32)
            try:
                (images_eval, labels_eval) = iterator.get_next()
            except Exception as ex:
                traceback.print_exc()
            assert images_eval.dtype == self.dtype
        with tf.device(self.default_device):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE,
                                   use_resource=self.params['use_resource']):
                logits, _ = build_model(images_eval, self.params,
                                        is_training=False,
                                        weight_init=weight_init)
                logits = tf.cast(logits, tf.float32)
                correct_predictions = tf.equal(
                    tf.cast(tf.argmax(logits, axis=1), tf.int32),
                    labels_eval)
                accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32))
                tf.identity(accuracy, name='accuracy')
                with tf.device(self.cpu_device):
                    accuracy_top_5 = tf.reduce_mean(
                        tf.cast(tf.nn.in_top_k(predictions=logits,
                                               targets=labels_eval, k=5),
                                dtype=tf.float32))
                    tf.identity(accuracy_top_5, name='accuracy_top_5')

        return accuracy, accuracy_top_5

    def build_inference_graph(self, iterator):
        with tf.device(self.cpu_device):
            weight_init = tf.glorot_uniform_initializer(dtype=tf.float32)
            try:
                (image_infer, label_infer) = iterator.get_next()
            except Exception as ex:
                traceback.print_exc()
            assert image_infer.dtype == self.dtype
            tf.summary.image('images', tf.cast(image_infer, tf.float32),
                             max_outputs=self.params['batch_size'])
        with tf.device(self.default_device):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE,
                                   use_resource=self.params['use_resource']):
                logits, _ = build_model(image_infer, self.params,
                                        is_training=False,
                                        weight_init=weight_init)
                predictions = {
                    'predict class': tf.argmax(tf.nn.softmax(logits), axis=1),
                    'real class': label_infer,
                    'prob': tf.nn.softmax(logits, name='softmax_tensor')}
        return predictions

    def variables_initial_op(self):
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        if self.params['optimizer'] == 'adam':
            collection_variables = tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES)
            collection_init_op = tf.variables_initializer(collection_variables)
            return tf.group(
                *(collection_init_op, global_init_op, local_init_op))
        else:
            return tf.group(*(global_init_op, local_init_op))


if __name__ == '__main__':
    params = {
        'batch_size': 4,
        'dtype': 'fp32',
        'device': 'dtu',
        'optimizer': 'momentum',
        'is_training': True,
        'enable_horovod': False,
        'enable_evaluate': True,
        'use_resource': True
    }
    bench = Benchmark(params)
    print(bench.cpu_device)
