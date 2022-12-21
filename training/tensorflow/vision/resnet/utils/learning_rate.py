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

import tensorflow as tf

BOUNDARY_EPOCHS = [30, 60, 80, 90]
DECAY_RATES = [1, 0.1, 0.01, 0.001, 1e-4]
BATCH_DENOM = 256
LR_SCHEDULE = [
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


class LearningRate(object):
    def __init__(self, params, batches_per_epoch, global_step, logger):
        """
           Args:
              params: all collected parameters for training.
              batches_per_epoch means total batch size of one epoch, with which learning rate is correlated.
              global_step: a global step counter, used to compare with epoch boundaries.
        """
        self.params = params
        self.batches_per_epoch = batches_per_epoch
        self.global_step = global_step
        self.logger = logger
        self.logger.info("batches_per_epoch={}".format(batches_per_epoch))

    def learning_rate_schedule(self):
        """A reserved interface for fixed step learning rate.
           Returns:
               A tensor of learning rate.
        """
        scaled_lr = self.params['base_learning_rate'] * self.params[
            'batch_size'] * \
                    self.params['num_cluster'] * self.params[
                        'hvd_size'] / BATCH_DENOM

        decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                      self.global_step / LR_SCHEDULE[0][1])
        for mult, start_epoch in LR_SCHEDULE:
            decay_rate = tf.where(self.global_step < start_epoch,
                                  decay_rate, scaled_lr * mult)
        return decay_rate

    def get_fixed_learning_rate(self):
        """Handles fixed learning rate which are tuned case by case.
           Returns:
               A tensor of fixed learning rate.
        """
        if self.params['optimizer'] == 'adam':
            learning_rate = 1e-4
        elif self.params['optimizer'] == 'sgd':
            if self.params['batch_size'] == 4 and self.params[
                'num_cluster'] == 4:
                learning_rate = 2e-2
            else:
                learning_rate = self.params['base_learning_rate'] * self.params[
                    'batch_size'] * \
                                self.params['num_cluster'] * self.params[
                                    'hvd_size'] / 256.0
        else:
            learning_rate = 1e-4
        return learning_rate

    def decay_learning_rate(self):
        initial_learning_rate = self.params['base_learning_rate'] * self.params[
            'batch_size'] * \
                                self.params['num_cluster'] * self.params[
                                    'hvd_size'] / BATCH_DENOM
        self.logger.info(
            "initial_learning_rate={}".format(initial_learning_rate))
        vals = [initial_learning_rate * decay for decay in DECAY_RATES]
        boundaries = [int(self.batches_per_epoch * epoch) for epoch in
                      BOUNDARY_EPOCHS]
        lr = tf.train.piecewise_constant(self.global_step, boundaries, vals)
        warmup_steps = int(self.batches_per_epoch * 5)
        self.logger.info(
            "batches_per_epoch={}, warmup_steps={}".format(
                self.batches_per_epoch, warmup_steps))
        warmup_lr = (
                initial_learning_rate * tf.cast(self.global_step,
                                                tf.float32) / tf.cast(
            warmup_steps, tf.float32))

        return tf.cond(self.global_step < warmup_steps, lambda: warmup_lr,
                       lambda: lr)

    def poly_learning_rate(self):
        total_batch_size = self.params['batch_size'] * self.params[
            'num_cluster'] * self.params['hvd_size']
        if (total_batch_size == 256*8 and self.params['epoch'] == 50) \
            or (total_batch_size == 256*16 and self.params['epoch'] == 60):
            plr = 7.4
            w_epochs = 2
        elif total_batch_size < 8192:
            plr = 5.0
            w_epochs = 5
        elif total_batch_size < 16384:
            plr = 20.0
            w_epochs = 5
        elif total_batch_size < 40960:
            plr = 25.0
            w_epochs = 10
        else:
            plr = 32.0
            w_epochs = 14

        if self.params['poly_learning_rate'] is not None:
            plr = self.params['poly_learning_rate']

        if self.params['poly_warmup_epochs'] is not None:
            w_epochs = self.params['poly_warmup_epochs']

        w_steps = int(w_epochs * self.batches_per_epoch)
        if w_steps > 0:
            wrate = plr * tf.cast(self.global_step, tf.float32) / tf.cast(w_steps,
                                                                      tf.float32)
        else:
            wrate = plr

        train_steps = self.batches_per_epoch * self.params['epoch']
        self.logger.info(
            "poly_learning_rate:w_steps={},wrate={},train_steps={}, "
            "batches_per_epoch={},epoch={},total_batch_size={},plr={}".format(
                w_steps, wrate, train_steps, self.batches_per_epoch,
                self.params['epoch'],
                total_batch_size, plr
            ))
        min_step = tf.constant(1, dtype=tf.int64)
        decay_steps = tf.maximum(min_step,
                                 tf.subtract(self.global_step, w_steps))
        self.logger.info("decay_steps={}, global_step={}".format(decay_steps,
                                                                 self.global_step))
        poly_rate = tf.train.polynomial_decay(
            plr,
            decay_steps,
            train_steps - w_steps + 1,
            power=2.0)
        return tf.where(self.global_step <= w_steps, wrate, poly_rate)

    def cosine_learning_rate(self):
        global_batch_size = self.params['batch_size'] * \
                            self.params['num_cluster'] * self.params['hvd_size']
        if global_batch_size < 8192:
            w_epochs = 5
        elif global_batch_size < 16384:
            w_epochs = 5
        elif global_batch_size < 32768:
            w_epochs = 5
        else:
            w_epochs = 14
        warmup_steps = int(w_epochs * self.batches_per_epoch)
        decay_steps = self.batches_per_epoch * self.params['epoch']

        initial_learning_rate = self.params['base_learning_rate'] * \
                                global_batch_size / BATCH_DENOM
        warmup_lr = (
                initial_learning_rate *
                tf.cast(self.global_step, tf.float32) /
                tf.cast(warmup_steps, tf.float32)
        )
        lr = tf.train.cosine_decay(initial_learning_rate, self.global_step,
                                   decay_steps)
        learning_rate = tf.cond(self.global_step < warmup_steps,
                                lambda: warmup_lr,
                                lambda: lr)
        return learning_rate
