#!/usr/bin/env python
#
# Copyright 2018-2020 Enflame. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf


class LossFunc(object):
    def __init__(self, params, logger):
        self.params = params
        self.logger = logger

    def __call__(self, logits, labels):
        if self.params['label_smoothing'] != 0:
            # on_value = 1 - 0.1 + 0.1/params['num_class']
            # off_value = 0.1/params['num_class']
            # label_train_one_hot = tf.one_hot(labels, self.params['num_class'], on_value=on_value, off_value=off_value)
            self.logger.info(
                "label_smoothing is not ZERO, use softmax_cross_entropy to calculate loss")
            label_train_one_hot = tf.one_hot(labels, self.params['num_class'])
            return tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=label_train_one_hot,
                                                   label_smoothing=self.params[
                                                       'label_smoothing'])
        else:
            self.logger.info(
                "label_smoothing is ZERO, use sparse_softmax_cross_entropy to calculate loss")
            return tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                          labels=labels)
