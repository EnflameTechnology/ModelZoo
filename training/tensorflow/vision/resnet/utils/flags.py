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
from collections import OrderedDict
import uuid
import sys

sys.path.append("../")

FLAGS = tf.app.flags.FLAGS

"""test process flags"""
tf.app.flags.DEFINE_string('build_id', "9999",
                           help=(
                               """Build id. Default 9999 if no specific one"""))
tf.app.flags.DEFINE_string('nn_base_info', 'E2020SW109NN0002D0910',
                           help=(
                               """TopsModels modified version"""))
tf.app.flags.DEFINE_string('data_dir', '',
                           help=(
                               """Assigned path to dataset"""))
tf.app.flags.DEFINE_boolean('is_training', True,
                            help=(
                                """Whether to do training. True for training;"""
                                """False for inference"""))
tf.app.flags.DEFINE_enum('device', 'dtu',
                         ['cpu', 'gpu', 'dtu', 'tpu', 'xla_gpu', 'xla_cpu'],
                         help=(
                             """Assign on which device to execute model"""))
tf.app.flags.DEFINE_enum('dataset', 'imagenet2',
                         ['imagenet2', 'imagenet10', 'imagenet50', 'imagenet',
                          'mnist', 'cifar10', 'flowers'],
                         help=(
                             """dataset for training"""))
tf.app.flags.DEFINE_enum('dtype', 'fp32', ['fp16', 'fp32', 'bf16'],
                         help=(
                             """Brain floating point format"""))
tf.app.flags.DEFINE_enum('data_format', 'NHWC', ['NHWC', 'CHNW', 'NCHW'],
                         help=(
                             """Data format of input data. NHWC[channel_last]"""
                             """for CPU|TPU; NCHW[channels first] for GPU."""
                             """For dtu, option is NHWC or CHNW. CHNW will """
                             """improve performance"""))
tf.app.flags.DEFINE_float('resnet_version', 1.5,
                          help=(
                              """resnet version, option: 1|1.5|2|3, 1 and 2"""
                              """are follow paper; 1.5 is the version from"""
                              """Nvidia and Mlperf; 3 is the slim one"""))
tf.app.flags.DEFINE_integer('depth', 50,
                            help=(
                                """resnet depth"""))
tf.app.flags.DEFINE_integer('batch_size', 16,
                            help=(
                                """Number of images to process in one batch"""))
tf.app.flags.DEFINE_integer('epoch', 1,
                            help=(
                                """Number of epochs for training or inference"""))
tf.app.flags.DEFINE_integer('display_step', 10,
                            help=(
                                """Number of every steps to display loss"""))
tf.app.flags.DEFINE_integer('skip_steps', 15,
                            help=(
                                """Number of skip steps to calculate average FPS"""))
tf.app.flags.DEFINE_integer('num_between_eval', 5,
                            help=(
                                """Number of epochs between two cycle of"""
                                """evaluate processing"""))
tf.app.flags.DEFINE_integer('num_between_saver', 5,
                            help=(
                                """Number of epochs between two save points"""))
tf.app.flags.DEFINE_enum('optimizer', 'momentum',
                         ['sgd', 'momentum', 'rmsprop', 'adam', 'lars'],
                         help=(
                             """Use which optimizer for training process"""))
tf.app.flags.DEFINE_float('base_learning_rate', 128e-3,
                          help=(
                              """Basic learning rate"""))
tf.app.flags.DEFINE_float('poly_learning_rate', None,
                          help=(
                              """learning rate for poly learning rate decay"""))
tf.app.flags.DEFINE_float('poly_warmup_epochs', None,
                          help=(
                              """warmup epochs for poly learning rate decay"""))
tf.app.flags.DEFINE_float('label_smoothing', 0.,
                          help=(
                              """A scale when calculate loss. Default 0 will """
                              """use sparse_softmax_cross_entropy; others"""
                              """will use softmax_cross_entropy"""))
tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                          help=(
                              """weight decay of L2loss"""))
tf.app.flags.DEFINE_boolean('use_resource', True,
                            help=(
                                """True to enable use ResourceVariable; """
                                """False to disable"""))
tf.app.flags.DEFINE_integer('training_step_per_epoch', -1,
                            help=(
                                """Number of training steps for each epoch"""
                                """Default -1, all images in train dataset"""
                                """will be feed. A specific int larger than 0"""
                                """is allowed to replace it"""))
tf.app.flags.DEFINE_integer('evaluate_step_per_epoch', -1,
                            help=(
                                """Number of evaluate steps for each epoch"""
                                """Default -1, all images in eval dataset"""
                                """will be feed. A specific int larger than 0"""
                                """is allowed to replace it"""))
tf.app.flags.DEFINE_boolean('use_synthetic_data', False,
                            help=(
                                """True for synthetic_data"""))
tf.app.flags.DEFINE_float('batch_norm_decay', None,
                            help=(
                                """the value for batch norm decay"""))

"""optional function flags"""
tf.app.flags.DEFINE_boolean('enable_evaluate', False,
                            help=(
                                """True will enable eval"""))
tf.app.flags.DEFINE_boolean('enable_horovod', False,
                            help=(
                                """True will import horovod and use on """
                                """distributed training"""))
tf.app.flags.DEFINE_boolean('enable_profiler', False,
                            help=(
                                """True will enable TF profiler"""))
tf.app.flags.DEFINE_boolean('enable_dump_graph', False,
                            help=(
                                """True will dump graph as pbtxt before and """
                                """after initializer"""))
tf.app.flags.DEFINE_boolean('enable_saver', True,
                            help=(
                                """True will save checkpoints snapshot and """
                                """restore from the latest one if exists"""))
tf.app.flags.DEFINE_boolean('debug_mode', False,
                            help=(
                                """True will enable debug mode, including """
                                """fixed seed, fixed data pipeline, logger """
                                """with more message, single tensorflow """
                                """parallelism_thread, etc."""))
tf.app.flags.DEFINE_boolean('xla_jit', False,
                            help=("""True will use jit scope and optimizer """
                                  """option global_jit_level ON_1"""))
tf.app.flags.DEFINE_enum('hvd_compression', 'fp32',['fp16', 'fp32'],
                            help=(
                                """data type for horovod allreduce, """
                                """now support fp32 and fp16, if model dtype"""
                                """ is fp16 hvd compression will be fp16."""))
tf.app.flags.DEFINE_boolean('enable_amp', False,
                            help=(
                                """True will enable amp training"""))
tf.app.flags.DEFINE_integer('seed', None,
                            help=(
                                """set the seed for python, numpy and tensorflow random number generators"""))


class InfoDict(object):
    def __init__(self):
        self.test_info = self._test_information()
        self.func_info = self._functional_information()

    def _test_information(self):
        test_info = OrderedDict([('hvd_compression',FLAGS.hvd_compression),
                                 ('nn_base_info', FLAGS.nn_base_info),
                                 ('build_id', FLAGS.build_id),
                                 ('test_id', uuid.uuid4().hex),
                                 ('is_training', FLAGS.is_training),
                                 ('data_dir', FLAGS.data_dir),
                                 ('device', FLAGS.device.lower()),
                                 ('dataset', FLAGS.dataset.lower()),
                                 ('dtype', FLAGS.dtype.lower()),
                                 ('data_format', FLAGS.data_format.upper()),
                                 ('resnet_version', FLAGS.resnet_version),
                                 ('depth', FLAGS.depth),
                                 ('model', 'resnet{}_v{}'.format(FLAGS.depth, FLAGS.resnet_version)),
                                 ('batch_size', FLAGS.batch_size),
                                 ('epoch', FLAGS.epoch),
                                 ('display_step', FLAGS.display_step),
                                 ('skip_steps', FLAGS.skip_steps),
                                 ('num_between_eval', FLAGS.num_between_eval),
                                 ('num_between_saver', FLAGS.num_between_saver),
                                 ('optimizer', FLAGS.optimizer.lower()),
                                 ('base_learning_rate', FLAGS.base_learning_rate),
                                 ('poly_learning_rate', FLAGS.poly_learning_rate),
                                 ('poly_warmup_epochs', FLAGS.poly_warmup_epochs),
                                 ('label_smoothing', FLAGS.label_smoothing),
                                 ('weight_decay', FLAGS.weight_decay),
                                 ('use_resource', FLAGS.use_resource),
                                 ('training_step_per_epoch', FLAGS.training_step_per_epoch),
                                 ('evaluate_step_per_epoch', FLAGS.evaluate_step_per_epoch),
                                 ('use_synthetic_data', FLAGS.use_synthetic_data),
                                 ('hvd_size', 1),
                                 ('local_rank', 0),
                                 ('rank', 0),
                                 ('batch_norm_decay',FLAGS.batch_norm_decay),
                                 ('seed', FLAGS.seed)
                                 ])
        return test_info

    def _functional_information(self):
        func_info = OrderedDict([('enable_evaluate', FLAGS.enable_evaluate),
                                 ('enable_horovod', FLAGS.enable_horovod),
                                 ('enable_profiler', FLAGS.enable_profiler),
                                 ('enable_dump_graph', FLAGS.enable_dump_graph),
                                 ('enable_saver', FLAGS.enable_saver),
                                 ('debug_mode', FLAGS.debug_mode),
                                 ('xla_jit', FLAGS.xla_jit),
                                 ('enable_amp', FLAGS.enable_amp)
                                 ])
        return func_info


if __name__ == '__main__':
    info_dict = InfoDict()
    test_info = info_dict.test_info
    print(test_info)
