#!/usr/bin/env python
#
# Copyright 2018-2022 Enflame. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import sys
import json
import numpy as np
import signal
import os
from collections import OrderedDict
from datetime import datetime
from tensorflow.core.framework import summary_pb2
import tops_models.common_utils as common_utils
import tops_models.tf_utils as tf_utils
from tops_models.logger import tops_logger
from tops_models.time_watcher import EnflameThread
from tops_models.datasets import dataset_mapping
from tops_models.internal_utils import HttpUtil
from utils.flags import InfoDict
import classifier_main

FLAGS = tf.app.flags.FLAGS

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

REPORT_DIR = "{}/test_report".format(ROOT_PATH)
tf.io.gfile.makedirs(REPORT_DIR)
CKPT_DIR = "{}/models/checkpoints".format(ROOT_PATH)
tf.io.gfile.makedirs(CKPT_DIR)
EVENT_DIR = "{}/models/tensorboard".format(ROOT_PATH)
tf.io.gfile.makedirs(EVENT_DIR)

def quit_gracefully(*args):
    print('caught signal ,quit')
    exit(0)


def main(_):
    # global constant
    NUM_CHANNEL = 3
    OUTPUT_SIZE = 224 if FLAGS.dataset != 'cifar10' else 32
    signal.signal(signal.SIGINT, quit_gracefully)

    # collect information
    info_dict = InfoDict()
    params = info_dict.test_info
    params.update(info_dict.func_info)
    params.update(common_utils.tops_models_collection(params['device']))
    params['tensorflow'] = "%i.%i.%s" % (tf_utils.get_tf_version())
    params['output_size'] = OUTPUT_SIZE
    params['num_channels'] = NUM_CHANNEL

    if params['debug_mode']:
        params['display_step'] = 1
        params['num_between_eval'] = 1
        params['num_between_saver'] = 1
        inter_op_parallelism_threads = 1
        intra_op_parallelism_threads = 1
        seed = 1234
        random.seed(seed)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        np.set_printoptions(threshold=int(sys.maxsize))
        logger = tops_logger('{}/debug.log'.format(ROOT_PATH))
    else:
        if params['seed']:
            np.random.seed(params['seed'])
            random.seed(params['seed'])
            tf.set_random_seed(params['seed'])
        inter_op_parallelism_threads = 0
        intra_op_parallelism_threads = 0
        logger = tops_logger()

    global_watcher = EnflameThread().watcher(
        batch_size_per_step=params['batch_size'] * params['num_cluster'],
        logger=logger, skip_steps=params['skip_steps'])

    if params['enable_horovod']:
        global_watcher.watch("import horovod start")
        try:
            """Try to import and initialize horovod."""
            import horovod.tensorflow as hvd
            hvd.init()
            hvd_info = OrderedDict([('hvd_size', hvd.size()),
                                    ('local_rank', hvd.local_rank()),
                                    ('rank', hvd.rank())
                                    ])
            params.update(hvd_info)
        except ImportError as ex:
            logger.error(
                "Enable_horovod True but horovod package import or init failed!"
                " Error message: {}".format(ex))
        global_watcher.watch("import horovod end")

    logger.info("Base report:\n{}".format(
        json.dumps(params, indent=4, sort_keys=False)))

    params['start_time'] = datetime.now()

    if params['data_dir']:
        dataset_location = params['data_dir']
    else:
        dataset_location = "{}/dataset/{}".format(ROOT_PATH, params['dataset'])

    # global check
    global_watcher.watch("global check start")
    assert params[
               'device'] in common_utils.device_mapping, 'Unsupported device: {}'.format(
        params['device'])
    assert params[
               'dtype'] in tf_utils.str2tf, 'Unsupported data dtype: {}'.format(
        params['dtype'])
    assert params[
               'dataset'] in dataset_mapping, 'Unsupported dataset: {}'.format(
        params['dataset'])
    params['num_class'] = dataset_mapping[params['dataset']]['class_number']
    global_watcher.watch("global check end")

    def create_config():
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True,
                                inter_op_parallelism_threads=inter_op_parallelism_threads,
                                intra_op_parallelism_threads=intra_op_parallelism_threads)
        if params['device'] in ['gpu', 'xla_gpu']:
            if params['xla_jit']:
                config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            else:
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.90
            config.gpu_options.visible_device_list = str(params['local_rank'])
        elif params['device'] in ['dtu', 'xla_dtu']:
            from tensorflow.core.protobuf import rewriter_config_pb2
            off = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.memory_optimization = off
            config.dtu_options.visible_device_list = str(params['local_rank'])
        return config

    if params['enable_profiler']:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        step_run_metadata = tf.RunMetadata()
    else:
        options = None
        step_run_metadata = None

    bench = classifier_main.Benchmark(params=params, logger=logger)
    global_step = tf.train.get_or_create_global_step()

    # data processing
    global_watcher.watch("data processing start")
    with tf.device(bench.cpu_device):
        if params['is_training']:
            train_data_dir = "{}/train".format(dataset_location)
            iterator_train = bench._generate_data(data_dir=train_data_dir,
                                                  is_training=True)
            iterator_initial_op = iterator_train.initializer
        else:
            infer_data_dir = '{}/test'.format(dataset_location)
            iterator_infer = bench._generate_data(data_dir=infer_data_dir,
                                                  is_training=False)
            iterator_initial_op = iterator_infer.initializer
        if params['enable_evaluate']:
            eval_data_dir = "{}/evaluate".format(dataset_location)
            iterator_eval = bench._generate_data(data_dir=eval_data_dir,
                                                 is_training=False)
            iterator_initial_op = tf.group(iterator_eval.initializer,
                                           iterator_initial_op)
    global_watcher.watch("data processing end")

    # build graph
    global_watcher.watch("build graph start")
    if params['xla_jit']:
        with tf.contrib.compiler.jit.experimental_jit_scope():
            if params['is_training']:
                train_op, accuracy_op, loss_op = bench.build_training_graph(
                    iterator=iterator_train,
                    global_step=global_step)
            else:
                predict_op = bench.build_inference_graph(
                    iterator=iterator_infer)
            if params['enable_evaluate']:
                eval_accuracy_op, eval_accuracy_top_5_op = bench.build_evaluate_graph(
                    iterator=iterator_eval)
    else:
        with tf.device(bench.default_device):
            if params['is_training']:
                train_op, accuracy_op, loss_op = bench.build_training_graph(
                    iterator=iterator_train,
                    global_step=global_step)
                tf.summary.scalar('loss', loss_op)
            else:
                predict_op = bench.build_inference_graph(
                    iterator=iterator_infer)
            if params['enable_evaluate']:
                eval_accuracy_op, eval_accuracy_top_5_op = bench.build_evaluate_graph(
                    iterator=iterator_eval)
    global_watcher.watch("build graph end")

    merged_summary = tf.summary.merge_all()

    epoch_tensor = tf.Variable(1, name='epoch', trainable=False)
    # session process
    with tf.device(bench.default_device):
        with tf.Session(config=create_config()) as sess:
            if FLAGS.enable_dump_graph:
                graph_path = "{}/resnet_before_init_{}.pbtxt".format(CKPT_DIR,
                                                                     datetime.now().strftime(
                                                                         '%Y%m%d%H%M%S.%f')[:-3])
                tf.io.write_graph(
                    graph_or_graph_def=tf.get_default_graph().as_graph_def(
                        add_shapes=True),
                    logdir='{}/'.format(ROOT_PATH),
                    name=graph_path)

            # initial
            global_watcher.watch("variable init start")
            sess.run(bench.variables_initial_op())
            sess.run(iterator_initial_op)
            global_watcher.watch("variable init end")

            if params['enable_dump_graph']:
                graph_path = "{}/resnet_after_init_{}.pbtxt".format(CKPT_DIR,
                                                                    datetime.now().strftime(
                                                                        '%Y%m%d%H%M%S.%f')[:-3])
                tf.io.write_graph(
                    graph_or_graph_def=tf.get_default_graph().as_graph_def(
                        add_shapes=True),
                    logdir='./',
                    name=graph_path)

            # create summary writer
            if params['is_training']:
                writer = tf.summary.FileWriter(
                    logdir=EVENT_DIR + '/train-{}'.format(
                        bench.params['local_rank']),
                    graph=sess.graph,
                    filename_suffix='.train')
            else:
                writer = tf.summary.FileWriter(
                    logdir=EVENT_DIR + '/test-{}'.format(
                        bench.params['local_rank']),
                    graph=sess.graph,
                    filename_suffix='.test')
            writer.add_graph(sess.graph)
            if params['enable_evaluate']:
                eval_writer = tf.summary.FileWriter(
                    logdir=EVENT_DIR + '/eval-{}'.format(
                        bench.params['local_rank']),
                    graph=sess.graph,
                    filename_suffix='.eval')

            start_epoch = 1
            if params['enable_saver'] or not params['is_training']:
                saver = tf.train.Saver(max_to_keep=100, save_relative_paths=True)
                ckpt = tf.train.latest_checkpoint(CKPT_DIR,
                                                  latest_filename="checkpoint_resnet{}_v{}_{}".format(
                                                      params['depth'],
                                                      params['resnet_version'],
                                                      FLAGS.dataset))
                if ckpt is not None:
                    saver.restore(sess, ckpt)
                    logger.info("restore checkpoint from {}".format(
                        ckpt))
                    try:
                        start_epoch = sess.run(epoch_tensor) + 1
                    except:
                        start_epoch = 1
                    if params['is_training'] and start_epoch == params['epoch'] + 1:
                        logger.info("model already finished training.")
                        return

                    if not params['is_training']:
                        start_epoch = 1

            # horovod broadcast
            if params['enable_horovod']:
                global_watcher.watch("horovod broadcast start")
                sess.run(hvd.broadcast_global_variables(0))
                global_watcher.watch("horovod broadcast end")

            for epoch in range(start_epoch, params['epoch'] + 1 ):
                if params['is_training']:
                    params['training_step_per_epoch'] = bench.training_step_num_per_epoch
                    for local_step in range(1,
                                            bench.training_step_num_per_epoch + 1):
                        global_watcher.watch(
                            "epoch:{}, local_step:{}, session_start".format(
                                epoch, local_step))
                        _, loss = sess.run([train_op, loss_op],
                                           options=options,
                                           run_metadata=step_run_metadata)
                        global_watcher.watch(
                            "epoch:{}, local_step:{}, session_end".format(
                                epoch, local_step))
                        if local_step % FLAGS.display_step == 0:
                            accuracy, s, step_count = sess.run(
                                [accuracy_op, merged_summary, global_step])
                            logger.info(
                                "global_step: {}; local_step: {}; loss: {}; accuracy: {}".format(
                                    step_count,
                                    local_step,
                                    loss,
                                    accuracy))
                            writer.add_summary(s, step_count)
                            assert not common_utils.check_valid_data(loss), \
                                'Found valid data in loss, NAN or INF'
                            assert not common_utils.check_valid_data(
                                accuracy[1]), \
                                'Found valid data in accuracy, NAN or INF'
                        if params['enable_profiler']:
                            run_metadata.MergeFrom(step_run_metadata)

                    if epoch % params['num_between_saver'] == 0 and \
                            params['enable_saver'] and \
                            params['local_rank'] == 0:
                        if params['resnet_version'] == 3:
                            output_name = ['resnet_v2_50/SpatialSqueeze']
                        else:
                            output_name = ['resnet_model/final_dense']
                        _ = tf_utils.freeze_graph(sess=sess,
                                                  output_name=output_name,
                                                  output_pb='{}/resnet{}_v{}_{}.pb'.format(
                                                      CKPT_DIR,
                                                      params['depth'],
                                                      params['resnet_version'],
                                                      params['dataset']))
                        epoch_tensor.assign(epoch).eval()
                        saver.save(sess=sess,
                                   save_path="{}/enflame_resnet{}_v{}_{}".format(
                                       CKPT_DIR,
                                       params['depth'],
                                       params['resnet_version'],
                                       params['dataset']),
                                   global_step=epoch_tensor,
                                   latest_filename="checkpoint_resnet{}_v{}_{}".format(
                                       params['depth'],
                                       params['resnet_version'],
                                       params['dataset']))
                else:
                    global_watcher.watch(
                        "epoch:{}, session_start".format(epoch))
                    predict, s = sess.run([predict_op, merged_summary],
                                          options=options,
                                          run_metadata=step_run_metadata)
                    global_watcher.watch("epoch:{}, session end".format(epoch))
                    logger.info(
                        "predict class: {}, real class: {}; accuracy: {}".format(
                            predict['predict class'],
                            predict['real class'],
                            common_utils.calc_inference_accuracy(
                                predict['predict class'],
                                predict['real class'])))
                    # logger.info("probability: {}".format(predict['prob']))
                    writer.add_summary(s, epoch)
                    if params['enable_profiler']:
                        run_metadata.MergeFrom(step_run_metadata)

                if params['enable_evaluate'] and \
                        (epoch % params['num_between_eval'] == 0 or epoch == 1):
                    params['evaluate_step_per_epoch'] = bench.evaluate_step_num_per_epoch
                    tower_eval_accuracy = 0
                    tower_eval_accuracy_top_5 = 0
                    for local_step in range(bench.evaluate_step_num_per_epoch):
                        eval_accuracy, eval_accuracy_top_5, step_count = sess.run(
                            [eval_accuracy_op, eval_accuracy_top_5_op,
                             global_step])
                        logger.info(
                            "eval_accuracy: {}; eval_accuracy_top_5: {}".format(
                                eval_accuracy,
                                eval_accuracy_top_5))
                        tower_eval_accuracy += eval_accuracy
                        tower_eval_accuracy_top_5 += eval_accuracy_top_5
                    avg_eval_accuracy = tower_eval_accuracy / bench.evaluate_step_num_per_epoch
                    avg_eval_accuracy_top_5 = tower_eval_accuracy_top_5 / bench.evaluate_step_num_per_epoch
                    logger.info(
                        "avg_eval_accuracy: {}; avg_eval_accuracy_top_5: {}".format(
                            avg_eval_accuracy,
                            avg_eval_accuracy_top_5))
                    s = summary_pb2.Summary(
                        value=[summary_pb2.Summary.Value(tag="eval_accuracy",
                                                         simple_value=avg_eval_accuracy)])
                    s_5 = summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(tag="eval_accuracy_top_5",
                                                      simple_value=avg_eval_accuracy_top_5)])
                    eval_writer.add_summary(s, step_count)
                    eval_writer.add_summary(s_5, step_count)

    with tf.device(bench.cpu_device):
        # profile
        if params['enable_profiler']:
            global_watcher.watch("Profile start")
            profile_log = "{}/{}_profile.log".format(REPORT_DIR,
                                                     params['device'])
            timeline_json = "{}/{}_timeline.json".format(REPORT_DIR,
                                                         params['device'])
            tf_utils.do_profile(graph=tf.get_default_graph(),
                                run_metadata=run_metadata,
                                profile_log=profile_log,
                                timeline_json=timeline_json)
            global_watcher.watch("Profile end")

        # update all information
        runtime_info = OrderedDict(
            [('fps', global_watcher.calc_fps()),
             ('min_fps', global_watcher.min_fps()),
             ('max_fps', global_watcher.max_fps()),
             ('session_duration', sum(global_watcher.duration_time_list)),
             ('total_duration',
              (datetime.now() - params['start_time']).total_seconds()),
             ('start_time', params['start_time'].strftime("%Y-%m-%d %H:%M:%S"))])
        params.update(runtime_info)
        json_report = json.dumps(params, indent=4, sort_keys=False)
        logger.info("Final report:\n{}".format(json_report))

        # post json report to web server
        if params['local_rank'] == 0:
            global_watcher.dump_json()
            HttpUtil().post_json(json_report, logger)


if __name__ == '__main__':
    tf.app.run()
