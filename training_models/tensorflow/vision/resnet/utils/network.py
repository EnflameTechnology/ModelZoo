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

slim = tf.contrib.slim

from models.resnet import resnet_arg_scope, resnet_model
from models.resnet_model import resnet_cifar_model, \
    resnet_imagenet_model

format_mapping = {
    'NHWC': 'channels_last',
    'NCHW': 'channels_first',
    'CHNW': 'channels_chnw',
}


def build_model(x, params, is_training=True, weight_init=None,
                reuse=tf.AUTO_REUSE):
    # Series of resnet networks have been configurable by depth and version
    end_points = None

    assert params['resnet_version'] in (1, 1.5, 2, 3), \
        'Unsupported resnet version: {}'.format(params['resnet_version'])

    if params['resnet_version'] == 3:
        with slim.arg_scope(resnet_arg_scope(weight_init=weight_init)):
            net, end_points = resnet_model(x,
                                           depth=params['depth'],
                                           num_classes=params['num_class'],
                                           is_training=is_training,
                                           data_format=params['data_format'],
                                           use_resource=params['use_resource'],
                                           reuse=reuse)
    else:
        assert params['data_format'] in format_mapping, \
            "Unsupport data format: {}".format(params['data_format'])
        data_format = format_mapping[params['data_format']]

        if params['dataset'] == 'cifar':
            resnet_model_fn = resnet_cifar_model
        else:
            resnet_model_fn = resnet_imagenet_model
        net = resnet_model_fn(x,
                              resnet_size=params['depth'],
                              dtype=params['dtype'],
                              num_classes=params['num_class'],
                              data_format=data_format,
                              resnet_version=params['resnet_version'],
                              use_resource=params['use_resource'],
                              is_training=is_training)

    return net, end_points
