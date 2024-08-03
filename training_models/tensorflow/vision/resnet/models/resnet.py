from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16, tf.bfloat16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def _custom_dtype_getter(getter, name, shape=None, dtype=DEFAULT_DTYPE,
                         *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
        var = getter(name, shape, tf.float32, *args, **kwargs)
        return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
        return getter(name, shape, dtype, *args, **kwargs)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, data_format, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.
    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], data_format=data_format,
                               stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1,
                data_format='NHWC', scope=None):
    """Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    Note that
       net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)
    whereas
       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           rate=rate,
                           biases_initializer=None, padding='SAME',
                           data_format=data_format, scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == "CHNW":
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [0, 0],
                             [pad_beg, pad_end]])
        else:
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end],
                             [0, 0]])

        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           biases_initializer=None,
                           rate=rate, padding='VALID', data_format=data_format,
                           scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None, data_format='NHWC'):
    """Stacks ResNet `Blocks` and controls output feature density.
    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.
    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.
    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This results
    to a nominal ResNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.
    Control of the output feature density is implemented by atrous convolution.
    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      output_stride: If `None`, then the output will be computed at the nominal
        network stride. If output_stride is not `None`, it specifies the requested
        ratio of input to output spatial resolution, which needs to be equal to
        the product of unit strides from the start up to some level of the ResNet.
        For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
        then valid values for the output_stride are 1, 2, 6, 24 or None (which
        is equivalent to output_stride=24).
      outputs_collections: Collection to add the ResNet block outputs.
    Returns:
      net: Output tensor with stride equal to the specified output_stride.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError(
                        'The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate,
                                            **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name,
                                                   net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True,
                     weight_init=None):
    """Defines the default ResNet arg scope.
    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      activation_fn: The activation function which is used in ResNet.
      use_batch_norm: Whether or not to use batch normalization.
    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        #        'zero_debias_moving_mean': True,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    if weight_init is None:
        with tf.device('/cpu:0'):
            weight_init = tf.contrib.layers.variance_scaling_initializer(
                dtype=tf.float32)

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=weight_init,
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               data_format='NHWC',
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.
    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # HACK here: change last_dimension to be channel dimension, but it doesn't work for CHNW
        #            Hack to use _get_dimension(0) here
        if data_format == 'CHNW':
            depth_in = slim.utils._get_dimension(inputs.get_shape(), 0,
                                                 min_rank=4)
        else:
            # NHWC
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        print("depth in:", depth_in)
        print("depth:", depth)
        # param_initializers = {
        #     "beta": tf.zeros_initializer(dtype=tf.bfloat16),
        #     "gamma": tf.random_normal_initializer(mean=1, stddev=0.045)
        # }

        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu,
                                 scope='preact',
                                 data_format=data_format)
        print("preact:", preact)
        tf.add_to_collection('dump', preact)
        if depth == depth_in:
            print("down sample")
            print("stride:", stride)
            shortcut = subsample(inputs, stride, data_format=data_format,
                                 scope='shortcut')
            tf.add_to_collection('dump', shortcut)
        else:
            print("stride:", stride)
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   biases_initializer=None,
                                   normalizer_fn=None, activation_fn=None,
                                   data_format=data_format,
                                   scope='shortcut')
        print("shortcut: ", shortcut)
        tf.add_to_collection('dump', shortcut)

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               data_format=data_format,
                               biases_initializer=None, scope='conv1')
        print("residual: ", residual)
        tf.add_to_collection('dump', residual)
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               rate=rate, data_format=data_format,
                               scope='conv2')
        print("residual: ", residual)
        tf.add_to_collection('dump', residual)
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               biases_initializer=None,
                               normalizer_fn=None, activation_fn=None,
                               data_format=data_format,
                               scope='conv3')
        print("residual: ", residual)
        tf.add_to_collection('dump', residual)

        output = shortcut + residual
        print("output: ", output)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              data_format='NHWC',
              use_resource=False,
              scope=None):
    """Generator for v2 (preactivation) ResNet models.
    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.
    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.
    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether is training or not.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          To use this parameter, the input images must be smaller than 300x300
          pixels, in which case the output logit layer does not contain spatial
          information and can be removed.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse,
                           use_resource=use_resource,
                           custom_getter=_custom_dtype_getter) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection,
                            data_format=data_format):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if data_format == "CHNW":
                    net = tf.transpose(inputs, [3, 1, 0, 2])
                tf.add_to_collection('dump', net)
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError(
                                'The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # We do not include batch normalization or activation functions in
                    # conv1 because the first ResNet unit will perform these. Cf.
                    # Appendix of [2].
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        # first conv2d, channel=64, kenrel 7*7, stride=2
                        net = conv2d_same(net, 64, 7, stride=2,
                                          data_format=data_format,
                                          scope='conv1')
                        tf.add_to_collection('dump', net)
                    net = slim.max_pool2d(net, [3, 3], stride=2,
                                          data_format=data_format,
                                          scope='pool1')
                    tf.add_to_collection('dump', net)

                net = stack_blocks_dense(net, blocks, output_stride,
                                         data_format=data_format)
                tf.add_to_collection('dump', net)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(net, activation_fn=tf.nn.relu,
                                      data_format=data_format,
                                      scope='postnorm')
                tf.add_to_collection('dump', net)
                if global_pool:
                    # Global average pooling.
                    if data_format == "CHNW":
                        net = tf.reduce_mean(net, [1, 3], name='pool5',
                                             keep_dims=True)
                    else:
                        net = tf.reduce_mean(net, [1, 2], name='pool5',
                                             keep_dims=True)
                    tf.add_to_collection('dump', net)
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      data_format=data_format, scope='logits')
                    tf.add_to_collection('dump', net)
                    if data_format == "CHNW":
                        net = tf.transpose(net, [2, 1, 3, 0])
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        tf.add_to_collection('dump', net)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net,
                                                             scope='predictions')
                return net, end_points


resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v2 bottleneck block.
    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
    Returns:
      A resnet_v2 bottleneck block.
    """
    return Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


resnet_v2.default_image_size = 224


def resnet_model(inputs,
                 depth=50,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 data_format='NHWC',
                 use_resource=False):
    """ResNet model of [1]. See resnet_v2() for arg and return description."""
    scope = 'resnet_v2_{}'.format(depth)
    choices = {
        14: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]}
    num_units = choices[depth]
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=num_units[0],
                        stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=num_units[1],
                        stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=num_units[2],
                        stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=num_units[3],
                        stride=1),
    ]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, data_format=data_format,
                     use_resource=use_resource, scope=scope)


resnet_model.default_image_size = resnet_v2.default_image_size
