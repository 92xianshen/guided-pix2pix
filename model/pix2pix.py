"""
MIT License

Copyright (c) 2021 Libin Jiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
    """ Instance Normalization Layer (https://arxiv.org/abs/1607.08022). """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', 
            shape=input_shape[-1:], 
            initializer=tf.random_normal_initializer(1., 0.02), 
            trainable=True
        )

        self.offset = self.add_weight(
            name='offset', 
            shape=input_shape[-1:], 
            initializer='zeros', 
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class GuidedFilter(tf.keras.layers.Layer):
    """ A layer implementing guided filter """
    
    def __init__(self):
        super(GuidedFilter, self).__init__()

    def call(self, I, p, r, eps=1e-8):
        def diff_x(inputs, r):
            assert inputs.shape.ndims == 4

            left    = inputs[:,         r:2 * r + 1]
            middle  = inputs[:, 2 * r + 1:         ] - inputs[:,           :-2 * r - 1]
            right   = inputs[:,        -1:         ] - inputs[:, -2 * r - 1:    -r - 1]

            outputs = tf.concat([left, middle, right], axis=1)

            return outputs

        def diff_y(inputs, r):
            assert inputs.shape.ndims == 4

            left    = inputs[:, :,         r:2 * r + 1]
            middle  = inputs[:, :, 2 * r + 1:         ] - inputs[:, :,           :-2 * r - 1]
            right   = inputs[:, :,        -1:         ] - inputs[:, :, -2 * r - 1:    -r - 1]

            outputs = tf.concat([left, middle, right], axis=2)

            return outputs

        def box_filter(x, r):
            assert x.shape.ndims == 4

            return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=1), r), axis=2), r)

        assert I.shape.ndims == 4 and p.shape.ndims == 4

        I_shape = tf.shape(I)
        p_shape = tf.shape(p)

        # N
        N = box_filter(tf.ones((1, I_shape[1], I_shape[2], 1), dtype=I.dtype), r)

        # mean_x
        mean_I = box_filter(I, r) / N
        # mean_y
        mean_p = box_filter(p, r) / N
        # cov_xy
        cov_Ip = box_filter(I * p, r) / N - mean_I * mean_p
        # var_x
        var_I = box_filter(I * I, r) / N - mean_I * mean_I

        # A
        A = cov_Ip / (var_I + eps)
        # b
        b = mean_p - A * mean_I

        mean_A = box_filter(A, r) / N
        mean_b = box_filter(b, r) / N

        q = mean_A * I + mean_b

        return q

def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """ Downsamples an input.

    Conv2D => Batchnorm => LeakyReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer

    Returns:
        Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same', 
            kernel_initializer=initializer, use_bias=False
        )
    )
    
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """ Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => ReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'
        apply_dropout: If True, adds the dropout layer

    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, 
            padding='same', kernel_initializer=initializer, use_bias=False
        )
    )

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())

    return result

def dehaze_generator(input_channels=3, estimation_channels=1, norm_type='batchnorm'):
    """ Create a generator for dehazing. The global atmospheric light is given.

    Args:
        input_channels: Number of input channels
        estimation_channels: Number of output channels of UNet generator
        norm_type: Normalization type, either 'batchnorm' or 'instancenorm'
        num_or_size_splits: Split the estimation of UNet generator

    Returns:
        Callable Keras model for dehazing
    """

    # input size is 512 x 512
    down_stack = [
        downsample(64, 4, norm_type, apply_norm=False), # (bs, 256, 256, 64)
        downsample(128, 4, norm_type), # (bs, 128, 128, 128)
        downsample(256, 4, norm_type), # (bs, 64, 64, 256)
        downsample(512, 4, norm_type), # (bs, 32, 32, 512)
        downsample(512, 4, norm_type), # (bs, 16, 16, 512)
        downsample(512, 4, norm_type), # (bs, 8, 8, 512)
        downsample(512, 4, norm_type), # (bs, 4, 4, 512)
        downsample(512, 4, norm_type), # (bs, 2, 2, 512)
        downsample(512, 4, norm_type), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4, norm_type), # (bs, 16, 16, 1024)
        upsample(512, 4, norm_type), # (bs, 32, 32, 1024)
        upsample(256, 4, norm_type), # (bs, 64, 64, 512)
        upsample(128, 4, norm_type), # (bs, 128, 128, 256)
        upsample(64, 4, norm_type), # (bs, 256, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        estimation_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer, 
        activation='sigmoid'
    ) # (bs, 512, 512, estimation_channels)

    concat = tf.keras.layers.Concatenate()

    hazy = tf.keras.layers.Input(shape=[None, None, input_channels])
    atmospheric_light = tf.keras.layers.Input(shape=[None, None, input_channels])
    # lap = tf.keras.layers.Input(shape=[None, None, input_channels])
    
    # 0) 20200513 Online laplacian sharpening
    # 01) Define 4D laplacian kernel
    # kernel = np.array([
    #     [0, 1, 0], 
    #     [1, -4, 1], 
    #     [0, 1, 0]
    # ])[..., np.newaxis, np.newaxis]
    # # 02) convert rgb of hazy to gray
    # gray = tf.image.rgb_to_grayscale(hazy)
    # lap = tf.nn.conv2d(gray, kernel, strides=[1, 1, 1, 1], padding='SAME', name='Laplacian')
    # lap_add = hazy - lap

    # 1) laplacian_add as input
    x = hazy

    # Downsample through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])

    # Upsample and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    # Dehaze using equation 
    transmission_map = x
    # Refine TME
    guided_filter = GuidedFilter()
    # 20200514 replace guide with the gray of lap_add
    # guide = tf.image.rgb_to_grayscale(lap_add)
    # 2020-06-05 reuse gray of hazy as the guide
    guide = tf.image.rgb_to_grayscale(hazy)
    refined_transmission_map = guided_filter(guide, transmission_map, r=60, eps=0.0001)

    dehazy0 = (hazy - atmospheric_light) / transmission_map + atmospheric_light
    dehazy = (hazy - atmospheric_light) / refined_transmission_map + atmospheric_light

    return tf.keras.Model(inputs=[hazy, atmospheric_light], outputs=[dehazy, refined_transmission_map, dehazy0, transmission_map])

def discriminator(input_channels, norm_type='batchnorm', target=True):
    """ PatchGAN discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
        input_channels: Input channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        target: Bool, indicating whether target image is an input or not.

    Returns:
        Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, input_channels], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, input_channels], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 512, 512, channels * 2)

    down1 = downsample(64, 4, norm_type, False)(x) # (bs, 256, 256, 64)
    down2 = downsample(128, 4, norm_type)(down1) # (bs, 128, 128, 128)
    down3 = downsample(256, 4, norm_type)(down2) # (bs, 64, 64, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 66, 66, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, 
        use_bias=False
    )(zero_pad1) # (bs, 63, 63, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 64, 64, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, 
        kernel_initializer=initializer
    )(zero_pad2) # (bs, 61, 61, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)

