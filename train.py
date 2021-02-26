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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import shutil
import sys
import time
from xml.dom.minidom import parse

import numpy as np
import tensorflow as tf
from PIL import Image

from model import pix2pix

input_height, input_width = 512, 512
batch_size = 1
buffer_size = batch_size * 4
checkpoint_path = 'checkpoints/train/'
LAMBDA = 100
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def readXML(config_xml):
    domTree = parse(config_xml)
    rootNode = domTree.documentElement
    args = {}

    args['num_epochs'] = int(rootNode.getElementsByTagName(
        'numEpochs')[0].childNodes[0].data)
    args['input_channels'] = int(rootNode.getElementsByTagName(
        'inputChannels')[0].childNodes[0].data)
    args['output_channels'] = int(rootNode.getElementsByTagName(
        'outputChannels')[0].childNodes[0].data)
    args['from_tfrecord'] = rootNode.getElementsByTagName(
        'fromTFRecord')[0].childNodes[0].data == 'True'
    args['initial_learning_rate'] = float(
        rootNode.getElementsByTagName('initLR')[0].childNodes[0].data)
    args['decay_steps'] = int(rootNode.getElementsByTagName(
        'decaySteps')[0].childNodes[0].data)
    args['decay_rate'] = float(rootNode.getElementsByTagName(
        'decayRate')[0].childNodes[0].data)
    args['checkpoint_path'] = str(rootNode.getElementsByTagName(
        'checkpointPath')[0].childNodes[0].data)
    args['restore'] = rootNode.getElementsByTagName(
        'restoreCheckpoint')[0].childNodes[0].data == 'True'
    args['visualization'] = rootNode.getElementsByTagName(
        'visualization')[0].childNodes[0].data == 'True'

    print('==== Train configuration ====')
    for key, value in args.items():
        print('{} = {}'.format(key, value))
    print()

    return args


def _byte_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image2tfrecord(image_path, label_path, output_path):
    """ Convert images and labels into TFRecord """

    image_names = os.listdir(image_path)
    tfrecord_path = os.path.join(output_path, 'tfrecord/')
    if os.path.exists(tfrecord_path):
        shutil.rmtree(tfrecord_path)
    os.makedirs(tfrecord_path)

    for im_name in image_names:
        print('Converting {}...'.format(im_name))
        im_name_prefix = os.path.splitext(im_name)[0]
        try:
            image = np.asarray(Image.open(
                os.path.join(image_path, im_name)))
            label = np.asarray(Image.open(
                os.path.join(label_path, im_name)))
            writer = tf.io.TFRecordWriter(
                os.path.join(tfrecord_path, im_name_prefix + '.tfrecord'))
        except:
            print('File {} open error!'.format(im_name))
            continue

        assert image.shape == label.shape and len(image.shape) == 3
        height, width, num_channels = image.shape

        image = image.tostring()
        label = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _byte_feature(image),
            'label': _byte_feature(label),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'num_channels': _int64_feature(num_channels)
        }))

        writer.write(example.SerializeToString())
        writer.close()
        print('{} converted.'.format(im_name))


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(
            example_proto, feature_description)

        image = tf.io.decode_raw(example['image'], tf.uint8)
        label = tf.io.decode_raw(example['label'], tf.uint8)
        height, width, num_channels = example['height'], example['width'], example['num_channels']

        image = tf.reshape(image, [height, width, num_channels])
        label = tf.reshape(label, [height, width, num_channels])

        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        label = tf.cast(label, tf.float32)
        label = label / 127.5 - 1

        example['image'] = image
        example['label'] = label

        return example

    dataset = dataset.map(_parse_function).batch(batch_size)

    return dataset


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(
        disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(
        disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def train_loop(args, output_path):
    # Load train and val sets
    if not args['from_tfrecord']:
        tfrecord_path = os.path.join(output_path, 'tfrecord')
    else:
        tfrecord_path = os.path.join(input_path, 'tfrecord')
    train_names = os.listdir(tfrecord_path)
    train_names = [os.path.join(tfrecord_path, name) for name in train_names]
    train_set = load_dataset(train_names)

    # Log
    logdir = os.path.join(output_path, 'logs/')
    file_writer = tf.summary.create_file_writer(logdir + 'metrics')
    file_writer.set_as_default()

    if args['visualization']:
        import matplotlib.pyplot as plt
        vis_path = os.path.join(output_path, 'visualizations')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

    # Generator: hazy --> dehazy
    generator = pix2pix.dehaze_generator(
        input_channels=args['input_channels'], estimation_channels=args['output_channels'], norm_type='instancenorm')
    # Discriminator: real or fake
    discriminator = pix2pix.discriminator(
        input_channels=args['input_channels'], norm_type='instancenorm', target=True)

    # Optimizers
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args['initial_learning_rate'],
        decay_steps=args['decay_steps'],
        decay_rate=args['decay_rate']
    )

    generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

    # Checkpoints and manager
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, os.path.join(output_path, args['checkpoint_path']), max_to_keep=5)

    # if restoration is enabled and a checkpoint exists, restore the latest checkpoint.
    if args['restore'] and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored, from {}'.format(
            ckpt_manager.latest_checkpoint))

    def _random_crop(record):
        hazy, gt = record['image'], record['label']

        images = tf.concat([hazy, gt], axis=-1)
        images = tf.image.resize_with_pad(images, input_height + input_height // 2,
                                          input_width + input_width // 2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        images = tf.image.random_crop(
            images, [batch_size, input_height, input_width, args['input_channels'] * 2])
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)

        hazy, gt = images[..., :args['input_channels']], images[...,
                                                                args['input_channels']:]

        record['image'], record['label'] = hazy, gt

        return record

    def _est_ale(hazy):
        hazy_shape = tf.shape(hazy)
        hsv = tf.image.rgb_to_hsv(hazy * 0.5 + 0.5)
        hsv_shape = tf.shape(hsv)

        hsv = tf.reshape(hsv[..., 2], [-1])
        idx = tf.argmax(hsv)
        ale = tf.reshape(hazy, [-1, hazy_shape[-1]])[idx]
        ale = ale[tf.newaxis, tf.newaxis, tf.newaxis, ...]

        return ale

    @tf.function
    def _train_step(hazy, gt, ale):
        with tf.GradientTape(persistent=True) as tape:
            dehazy, rtme, dehazy0, tme = generator([hazy, ale], training=True)

            disc_real_output = discriminator([hazy, gt], training=True)
            disc_generated_output = discriminator(
                [hazy, dehazy], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, dehazy, gt)
            disc_loss = discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = tape.gradient(
            gen_total_loss, generator.trainable_variables)
        discriminator_gradients = tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

    with file_writer.as_default():
        step = 0
        for epoch in range(args['num_epochs']):
            start = time.time()
            train_set_ = train_set.map(_random_crop).shuffle(
                buffer_size=buffer_size)

            for record in train_set_:
                hazy, gt = record['image'], record['label']
                ale = _est_ale(hazy)
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = _train_step(
                    hazy, gt, ale)
                step += 1

                tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
                tf.summary.scalar('disc_loss', disc_loss, step=step)

                print('Step {}, gen_total_loss: {}, gen_gan_loss: {}, gen_l1_loss: {}, disc_loss: {}'.format(
                    step, gen_total_loss.numpy(), gen_gan_loss.numpy(), gen_l1_loss.numpy(), disc_loss.numpy()))

            print('Time taken for epoch {} is {} sec'.format(
                epoch + 1, time.time() - start))

            ckpt_save_path = ckpt_manager.save()
            print('Checkpoint for epoch {} saved at {}'.format(
                epoch + 1, ckpt_save_path))

            # Train visualization
            if args['visualization']:
                for record in train_set_.take(1):
                    hazy, gt = record['image'], record['label']
                    ale = _est_ale(hazy)
                    dehazy, rtme, dehazy0, tme = generator(
                        [hazy, ale], training=False)

                    hazy, gt, ale = hazy.numpy()[0], gt.numpy()[
                        0], ale.numpy()[0]
                    dehazy, rtme, dehazy0, tme = dehazy.numpy()[0], rtme.numpy()[
                        0], dehazy0.numpy()[0], tme.numpy()[0]

                    np.savez(os.path.join(
                        vis_path, 'epoch_{}.npz'.format(epoch)),
                        hazy=hazy,
                        gt=gt,
                        ale=ale,
                        dehazy=dehazy,
                        rtme=rtme,
                        dehazy0=dehazy0,
                        tme=tme)

                    hazy, gt = hazy / 2 + 0.5, gt / 2 + 0.5
                    dehazy, dehazy0 = dehazy.clip(-1, 1) / \
                        2 + 0.5, dehazy0.clip(-1, 1) / 2 + 0.5
                    ale = np.broadcast_to(ale, hazy.shape)
                    rtme, tme = np.broadcast_to(
                        rtme, hazy.shape), np.broadcast_to(tme, hazy.shape)
                    rtme = (rtme - rtme.min()) / (rtme.max() - rtme.min())
                    tme = (tme - tme.min()) / (tme.max() - tme.min())

                    imgs = [hazy, gt, ale, np.ones_like(hazy),
                            dehazy, rtme, dehazy0, tme]
                    titles = ['Hazy', 'GT', 'ALE', '',
                              'Dehazy', 'RTME', 'Hazy', 'TME']

                    row, col = 2, 4
                    plt.figure(figsize=(4 * col, 4 * row))
                    for i in range(len(imgs)):
                        plt.subplot(row, col, i + 1)
                        plt.title(titles[i])
                        plt.imshow(imgs[i])
                    plt.savefig(os.path.join(
                        vis_path, 'epoch_{}.png'.format(epoch)))
                    plt.close()

    tf.saved_model.save(generator, os.path.join(output_path, 'generator_g/1/'))


if __name__ == "__main__":
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]

        train_args = readXML(os.path.join(input_path, 'train.xml'))

        if not train_args['from_tfrecord']:
            image_path = os.path.join(input_path, 'cloud')
            label_path = os.path.join(input_path, 'label')
            image2tfrecord(image_path, label_path, output_path)

        train_loop(train_args, output_path)
    except Exception as e:
        print(e)
