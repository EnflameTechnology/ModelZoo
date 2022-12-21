#!/usr/bin/env python
#
# Copyright 2021 Enflame. All Rights Reserved.
#

from os import mkdir
from os.path import exists, join
from shutil import rmtree, move
import wget
import tarfile
import numpy as np
from absl import app
import tensorflow as tf

def download():
  if False == exists('cifar-10-python.tar.gz'):
    filename = wget.download('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', out = '.')
  else:
    filename = "cifar-10-python.tar.gz"
  if exists('cifar-10-batches-py'): rmtree('cifar-10-batches-py')
  tar = tarfile.open(filename)
  subdir_and_files = [tarinfo for tarinfo in tar.getmembers()]
  tar.extractall('.', subdir_and_files)
  return 'cifar-10-batches-py'

def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def create_dataset(dirname, filenames, output):

  if exists(output): rmtree(output)
  sess = tf.Session()
  rgb_data = tf.placeholder(dtype = tf.uint8)
  rgb_to_jpeg = tf.io.encode_jpeg(rgb_data, quality = 100)
  mkdir(output)
  shard = 0
  for filename in filenames:
    d = unpickle(join(dirname, filename))
    writer = tf.io.TFRecordWriter(join(output, '%s-%.5d-of-%.5d' % (output, shard, len(filenames))))
    imgs = d[b'data']
    labels = d[b'labels']
    for i in range(len(imgs)):
      img = np.reshape(imgs[i], (3, 32, 32))
      img = np.transpose(img, (1, 2, 0))
      label = labels[i]
      example = tf.train.Example(features = tf.train.Features(
        feature = {
          'image/height': tf.train.Feature(int64_list = tf.train.Int64List(value = [32])),
          'image/width': tf.train.Feature(int64_list = tf.train.Int64List(value = [32])),
          'image/colorspace': tf.train.Feature(bytes_list = tf.train.BytesList(value = [b'RGB'])),
          'image/channels': tf.train.Feature(int64_list = tf.train.Int64List(value = [3])),
          'image/class/label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
          'image/format': tf.train.Feature(bytes_list = tf.train.BytesList(value = [b'JPEG'])),
          'image/encoded': tf.train.Feature(bytes_list = tf.train.BytesList(value = [sess.run(rgb_to_jpeg, feed_dict = {rgb_data: img})]))
        }
      ))
      writer.write(example.SerializeToString())
    writer.close()
    shard += 1

def main(argv):

  trainset = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
  testset = ['test_batch']
  dirname = download()
  if exists('output'): rmtree('output')
  create_dataset(dirname, trainset, 'train')
  create_dataset(dirname, testset, 'evaluate')
  move('train', join('output', 'train'))
  move('evaluate', join('output', 'evaluate'))

if __name__ == "__main__":

  app.run(main)
