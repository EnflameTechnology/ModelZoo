import math
import os
from pathlib import Path
import random
import tarfile
import sys
from absl import app
from absl import flags
import tensorflow as tf
import ntpath
from random import shuffle
sys.path.append("../")
from tops_models.logger import tops_logger
logger = tops_logger()

flags.DEFINE_string(
    'local_scratch_dir', None, 'Scratch directory path for temporary files.')
flags.DEFINE_string(
    'raw_data_dir', None, 'Directory path for raw Imagenet dataset. '
                          'Should have train and validation subdirectories inside it.')

FLAGS = flags.FLAGS

TRAINING_SHARDS = 16
VALIDATION_SHARDS = 8

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
    """Determine if file contains a CMYK JPEG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a JPEG encoded with CMYK color space.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = set(['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                     'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                     'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                     'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                     'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                     'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                     'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                     'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                     'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                     'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                     'n07583066_647.JPEG', 'n13037406_4650.JPEG'])
    return os.path.basename(filename) in blacklist


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Clean the dirty data.
    if _is_png(filename):
        # 1 image is a PNG.
        tf.logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    elif _is_cmyk(filename):
        # 22 JPEG images are in CMYK colorspace.
        tf.logging.info('Converting CMYK to RGB for %s' % filename)
        image_data = coder.cmyk_to_rgb(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def gen_train_data_txt():
    logger.info("gen_train_data_txt enter")
    dataset_location = "{}/dataset/".format(ROOT_PATH)
    os.chdir(dataset_location)
    train_txt_file = "{}/train.txt".format(dataset_location)
    val_txt_file = "{}/val.txt".format(dataset_location)
    logger.info("{}\t{}".format(train_txt_file, val_txt_file))
    if os.path.exists(train_txt_file) and os.path.exists(val_txt_file):
        logger.info("image txt file already found.")
        return train_txt_file, val_txt_file
    train_img_dir = "{}/train".format(dataset_location)
    logger.info(train_img_dir)
    if not os.path.exists(train_img_dir):
        logger.info("train dir {} not found".format(train_img_dir))
        return train_txt_file, val_txt_file
    base_dir = train_img_dir
    lines = []
    for img in os.listdir(base_dir):
        if img.endswith(".jpg"):
            # logger.info(basename(img))
            catag_name = ntpath.basename(img).split('.')[0]
            # logger.info(catag_name)
            if sys.platform.startswith('linux'):
                img_path = "{0}/{1}".format(base_dir, img)
            else:
                img_path = "{0}\{1}".format(base_dir, img)
            if catag_name == 'cat':
                label = 0
            elif catag_name == 'dog':
                label = 1
            else:
                label = 2

            x_line = "{0} {1}".format(img_path, label)
            lines.append(x_line)
    # To do: support multi dataset
    # total = len(lines)
    total = 24576
    assert total <= len(lines)
    shuffle(lines)
    train_data = lines[0:int(total * 0.75)]
    val_data = lines[int(total * 0.75):total]

    with open('train.txt', 'w') as out_file:
        for line in train_data:
            out_file.write(line + "\n")
    with open('val.txt', 'w') as out_file:
        for line in val_data:
            out_file.write(line + "\n")
    logger.info("generate image txt file done")
    return train_txt_file, val_txt_file


def load_cat_dog_raw_data():
    ftp_url_train = "ftp://10.168.20.63/software/dataset/cat_dog/train.zip"
    ftp_url_test = "ftp://10.168.20.63/software/dataset/cat_dog/test.zip"
    local_dir = "{}/dataset".format(ROOT_PATH)
    local_train_file = "{}/{}".format(local_dir, 'train.zip')
    local_test_file = "{}/{}".format(local_dir, 'test.zip')

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    if not os.path.exists(local_train_file):
        os.system("wget -O dataset/train.zip {} ".format(ftp_url_train))
        os.system("unzip {}  -d {} ".format(local_train_file, local_dir))

    if not os.path.exists(local_test_file):
        os.system("wget -O dataset/test.zip {} ".format(ftp_url_test))
        os.system("unzip {} -d {} ".format(local_test_file, local_dir))


def convert_to_tf_records():
    """Convert the Imagenet dataset into TF-Record dumps."""
    train_index_file, val_index_file = gen_train_data_txt()

    def do_processing(index_file, shards, prefix='train'):
        img_path_files = []
        label_list = []
        total_count = 0
        with open(index_file, 'r') as in_file:
            lines = in_file.readlines()
            for line in lines:
                items = line.split(' ')
                img_path = items[0]
                label = int(items[1])
                img_path_files.append(img_path)
                label_list.append(label)
                total_count += 1
        print("==index_file={}=total===={}".format(index_file, total_count))
        file_per_shard = total_count / shards
        print("==file_per_shard==={}".format(file_per_shard))

        def make_record(img_path_files, label_list, out_file, shard):
            coder = ImageCoder()
            record_count = 0
            train_writer = tf.python_io.TFRecordWriter(out_file)
            for image_path, label in zip(img_path_files[file_per_shard * shard: file_per_shard * (shard + 1)],
                                         label_list[file_per_shard * shard: file_per_shard * (shard + 1)]):
                print(image_path, label)
                print(record_count, file_per_shard * shard, file_per_shard * (shard + 1))
                image_buffer, height, width = _process_image(image_path, coder)
                example = _convert_to_example(image_path, image_buffer, label,
                                              height, width)
                record_count += 1
                train_writer.write(example.SerializeToString())
            print("==shard={}=record_count==={}".format(shard, record_count))
            train_writer.close()

        output_directory = "{}/output".format(ROOT_PATH)
        for shard in range(shards):
            train_output_file = os.path.join(
                output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, shards))
            make_record(img_path_files, label_list, train_output_file, shard)
            print("shard={},output_file={}".format(shard, train_output_file))

    do_processing(train_index_file, TRAINING_SHARDS, 'train')
    do_processing(val_index_file, VALIDATION_SHARDS, 'validation')


def main(argv):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.INFO)

    # Download the dataset if it is not present locally
    raw_data_dir = FLAGS.raw_data_dir

    load_cat_dog_raw_data()

    logger.info("=======")
    logger.info(raw_data_dir)
    if not os.path.exists('output'):
        os.makedirs('output')

    # Convert the raw data into tf-records
    convert_to_tf_records()
    logger.info("====end====")


if __name__ == '__main__':
    app.run(main)

