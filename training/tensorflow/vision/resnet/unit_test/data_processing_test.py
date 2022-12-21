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
import sys, os

realpath = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(realpath)
from data_utils.data_processing import DataProcessing

class DataProcessingTest(tf.test.TestCase):
    def __init__(self, methodName):
        super(DataProcessingTest, self).__init__(methodName)
        self.test_params = {'debug_mode': False,
                            'output_size': 224,
                            'batch_size': 1,
                            'num_channels': 3,
                            'num_class': 2,
                            'hvd_size': 1,
                            'rank': 0}
        self.data_dir = os.path.realpath(os.path.dirname(os.path.dirname(__file__))) + '/dataset/imagenet2/train'
        self.dtype = tf.float32

    def testWholeProcessing(self):
        dataset = DataProcessing(is_training=True,
                                 data_dir=self.data_dir,
                                 dtype=self.dtype,
                                 params=self.test_params)
        train_data = dataset.input_fn(batch_size=self.test_params['batch_size'])
        iterator_train = train_data.make_initializable_iterator()
        (image_train, label_train) = iterator_train.get_next()
        with self.session() as sess:
            sess.run(iterator_train.initializer)
            for i in range(1):
                img, label = sess.run([image_train, label_train])

    def testDebugMode(self):
        self.test_params['debug_mode'] = True
        dataset = DataProcessing(is_training=True,
                                 data_dir=self.data_dir,
                                 dtype=self.dtype,
                                 params=self.test_params)
        train_data = dataset.input_fn(batch_size=self.test_params['batch_size'])
        iterator_train = train_data.make_initializable_iterator()
        (image_train, label_train) = iterator_train.get_next()
        with self.session() as sess:
            sess.run(iterator_train.initializer)
            img_1, label_1 = sess.run([image_train, label_train])
            sess.run(iterator_train.initializer)
            img_2, label_2 = sess.run([image_train, label_train])
            self.assertAllEqual(img_1, img_2)
            self.assertAllEqual(label_1, label_2)


if __name__ == "__main__":
    tf.test.main()
