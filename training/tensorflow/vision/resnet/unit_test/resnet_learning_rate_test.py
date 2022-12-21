#!/usr/bin/env python
#
# Copyright 2020 Enflame. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for Optimizers used in ResNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest

from tops_models.tf_test.learning_rate_test import LRDecayTest
from tops_models.tf_test.learning_rate_test import LinearDecayTest
from tops_models.tf_test.learning_rate_test import SqrtDecayTest
from tops_models.tf_test.learning_rate_test import PolynomialDecayTest
from tops_models.tf_test.learning_rate_test import ExponentialDecayTest
from tops_models.tf_test.learning_rate_test import InverseDecayTest
from tops_models.tf_test.learning_rate_test import CosineDecayTest
from tops_models.tf_test.learning_rate_test import CosineDecayRestartsTest
from tops_models.tf_test.learning_rate_test import LinearCosineDecayTest
from tops_models.tf_test.learning_rate_test import NoisyLinearCosineDecayTest

if __name__ == "__main__":
  googletest.main()
