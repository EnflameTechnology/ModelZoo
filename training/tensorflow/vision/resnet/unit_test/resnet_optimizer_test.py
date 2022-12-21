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

from tensorflow.python.platform import test

from tops_models.tf_test.adam_optimizer_test import AdamOptimizerTest
from tops_models.tf_test.gradient_descent_optimizer_test import GradientDescentOptimizerTest
from tops_models.tf_test.lars_optimizer_test import LARSOptimizerTest
from tops_models.tf_test.momentum_optimizer_test import MomentumOptimizerTest
from tops_models.tf_test.rmsprop_optimizer_test import RMSPropOptimizerTest

if __name__ == "__main__":
  test.main()
