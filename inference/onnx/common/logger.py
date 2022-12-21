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

import os
import sys
import logging
import logging.handlers
from collections import OrderedDict
import json
import time
from .utils import get_environment_info


class TopsLogger(object):
    """tops model test logger"""

    def __init__(self, log_file=None):
        # step 1: create a logger class with name
        self.__logger = logging.getLogger("TopsModel")
        self.__logger.setLevel(logging.INFO)

        # step 2: create control log hander
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)

        # step 3: Specify the log output format
        formatter = logging.Formatter(
            '%(asctime)s: D [M APP] [T %(process)d] %(filename)s:%(lineno)d : %(message)s')
        sh.setFormatter(formatter)

        # step 4: Adding a logging to handlers
        self.__logger.addHandler(sh)

        # if log file exist, add logging to file
        if(log_file):
            self.log_file = os.path.abspath(log_file)
            if os.path.exists(self.log_file):
                self.__logger.warning(
                    "{} is existent, it will logging with append mode. ".format(self.log_file))
            fh = logging.handlers.RotatingFileHandler(self.log_file,
                                                      maxBytes=50 * 1024 * 1024,
                                                      backupCount=10)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.__logger.addHandler(fh)

    @property
    def logger(self):
        return self.__logger

    def setLevel(self, level=logging.INFO):
        assert isinstance(
            level, int), "level should be one of [logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.FATAL]"
        self.__logger.setLevel(level)


def tops_logger(log_file=None):
    return TopsLogger(log_file).logger


def final_report(logger, dict=OrderedDict(), write_report=True, file=None):
    report_dict = get_environment_info()
    report_dict.update(dict)
    json_report = json.dumps(report_dict, indent=4, sort_keys=False)
    logger.info("Final report:\n{}".format(json_report))
    if not write_report:
        return
    try:
        if file is None:
            file = './report/{}.json'.format(
                time.strftime("%Y%m%d%H%M", time.localtime()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            f.write(json_report)
    except Exception:
        logger.warning('Write final report to {} failed.'.format(file))
