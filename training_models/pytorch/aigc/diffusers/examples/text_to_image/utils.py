import os
import sys
import logging
import logging.handlers
import json
import psutil
import subprocess
from functools import wraps
import importlib.util
import torch
from diffusers.utils.import_utils import is_torch_version
if torch.cuda.is_available() and is_torch_version(">=", "2.0.0"):
    from torch import inf
else:
    from torch._six import inf
import torch



DEFAULT_FORMATTER = logging.Formatter(
            '%(asctime)s: D [M APP] [T %(process)d] %(filename)s:%(lineno)d : %(message)s')

class Loggers:
    """GPU model test logger"""
    def __init__(self, log_file=None, name="GPU_Diffuser", level=logging.INFO, formatter=DEFAULT_FORMATTER):
        # step 1: create a logger class with name
        self.__logger = logging.getLogger(name)
        self.__logger.propagate = False
        # To prevent duplicate log
        if self.__logger.handlers:
            return
        self.__logger.setLevel(level)

        # step 2: create control log handler
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)

        # step 3: Specify the log output format

        sh.setFormatter(formatter)

        # step 4: Adding a logging to handlers
        self.__logger.addHandler(sh)

        # if log file exist, add logging to file
        if log_file:
            self.log_file = os.path.abspath(log_file)
            if os.path.exists(self.log_file):
                self.__logger.warning(
                    "{} is existent, it will logging with append mode. ".format(self.log_file))
            fh = logging.handlers.RotatingFileHandler(self.log_file,
                                                      maxBytes=50 * 1024 * 1024,
                                                      backupCount=10)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.__logger.addHandler(fh)

    @property
    def logger(self):
        return self.__logger

def get_logger(log_file=None, name="GPU_Diffuser", log_level=logging.INFO, formatter=DEFAULT_FORMATTER):
    logger = Loggers(log_file, name, log_level, formatter)
    return logger.logger