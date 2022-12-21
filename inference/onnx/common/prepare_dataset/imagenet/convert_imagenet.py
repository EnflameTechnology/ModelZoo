#!/usr/bin/python
#
# Copyright 2022 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys


def walk(dir, label_map, labels, start_dir, results):
    for i in os.listdir(dir):
        path = os.path.join(dir, i)
        if os.path.isfile(path):
            if i.endswith('.JPEG') or i.endswith('.jpeg'):
                # ILSVRC2012_val_00050000.JPEG -> 49999
                index = int(i[-10:-5]) - 1
                results.append('./'+os.path.relpath(path, start_dir) +
                            ' '+str(label_map[labels[index]]))
        else:
            walk(path, label_map, labels, start_dir, results)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Invalid usage\n'
              'usage: convert_imagenet.py '
              '<validation data dir> imagenet_2012_validation_synset_labels.txt '
              'imagenet_lsvrc_2015_synsets.txt <val_map.txt dir>')
        sys.exit(-1)
    data_dir = sys.argv[1]
    validation_labels_file = sys.argv[2]
    label_map_file = sys.argv[3]
    output_file = sys.argv[4]
    # Read 1000 category order file.
    label_map = {}
    for idx, label in enumerate(open(label_map_file).readlines()):
        label_map[label.strip()] = idx

    # Read in the 50000 synsets associated with the validation data set.
    labels = [l.strip() for l in open(validation_labels_file).readlines()]

    # Generate val_map.txt
    results = []
    walk(data_dir, label_map, labels, data_dir, results)
    with open(output_file, 'w') as f:
        for i in results:
            f.write(i+'\n')
