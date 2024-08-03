##########################################################################
# File Name: md5.py
# Author: tianyu.jiang
# Mail: tianyu.jiang@enflame-tech.com
# Created Time: 2020-04-16 14:05:21
#########################################################################
#!/usr/bin/python
# coding=utf-8
import hashlib
import sys

filename = sys.argv[1]

blocksize = 65536
hash = hashlib.md5()
with open(filename, 'rb') as f:
    for block in iter(lambda: f.read(blocksize), b""):
        hash.update(block)
print(hash.hexdigest())

