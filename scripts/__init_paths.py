import os.path as sop
import sys


def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)
    # print(sys.path)


caffe_path = '../../caffe/python/'
add_path(caffe_path)

lib_path = '../lib/'
add_path(lib_path)
