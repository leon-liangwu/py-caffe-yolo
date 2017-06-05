import os.path as sop
import sys


def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)


caffe_path = '/home/wuliang/Workspace/py-caffe-yolo/python/'
add_path(caffe_path)

lib_path = './'
add_path(lib_path)
