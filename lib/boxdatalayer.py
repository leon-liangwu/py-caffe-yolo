import __init_paths
import caffe
import yaml
import numpy as np
from numpy import *
import cv2
import config
import random

class BoxDataLayer(caffe.Layer):

  def setup(self, bottom, top):
    cfg = yaml.load(self.param_str)
    print('config in yaml ', cfg)
    self.image_list = open(cfg['image_list'])
    self.batch_size = cfg['batch_size']
    self.side = cfg['side']
    self.image_size = cfg['image_size']
    self.width = self.image_size
    self.height = self.image_size
    self.version = cfg['version']
    self.lines = self.image_list.readlines() 
    self.data_num = len(self.lines)
    top[0].reshape(self.batch_size, 3, self.height, self.width)
    if self.version == 1:
      top[1].reshape(self.batch_size, self.side * self.side*(1+1+4))
    else:
      self.version == 2
      top[1].reshape(self.batch_size, 5*30)

    self.name_to_top_map = { 'data': 0, 'label': 1}


  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    blobs = self.get_next_minibatch()

    for blob_name, blob in blobs.iteritems():
      top_ind = self.name_to_top_map[blob_name]
      top[top_ind].data[...] = blob
      #print top[top_ind].data.shape

    #print top[0].data

  def backward(self, top, propagate_down, bottom):
    pass


  def im_list_to_blob(self, ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

  def constrain(self, min_val, max_val, a):
    if a < min_val:
      return min_val
    if a > max_val:
      return max_val
    return a

  def random_crop(self, image, box):
    ori_labels = box[:]
    box_labels = []
    (img_height, img_width, img_channel) = image.shape
    mirror = random.randint(0,1)

    rand_scale_w = 1 - random.random() * 0.3
    rand_scale_h = 1 - random.random() * 0.3
    rand_w = int(img_width * rand_scale_w) - 1
    rand_h = int(img_height * rand_scale_h) - 1
    rand_x = random.randint(0, img_width - rand_w)
    rand_y = random.randint(0, img_height - rand_h)
    rand_left = rand_x
    rand_top = rand_y

    for i in range( len(ori_labels)/5 ):
      box_index = i * 5
      ori_x = int(ori_labels[i+1] * img_width)
      ori_y = int(ori_labels[i+2] * img_height)
      ori_w = int(ori_labels[i+3] * img_width)
      ori_h = int(ori_labels[i+4] * img_height)

      ori_left = ori_x - ori_w / 2
      ori_top = ori_y - ori_h / 2
      ori_right = ori_x + ori_w / 2
      ori_bottom = ori_y + ori_h / 2

      new_left = ori_left - rand_left;
      new_top = ori_top - rand_top;
      new_right = ori_right - rand_left;
      new_bottom = ori_bottom - rand_top;

      new_left = max(new_left, 0);
      new_top  = max(new_top, 0);
      new_right = min(new_right, rand_w);
      new_bottom = min(new_bottom, rand_h);

      new_width = new_right - new_left;
      new_height = new_bottom - new_top;

      box_label = []
      box_label.append(ori_labels[box_index + 0])
      box_label.append(float(new_left + new_right ) / 2 / rand_w)
      box_label.append(float(new_top + new_bottom ) / 2 / rand_h)
      box_label.append(float( new_width ) / rand_w)
      box_label.append(float( new_height ) / rand_h)


      if box_label[3] < 0.05 or box_label[4] < 0.05:
        continue

      box_label[1] = self.constrain(0.0, 1.0, box_label[1])
      box_label[2] = self.constrain(0.0, 1.0, box_label[2])
      box_label[3] = self.constrain(0.0, 1.0, box_label[3])
      box_label[4] = self.constrain(0.0, 1.0, box_label[4])

      if mirror:
        box_label[1] = max(0., 1 - box_label[1])

      box_labels += box_label
      

    image_crop = image[rand_y: rand_y + rand_h, rand_x : rand_x + rand_w].copy()

    if mirror:
      image_crop = cv2.flip(image_crop, 1)

    return (image_crop, box_labels)


  def pre_im_for_blob(self, im):
    pixel_means = np.array([[[128, 128, 128]]])
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im *= 0.0078125
    im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
    return im

  def transform_label_v1(self, box):
    locations = self.side * self.side
    label = np.zeros(locations * (1 + 1 + 4),  dtype=np.float)

    label[locations:locations*2] = -1

    for i in range(len(box) / 5):
      class_label = box[i * 5]
      x = box[i * 5 + 1]
      y = box[i * 5 + 2]
      x_index = int(x * self.side)
      y_index = int(y * self.side)

      obj_index = self.side * y_index + x_index
      class_index = locations + obj_index
      cor_index = locations * 2 + obj_index * 4

      label[obj_index] = 1
      label[class_index] = class_label
      label[cor_index: cor_index+4] = box[i*5+1 : i*5+5]
 
    return label

  def transform_label_v2(self, box):
    label = np.zeros(5*30, dtype=np.float)
    for i in range(len(box) / 5):
      #x = box[i*5 + 1]
      #y = box[i*5 + 2]
      #w = box[i*5 + 3]
      #h = box[i*5 + 4]
      #id = box[i * 5 + 0]

      #if w < .005 or h  < .005:
      #  continue

      #label[i*5 + 0] = x
      #label[i*5 + 1] = y
      #label[i*5 + 2] = w
      #label[i*5 + 3] = h
      #label[i*5 + 4] = id

      if box[i*5 + 3] < .005 or box[i*5 + 4]  < .005:
        continue

      label[i * 5 : i * 5 + 5] = box[i*5 +1 : i * 5 +5] + [ box[i*5] ]

    return label

  def get_next_minibatch(self):
    num_images = self.batch_size

    #sample = []
    #while len(sample) < num_images:
    #  sample.append(random.randint(0, self.data_num-1))

    sample = random.sample(self.lines, self.batch_size)

    image_blob = []
    label_blob = []
    for image_path in sample:
      image_path = image_path.replace('\n', '')

      label_path = image_path.replace('JPEGImages', 'labels')
      label_path = label_path.replace('jpg', 'txt')

      image = cv2.imread(image_path)
      
      #image = image.transpose((2, 0, 1))

      box = []
      label_lines = open(label_path).readlines()
      for label_line in label_lines:
        label_line = label_line.replace('\n', '')
        #print label_line.split(' ')[0:5]
        box += [float(x) for x in list(label_line.split(' ')[0:5])]

      image, box = self.random_crop(image, box)
      #print image
      #print box
      image = self.pre_im_for_blob(image)

      if self.version == 1:
        label = self.transform_label_v1(box)
      else:
        label = self.transform_label_v2(box)

      image_blob.append(image)
      label_blob.append(label)

    image_blob = self.im_list_to_blob(image_blob)

    blobs = {'data': image_blob, 'label': label_blob}
    return blobs