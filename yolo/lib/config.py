import os

# Training image data path
image_list = '/data/ImageSets/VOC/2007_val.txt'
batch_size = 64
image_size = 224
side = 1
version = 2


#detection layer
num_class = 0
num_object = 3
sqrt = True
constraint = True

object_scale = 1.0
noobject_scale = 0.5
class_scale = 1.0
coord_scale = 5.0


#v2 configuration
num_coord = 4

sqrt = True
constraint = True


do_softmax = True
rescore = True

thresh = 0.4
bias_match = True

anchors = [0.4, 0.5, 1.2,1.3, 0.7, 0.8]
