import __init_paths
import caffe
import numpy as np
from numpy import *
import math
import box_func
import math_func
from math_func import logistic_activate as logst_a
from math_func import logistic_gradient as logst_g
from math_func import softmax as softmax_f
import config

class RegionLossLayer(caffe.Layer):
  def setup(self, bottom, top):
    self.data_seen = 0
    (batch_size, channel, height, width) = bottom[0].data.shape

    self.b = batch_size
    self.w = width
    self.h = height

    self.num_class = config.num_class
    self.num_object = config.num_object
    self.n = self.num_object
    self.num_coord = config.num_coord

    self.sqrt = config.sqrt
    self.constraint = config.constraint
   
    self.object_scale = config.object_scale
    self.noobject_scale = config.noobject_scale
    self.class_scale = config.class_scale
    self.coord_scale = config.coord_scale

    self.do_softmax = config.do_softmax
    self.rescore = config.rescore

    self.thresh = config.thresh
    self.bias_match = config.bias_match

    try:
      self.anchors = config.anchors
    except:
      self.anchors = [0.5 for i in range(2*self.num_object)]

    self.diff = np.zeros(bottom[0].data.shape)
    self.output = np.zeros(bottom[0].data.shape)
    top[0].reshape(1)

    #some checks need to be coded
      
  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    self.data_seen += self.b
    input_data = bottom[0].data
    label_data = bottom[1].data
    size = self.num_coord + self.num_class + 1

    recall = 0

    self.diff[...] = 0

    self.output = input_data.transpose((0, 2, 3, 1))
    self.diff = self.diff.transpose((0, 2, 3, 1))

    losses = np.zeros(6) #loss, obj_loss, noobj_loss, class_loss, coord_loss, area_loss
    averages = np.zeros(6) #avg_iou, avg_obj, avg_no_obj, avg_cls, avg_pos_cls, obj_count_f

    for i in range(self.num_object):
      self.output[:, :, :, i*size+4:i*size+5] = logst_a(self.output[:, :, :, i*size+4:i*size+5])

    if self.do_softmax:
      for i in range(self.num_object):
        self.output[:,:,:,i*size+5:i*size+5+self.num_class] = softmax_f(self.output[:,:,:,i*size+5:i*size+5+self.num_class])
    
    for b in range(self.b):
      for j in range(self.h):
        for i in range(self.w):
          for n in range(self.n):
            index = size * n
            output_slice = self.output[b,j,i,index:index+4]
            anchor_slice = self.anchors[2*n:2*(n+1)]
            diff_slice = self.diff[b,j,i,index:index+4]
            output_value = self.output[b,j,i,index+4]

            pred_box = self.get_region_box(output_slice, anchor_slice, i, j)
            best_iou = 0
            for t in range(30):
              truth = label_data[b, t*5:t*5+4]
              if truth[0] == 0:
                break
              iou = box_func.box_iou  (pred_box, truth)
              if iou > best_iou:
                best_iou = iou
              
              if best_iou > self.thresh:
                self.diff[b,j,i,index+4] = 0
              else:
                averages[2] += self.output[b,j,i,index+4]
                losses[2] += self.noobject_scale * math.pow(output_value, 2)
                self.diff[b,j,i,index+4] = (-1) * self.noobject_scale * ((0-output_value) * logst_g(output_value))
    
              
              if self.data_seen < 12800:
                truth = [0, 0, 0, 0]
                truth[0] = (i + 0.5) / self.w
                truth[1] = (j + 0.5) / self.h
                truth[2] = self.anchors[2*n] / self.w
                truth[3] = self.anchors[2*n + 1] / self.h

                self.delta_region_box(truth, output_slice, anchor_slice, i, j, 0.01, diff_slice, losses)     
      
      for t in range(30):
        truth = label_data[b,t*5: t*5+4]
        if truth[0] == 0:
          break
        best_iou = 0
        best_index = 0
        i = int(truth[0] * self.w)
        j = int(truth[1] * self.h)

        truth_shift = truth.copy()
        truth_shift[0] = 0
        truth_shift[1] = 0

        for n in range(self.n):        
          index = size * n
          output_slice = self.output[b,j,i,index:index+4]
          anchor_slice = self.anchors[2*n:2*(n+1)]
          diff_slice = self.diff[b,j,i,index:index+4]
          pred = self.get_region_box(output_slice, anchor_slice, i, j)
          if self.bias_match:
            pred[2] = self.anchors[2*n] / self.w
            pred[3] = self.anchors[2*n+1] / self.h
          pred[0] = 0
          pred[1] = 0

          iou = box_func.box_iou(pred, truth_shift)
          if iou > best_iou:
            best_index = index
            best_iou = iou
            best_n = n

        output_slice_best = self.output[b,j,i,best_index:best_index+4]
        anchor_slice_best = self.anchors[best_n*2:(best_n+1)*2]
        diff_slice_best = self.diff[b,j,i,best_index:best_index+4]
        output_value_best = self.output[b,j,i,best_index+4]
        iou = self.delta_region_box(truth, output_slice_best, anchor_slice_best, i, j, self.coord_scale, diff_slice_best,losses)
        if iou > 0.5:
          averages[4] += 1
        averages[0] += iou
        averages[1] += output_value_best

        log_o_g = logst_g(output_value_best)
        if self.rescore:
          losses[1] += self.object_scale * math.pow(iou - output_value_best, 2)
          self.diff[b,j,i,best_index+4] = (-1) * self.object_scale * (iou - output_value_best) * log_o_g
        else:
          losses[1] += self.object_scale * math.pow(1 - self.output[b,j,i,best_index + 4], 2)
          self.diff[b,j,i,best_index+4] = (-1) * self.object_scale * (1 - output_value_best) * log_o_g

        class_ind = label_data[b,t*5+4]
        output_slice_cls = self.output[b,j,i,best_index+5:best_index+5+self.n]
        diff_slice_cls = self.diff[b,j,i,best_index+5:best_index+5+self.n]

        self.delta_region_class(output_slice_cls, diff_slice_cls, class_ind, self.num_class, self.class_scale, losses, averages)
        averages[5] += 1

    self.output = self.output.transpose((0, 3, 1, 2))
    self.diff = self.diff .transpose((0, 3, 1, 2))

    locations = self.w * self.h
    num = self.b

    averages[5] += 0.01
    losses[1] /= averages[5]
    losses[2] /= (locations * self.num_object * num - averages[5])
    losses[3] /= averages[5]
    losses[4] /= averages[5]
    losses[5] /= averages[5]
    
    averages[0] /= averages[5]
    averages[1] /= averages[5]
    averages[2] /= (locations * self.num_object * num - averages[5])
    averages[3] /= averages[5]
    averages[4] /= averages[5]

    losses[0] = losses[3] + losses[4] + losses[5] + losses[1] + losses[2]
    averages[5] /= num
    
    top[0].data[...] = losses[0]

    loss_str = 'loss: %f obj_loss: %f noobj_loss: %f class_loss: %f coord_loss: %f area_loss: %f' \
                % ( losses[0], losses[1], losses[2], losses[3], losses[4], losses[5])
    avg_str = 'avg_iou: %f avg_obj: %f avg_no_obj: %f avg_cls: %f avg_recall: %f obj_count: %f' \
                % (averages[0], averages[1], averages[2], averages[3], averages[4], averages[5])
    
    space_str = '        '

    print space_str + loss_str
    print space_str + avg_str

  def backward(self, top, propagate_down, bottom):
    if propagate_down[0]:
      sign = 1.0
      #alpha = sign * top[0].diff[0] / len(bottom[0].data)
      alpha = sign / len(bottom[0].data)
      bottom[0].diff[...] = sign * alpha * self.diff
      #print bottom[0].diff[1,:,2,3]

  def get_region_box(self, x, anchors, i, j):
    w = self.w
    h = self.h
    pred_box = []
    pred_box.append((i + logst_a(x[0])) / w)
    pred_box.append((j + logst_a(x[1])) / h)
    pred_box.append(math.exp(x[2]) * anchors[0] / w)
    pred_box.append(math.exp(x[3]) * anchors[1] / h)
    return pred_box

  def delta_region_box(self, truth, x, anchors, i, j, scale, delta, losses):
    w = self.w
    h = self.h
    pred = self.get_region_box(x, anchors, i, j)
    iou = box_func.box_iou(pred, truth)
    tx = (truth[0] * w - i)
    ty = (truth[1] * h - j)
    tw = math.log(truth[2] * w / anchors[0])
    th = math.log(truth[3] * h / anchors[1])

    log_x = logst_a(x[0])
    log_y = logst_a(x[1])
    log_x_g = logst_g(log_x)
    log_y_g = logst_g(log_y)

    delta[0] = ((-1) * scale * (tx - log_x) * log_x_g)
    delta[1] = ((-1) * scale * (ty - log_y) * log_y_g)
    delta[2] = ((-1) * scale * (tw - x[2]))
    delta[3] = ((-1) * scale * (th - x[3]))
   
    losses[4] += scale * (math.pow(tx - log_x, 2) + math.pow(ty-log_y,2))
    losses[5] += scale * (math.pow(tw - x[2],2) + math.pow(th-x[3], 2))
    return iou 

  def delta_region_class(self, output, delta, class_ind, classes, scale, losses, averages):
    for n in range(classes):
      ins_tag = 1. if n == class_ind else 0.
      delta[n] = (-1) * scale * ins_tag - output[n]
      losses[3] += scale * math.pow(ins_tag - output[n], 2)
      #averages[3] += output[n]
      averages[3] += ins_tag * output[n]