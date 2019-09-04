import __init_paths
import caffe
import numpy as np
from numpy import *
import math
import yaml
import box_func
# import config


class DetectionLossLayer(caffe.Layer):
  def setup(self, bottom, top):
    param = yaml.load(self.param_str)
    
    self.side = param['side']
    self.num_class = param['num_class']
    self.num_object = param['num_object']

    self.sqrt = self.get_param(param, 'sqrt', True)
    self.constraint = self.get_param(param, 'constraint', True)

    self.object_scale = self.get_param(param, 'object_scale', 1.0)
    self.noobject_scale = self.get_param(param, 'noobject_scale', 0.5)
    self.class_scale = self.get_param(param, 'class_scale', 1.0)
    self.coord_scale = self.get_param(param, 'coord_scale', 5.0)

    self.diff = np.zeros(bottom[0].data.shape)
    top[0].reshape(1)
    #some checks need to be coded

  def get_param(self, param, name, def_val):
    if name in param:
      return param[name]
    return def_val
      
  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    input_data = bottom[0].data
    label_data = bottom[1].data
    self.diff[...] = 0

    losses = np.zeros(6) #loss, obj_loss, noobj_loss, class_loss, coord_loss, area_loss
    averages = np.zeros(6) #avg_iou, avg_obj, avg_no_obj, avg_cls, avg_pos_cls, obj_count_f
    
    locations = int(math.pow(self.side, 2))
    (num, feat_num) = input_data.shape

    #print input_data[0]
    #print label_data[0]

    for i in range(num):
      for j in range(locations):
        for k in range(self.num_object):
          p_index = self.num_class * locations + k * locations + j
          losses[2] += self.noobject_scale * math.pow(input_data[i][p_index] - 0, 2)
          self.diff[i][p_index] = self.noobject_scale * (input_data[i][p_index] - 0)
          averages[2] += input_data[i][p_index]

        isobj = label_data[i][j]
        if not isobj:
          continue

        averages[5] += 1
        label = label_data[i][locations + j]
        for c in range(self.num_class):
          class_index = c * locations + j
          target = 1.0 if (c == label) else 0.0
          averages[3] += target * input_data[i][class_index]
          losses[3] += self.class_scale * math.pow(input_data[i][class_index] - target, 2)
          self.diff[i][class_index] = self.class_scale * (input_data[i][class_index] - target)

        true_box_index = locations * 2 + j * 4
        true_box = label_data[i][true_box_index : true_box_index + 4].copy()
        box_index = (self.num_class + self.num_object) * locations + j
        best_iou = 0
        best_rmse = 20
        best_index = 0
        for k in range(self.num_object):
          box = []
          box.append(input_data[i][box_index + (k* 4 + 0) * locations])
          box.append(input_data[i][box_index + (k* 4 + 1) * locations])
          box.append(input_data[i][box_index + (k* 4 + 2) * locations])
          box.append(input_data[i][box_index + (k* 4 + 3) * locations])
          #print box

          if self.constraint:
            box[0] = (j % self.side + box[0]) / self.side
            box[1] = (j / self.side + box[1]) / self.side

          if self.sqrt:
            box[2] = math.pow(box[2], 2)
            box[3] = math.pow(box[3], 2)

          #print box
          #print true_box
          iou = box_func.box_iou(box, true_box)
          rmse = box_func.box_rmse(box, true_box)

          if best_iou > 0 or iou > 0:
              if iou > best_iou:
                best_iou = iou
                best_index = k
          else:
            if rmse < best_rmse:
              best_rmse = rmse
              best_index = k

        averages[0] += best_iou
        if best_iou > 0.5:
          averages[4] += 1
        #print 'best iou: %f' % (best_iou)
        p_index = self.num_class * locations + best_index * locations +j
        losses[2] -= self.noobject_scale * math.pow(input_data[i][p_index], 2)
        losses[1] += self.object_scale * math.pow(input_data[i][p_index] - 1., 2)
        averages[2] -= input_data[i][p_index]
        averages[1] += input_data[i][p_index]

        #rescore
        self.diff[i][p_index] = self.object_scale * (input_data[i][p_index] - best_iou)
        #no-rescore
        #self.diff[i][p_index] = self.object_scale * (input_data[i][p_index] - 1)
        box_index = (self.num_class + self.num_object + best_index*4) * locations + j
        best_box = []
        best_box.append(input_data[i][box_index + 0 * locations])
        best_box.append(input_data[i][box_index + 1 * locations])
        best_box.append(input_data[i][box_index + 2 * locations])
        best_box.append(input_data[i][box_index + 3 * locations])
        
        if self.constraint:
          true_box[0] = true_box[0] * self.side - j % self.side
          true_box[1] = true_box[1] * self.side - j / self.side

        if self.sqrt:
          true_box[2] = math.sqrt(true_box[2])
          true_box[3] = math.sqrt(true_box[3])

        for o in range(4):
          self.diff[i][box_index + o * locations] = self.coord_scale * (best_box[o] - true_box[o])

        losses[4] += self.coord_scale * math.pow(best_box[0] - true_box[0], 2)
        losses[4] += self.coord_scale * math.pow(best_box[1] - true_box[1], 2)
        losses[5] += self.coord_scale * math.pow(best_box[2] - true_box[2], 2)
        losses[5] += self.coord_scale * math.pow(best_box[3] - true_box[3], 2)


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
    

    #print top[0]
    #top[0].data[...] = [loss, avg_iou]
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