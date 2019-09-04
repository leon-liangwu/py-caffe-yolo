# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Train the network."""

import __init_paths
import caffe
from timer import Timer
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
import sys
import config
import fire


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver_prototxt,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = None # not in use for now

        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)


    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = (self.solver_param.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print('Wrote snapshot to: {:s}'.format(filename))


    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            timer.tic()
            self.solver.step(1)	    
            #print 'fc9_1:',sorted(self.solver.net.params['fc9_1'][0].data[0])[-1]
            #print 'fc9:',sorted(self.solver.net.params['fc9'][0].data[0])[-1]
            #print 'fc7:',sorted(self.solver.net.params['fc7'][0].data[0])[-1]
            #print 'fc6:',sorted(self.solver.net.params['fc6'][0].data[0])[-1]
            #print 'fc9:',(self.solver.net.params['fc9'][0].data[0])[0]
            #print 'fc7:',(self.solver.net.params['fc7'][0].data[0])[0]
            #print 'fc6:',(self.solver.net.params['fc6'][0].data[0])[0]
            #print 'conv5_3:',self.solver.net.params['conv5_3'][0].data[0][0][0]
            #print 'conv5_2:',self.solver.net.params['conv5_2'][0].data[0][0][0]
            #print 'conv5_1:',self.solver.net.params['conv5_1'][0].data[0][0][0]
            #print 'conv4_3:',self.solver.net.params['conv4_3'][0].data[0][0][0]
            #print 'fc9:',self.solver.net.params['fc9'][0].data[0][0]
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time)          )


def run(solver,
        weights=None):
    """Train network."""
    solver_prototxt = solver
    pretrained_model = weights

    max_iters = 30000 #config.MAX_ITERS
    sw = SolverWrapper(solver_prototxt, pretrained_model)
    
    print('Start solving...')
    sw.train_model(max_iters)
    print('Done solving!')


if __name__ == '__main__':
    fire.Fire(run)
