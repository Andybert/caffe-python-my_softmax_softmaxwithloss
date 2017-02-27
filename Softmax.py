import caffe
import numpy as np
import argparse
import pprint
import math
import os


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class SoftmaxLayer(caffe.Layer):
    """
    Compute the Sofamax in the same manner as the C++ Softmax Layer
    to demonstrate the class interface for developing layers in Python.
    """
    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='SoftmaxLayer')
        parser.add_argument('--rho', default=0.1, type=float)
        parser.add_argument('--phase', default='', type=str)
        parser.add_argument('--test_interval', default=0, type=int)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Softmax Layer takes a single blob as input.")
        if len(top) != 1:
            raise Exception("Softmax Layer takes a single blob as output.")
        self.params_ = SoftmaxLayer.parse_args(self.param_str)
        self.rho = self.params_.rho
        self.scale_data = np.zeros(bottom[0].shape[1], dtype=np.float32)
        self.icount = 1

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])

    def forward(self, bottom, top):
        print 'iteration: ', self.icount
        self.num = bottom[0].data.shape[0]
        self.dim = bottom[0].data.shape[1]
        top_data = bottom[0].data
        self.scale_data = np.max(top_data, axis=1)
        self.scale_data.shape = (self.num, 1)
        top_data = top_data - self.scale_data
        top_data = np.exp(top_data)
        self.scale_data = np.sum(top_data, axis=1)
        self.scale_data.shape = (self.num, 1)
        top_data = top_data / self.scale_data
        self.icount += 1
        top[0].data[...] = top_data

    def backward(self, top, propagate_down, bottom):
        bottom_diff = top[0].diff
        # self.scale_data = np.dot(bottom_diff, top[0].data)
        # self.scale_data.shape = (self.num, 1)
        # bottom_diff = bottom_diff - self.scale_data
        bottom_diff = bottom_diff - bottom_diff * top[0].data
        bottom_diff = bottom_diff * top[0].data
        bottom[0].diff[...] = bottom_diff


class Softmaxwithloss(caffe.Layer):
    """
    Compute the Sofamax in the same manner as the C++ Softmax Layer
    to demonstrate the class interface for developing layers in Python.
    """
    @classmethod
    def parse_args(cls, argsStr):
        parser = argparse.ArgumentParser(description='Softmaxwithloss')
        parser.add_argument('--has_ignore_label', default=False, type=bool)
        parser.add_argument('--ignore_label', default=None, type=int)
        parser.add_argument('--normalize', default=False, type=bool)
        parser.add_argument('--softmax_axis', default=1, type=int)
        args = parser.parse_args(argsStr.split())
        print('Using Config:')
        pprint.pprint(args)
        return args

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Softmaxwithloss Layer Needs two blob as input.")
        if len(top) != 1:
            raise Exception("Softmaxwithloss Layer takes a single blob as output.")
        self.params_ = Softmaxwithloss.parse_args(self.param_str)

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        label = bottom[1].data
        prob = bottom[0].data
        self.num = bottom[0].data.shape[0]
        loss = 0.0
        count = 0
        for i in range(self.num):
            if (self.params_.has_ignore_label is True and label[i] == self.params_.ignore_label):
                Programpause()
                continue
            loss -= np.log(max(prob[i][int(label[i])], 1.5e-44))
            count += 1
        top[0].data[...] = loss / float(count)

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            bottom_diff = bottom[i].data
            label = bottom[1].data
            count = 0
            for xi in range(self.num):
                if (self.params_.has_ignore_label is True and label[xi] == self.params_.ignore_label):
                    bottom_diff[:, int(label[xi])] = 0
                else:
                    bottom_diff[xi][int(label[xi])] -= 1
                    count += 1
            bottom[i].diff[...] = bottom_diff / count
            print 'bottom[%d].diff: \n' % i, bottom[i].diff
