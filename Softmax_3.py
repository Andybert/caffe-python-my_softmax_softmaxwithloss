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
        self.num = bottom[0].data.shape[0]
        self.dim = bottom[0].data.shape[1]
        self.scale_data = np.zeros((self.num, 1), dtype=np.float32)
        self.icount = 1

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])

    def forward(self, bottom, top):
        # print 'iteration: ', self.icount
        top_data = np.zeros((self.num, self.dim), dtype=np.float32)
        top_data = bottom[0].data
        # print 'input data: \n', top_data
        largeconstant = 40
        # print 'largeconstant: ', largeconstant
        for i in xrange(self.num):
            self.scale_data[i] = top_data[i, 0]
            for j in xrange(self.dim):
                self.scale_data[i] = max(self.scale_data[i], top_data[i, j])
                if (np.isnan(top_data[i, j])):
                    print 'top_data[%d,%d]' % (i, j), top_data[i, j]
                    raise Exception('raw data is nan')
            # if (self.scale_data[i] > largeconstant):
            #     self.scale_data[i] = self.scale_data[i] - largeconstant
            # else:
            #     self.scale_data[i] = 0
        # print 'max: \n', self.scale_data
        for i in xrange(self.num):
            for j in xrange(self.dim):
                # substract = top_data[i, j] - self.scale_data[i]
                # if (np.exp(substract) == 0 or np.isnan(np.exp(substract))):
                #     print 'top_data[%d, %d] ' % (i, j), top_data[i, j] 
                #     print 'scale_data[%d] ' % i, self.scale_data[i]
                #     print 'substract: ', substract
                #     print 'exp: ', np.exp(substract)
                #     raise Exception('exp data is zero')
                top_data[i, j] = top_data[i, j] - self.scale_data[i]
                top_data[i, j] = max(np.exp(top_data[i, j]), 1e-10)
        for i in xrange(self.num):
            self.scale_data[i] = 0
            for j in xrange(self.dim):
                self.scale_data[i] = self.scale_data[i] + top_data[i, j]
            if (self.scale_data[i] == 0 or np.isnan(self.scale_data[i])):
                print 'self.scale_data[%d]' % i, self.scale_data[i]
                raise Exception('sum is zero')
        for i in xrange(self.num):
            for j in xrange(self.dim):
                top_data[i, j] = top_data[i, j] / self.scale_data[i]
        top[0].data[...] = top_data

    def backward(self, top, propagate_down, bottom):
        self.icount += 1
        # print 'iteration: ', self.icount
        bottom_diff = np.zeros((self.num, self.dim), dtype=np.float32)
        temp_diff = np.zeros(self.dim, dtype=np.float32)
        for m in xrange(self.num):
            for n in xrange(self.dim):
                temp_diff = np.zeros((self.dim), dtype=np.float32)
                for j in xrange(self.dim):
                    if (j == n):
                        temp_diff[j] = top[0].data[m, n] - top[0].data[m, j] * top[0].data[m, n]
                    else:
                        temp_diff[j] = 0 - top[0].data[m, j] * top[0].data[m, n]
                    bottom_diff[m, n] += top[0].diff[m, j] * temp_diff[j]
            if (np.isnan(bottom_diff[m, n])):
                print 'temp_diff:\n', temp_diff
                print 'bottom_diff[%d,%d]: ' % (m, n), bottom_diff[m, n]
                Programpause()
        bottom[0].diff[...] = bottom_diff


class SoftmaxwithlossLayer(caffe.Layer):
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
        self.params_ = SoftmaxwithlossLayer.parse_args(self.param_str)

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        label = bottom[1].data
        prob = bottom[0].data
        self.num = bottom[0].data.shape[0]
        self.dim = bottom[0].data.shape[1]
        loss = 0.0
        count = 0
        for i in range(self.num):
            if (self.params_.has_ignore_label is True and label[i] == self.params_.ignore_label):
                continue
            else:
                loss = loss - np.log(max(prob[i][int(label[i])], 1e-10))
                count += 1
        top[0].data[...] = loss / float(count)

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            bottom_diff = np.zeros_like(bottom[0].data)
            label = bottom[1].data
            count = 0
            for xi in range(self.num):
                if (self.params_.has_ignore_label is True and label[xi] == self.params_.ignore_label):
                    bottom_diff[:, int(label[xi])] = 0
                else:
                    bottom_diff[xi][int(label[xi])] = 1 / bottom[0].data[xi][int(label[xi])] 
                    count += 1
            bottom[i].diff[...] = 0 - bottom_diff / count
            # print 'bottom[%d].diff: \n' % i, bottom[i].diff
