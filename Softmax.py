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
        self.scale_data = np.zeros((self.num, 1), dtype=np.float)
        self.icount = 1

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])

    def forward(self, bottom, top):
        # print 'iteration: ', self.icount
        top_data = np.zeros((self.num, self.dim), dtype=np.float)
        top_data = bottom[0].data
        for i in xrange(self.num):
            self.scale_data[i] = top_data[i, 0]
            for j in xrange(self.dim):
                if (self.scale_data[i] < top_data[i, j]):
                    self.scale_data[i] = top_data[i, j]
                if (np.isnan(top_data[i, j])):
                    print 'top_data[%d,%d]' % (i, j), top_data[i, j]
                    raise Exception('raw data is nan')
        for i in xrange(self.num):
            for j in xrange(self.dim):
                # print 'scale_data[%d]: ' % i, self.scale_data[i]
                top_data[i, j] = top_data[i, j] - self.scale_data[i]
                # print 'top_data[%d, %d]' % (i, j), top_data[i, j]
                top_data[i, j] = np.exp(top_data[i, j])
                # print 'exp top_data[%d, %d]' % (i, j), top_data[i, j]
        for i in xrange(self.num):
            self.scale_data[i] = 0
            for j in xrange(self.dim):
                self.scale_data[i] = self.scale_data[i] + top_data[i, j]
            if (self.scale_data[i] == 0 or np.isnan(self.scale_data[i])):
                for j in xrange(self.dim):
                    print 'top[%d,%d]' % (i, j), top_data[i, j]
                print 'self.scale_data[%d]' % i, self.scale_data[i]
                raise Exception('sum is zero')
        for i in xrange(self.num):
            for j in xrange(self.dim):
                top_data[i, j] = top_data[i, j] / self.scale_data[i]
        self.icount += 1
        top[0].data[...] = top_data

    def backward(self, top, propagate_down, bottom):
        bottom_diff = np.zeros((self.num, self.dim), dtype=np.float)
        bottom_diff = top[0].diff
        top_data = np.zeros((self.num, self.dim), dtype=np.float)
        top_data = top[0].data
        for i in xrange(self.num):
            self.scale_data[i] = np.inner(bottom_diff[i], top_data[i])
            print 'scale_data[%d]: ' % i, self.scale_data[i]
            print 'bottom_diff[%d]\n' % i, bottom_diff[i]
            bottom_diff[i] = bottom_diff[i] - self.scale_data[i]
            print 'bottom_diff[%d]\n' % i, bottom_diff[i]
        bottom_diff = bottom_diff * top_data
        # if (np.isnan(bottom_diff[m, n])):
        #     print 'temp_diff:\n', temp_diff
        #     print 'bottom_diff[%d,%d]: ' % (m, n), bottom_diff[m, n]
        #     Programpause()
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
        self.icount = 1

    def reshape(self, bottom, top):
        self.num = bottom[0].data.shape[0]
        self.dim = bottom[0].data.shape[1]
        self.scale_data = np.zeros((self.num, 1), dtype=np.float)
        self.prob = np.zeros((self.num, self.dim), dtype=np.float)
        top[0].reshape(1)

    def forward(self, bottom, top):
        print 'iteration: ', self.icount
        label = bottom[1].data
        bottom_data = bottom[0].data
        self.scale_data = np.max(bottom_data, axis=1)
        self.scale_data.shape = (self.num, 1)
        bottom_data = bottom_data - self.scale_data
        bottom_data = np.exp(bottom_data)
        self.scale_data = np.sum(bottom_data, axis=1)
        self.scale_data.shape = (self.num, 1)
        bottom_data = bottom_data / self.scale_data
        self.prob = bottom_data
        if (math.isnan(self.prob[0, 0])):
            Programpause()
        loss = 0.0
        count = 0
        for i in range(self.num):
            if (self.params_.has_ignore_label is True and label[i] == self.params_.ignore_label):
                continue
            else:
                loss = loss - np.log(max(self.prob[i][int(label[i])], 1e-40))
                count += 1
        top[0].data[...] = loss / float(count)
        self.icount += 1

    def backward(self, top, propagate_down, bottom):
        bottom_diff = self.prob
        label = np.zeros(self.num, dtype=int)
        label = bottom[1].data
        count = 0
        for xi in range(self.num):
            if (self.params_.has_ignore_label is True and label[xi] == self.params_.ignore_label):
                bottom_diff[:, int(label[xi])] = 0
            else:
                bottom_diff[xi][int(label[xi])] = bottom_diff[xi][int(label[xi])] - 1.0
                count += 1
        bottom[0].diff[...] = bottom_diff / count
