#!/usr/bin/env python
# -*- coding:utf-8 -*- -

import tensorflow as tf
import numpy as np
import gym
import os
import math
import csv
import sys
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import random
import time
import datetime
import pickle
from tqdm import *
import heapq

# sys.path.append('/home/pygmalionchen/PycharmProjects/treasure/tf_rl')
sys.path.append('/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/tf_rl')
save_path = '/home/pygmalionchen/PycharmProjects/treasure/logs/ddpg/models/'
import tf_rl
from gym import envs

#####################  hyper parameters  ####################

# MAX_EP_STEPS = 200
MAX_EPISODES = 20000
MAX_GOAL_STEPS = 1100  # 或许下面这些学习率也可以做一组对比试验
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.95  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
START_LEARN = 50000
nor = 1 #8  # 8 number of robots


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # fn = file("/home/szz/tf_rl/examples/multi/memory_tree.pkl", "rb")
        # self.tree = pickle.load(fn)
        # fn.close()
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n, update=True):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        if update:
            for i in range(n):
                a, b = pri_seg * i, pri_seg * (i + 1)
                v = np.random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(v)
                prob = p / self.tree.total_p
                ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
                b_idx[i], b_memory[i, :] = idx, data
        else:  # 树没有塞满，不update叶子节点的p，只是随机取样
            indices = np.random.choice(np.minimum(self.tree.data_pointer, MEMORY_CAPACITY), size=n)
            for i in range(n):
                # idx, p, data = self.tree.get_leaf(indices[i])
                data = self.tree.data[indices[i]]
                b_memory[i, :] = data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, p_dim, ):
        # self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + p_dim * 2 + a_dim + 1), dtype=np.float32)
        self.memory = Memory(MEMORY_CAPACITY)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.p_dim = a_dim, s_dim, p_dim,
        self.S = tf.placeholder(tf.float32, [None, s_dim, 1], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim, 1], 's_')
        self.P = tf.placeholder(tf.float32, [None, p_dim], 'p')
        self.P_ = tf.placeholder(tf.float32, [None, p_dim], 'p_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.loss_count = 0  # 用于绘制图线

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
        self.out_dir = os.path.abspath(os.path.join(save_path, timestamp))

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, self.P, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, self.P_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.P, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, self.P_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.abs_errors = tf.reduce_sum(tf.abs(q_target - q), axis=1)
        tf.summary.scalar('td_error', td_error)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        tf.summary.scalar('negative_q_val', self.a_loss)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

        # self.saver.restore(self.sess,"/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/logs/GoodModule/formation/model_190000.ckpt")
        # self.saver.restore(self.sess,"/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/logs/GoodModule/toPoints/model_53000.ckpt")
        self.saver.restore(self.sess,"/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/logs/DDPG_1Go/2019-07-02_23_07/models/model_880000.ckpt")

        self.merged = tf.summary.merge_all()
        self.loss_writer = tf.summary.FileWriter(self.out_dir, self.sess.graph)

    def choose_action(self, s, p):
        return self.sess.run(self.a, {self.S: s, self.P: p})
        # return self.sess.run(self.a, {self.S: s[np.newaxis, :], self.P: p[np.newaxis, :]})[0]

    def learn(self, update=True):
        # soft target replacement
        self.sess.run(self.soft_replace)

        # indices = np.random.choice(min(self.pointer,MEMORY_CAPACITY), size=BATCH_SIZE)
        # b_M = self.memory[indices, :]
        tree_idx, b_M, ISWeights = self.memory.sample(BATCH_SIZE, update)
        bs = b_M[:, :self.s_dim]
        bp = b_M[:, self.s_dim: self.s_dim + self.p_dim]
        ba = b_M[:, self.s_dim + self.p_dim: self.s_dim + self.a_dim + self.p_dim]
        br = b_M[:, -self.p_dim - self.s_dim - 1: -self.p_dim - self.s_dim]
        bs_ = b_M[:, -self.p_dim - self.s_dim: -self.p_dim]
        bp_ = b_M[:, -self.p_dim:]
        if update:
            abs_err = self.sess.run(self.abs_errors,
                                    {self.S: bs.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                     self.P: bp.reshape(bp.shape[0], self.p_dim),
                                     self.a: ba, self.R: br,
                                     self.S_: bs_.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                     self.P_: bp_.reshape(bp.shape[0], self.p_dim)})
            self.memory.batch_update(tree_idx, abs_err)
        self.sess.run(self.atrain, {self.S: bs.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                    self.P: bp.reshape(bp.shape[0], self.p_dim)})
        self.sess.run(self.ctrain, {self.S: bs.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                    self.P: bp.reshape(bp.shape[0], self.p_dim),
                                    self.a: ba, self.R: br,
                                    self.S_: bs_.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                    self.P_: bp_.reshape(bp.shape[0], self.p_dim)})
        summary = self.sess.run(self.merged, {self.S: bs.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                              self.P: bp.reshape(bp.shape[0], self.p_dim),
                                              self.a: ba, self.R: br,
                                              self.S_: bs_.reshape([BATCH_SIZE, 1, self.s_dim]).transpose(0, 2, 1),
                                              self.P_: bp_.reshape(bp.shape[0], self.p_dim)})
        self.loss_writer.add_summary(summary, self.loss_count)
        self.loss_count += 1
        if self.loss_count % 1000 == 0:
            self.saver.save(self.sess, os.path.abspath(os.path.join(self.out_dir, "models/model_%d.ckpt" % self.loss_count)))

    def store_transition(self, s, p, a, r, s_, p_):
        transition = np.hstack((s, p, a, [r], s_, p_))
        self.memory.store(transition)
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.pointer += 1

    def _build_a(self, s, p, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.03)
            # init_w = tf.contrib.layers.xavier_initializer()
            # init_b = tf.contrib.layers.xavier_initializer()
            init_b = tf.random_normal_initializer(0., 0.01)
            with tf.variable_scope('conv0'):
                c0 = tf.layers.conv1d(s, 64, 7, 3, 'same', kernel_initializer=init_w,
                                      bias_initializer=init_b, name='c0', trainable=trainable)
                bn0 = tf.layers.batch_normalization(c0, trainable=trainable)
                r0 = tf.nn.relu(bn0)
                p0 = tf.layers.max_pooling1d(r0, 3, 1, 'same')
                # pool = tf.layers.max_pooling1d(inputs=c0, pool_size=4, strides=4, padding='same')
            with tf.variable_scope('conv1'):
                c1 = tf.layers.conv1d(p0, 64, 3, 1, 'same', kernel_initializer=init_w,
                                      bias_initializer=init_b, name='c1', trainable=trainable)
                bn1 = tf.layers.batch_normalization(c1, trainable=trainable)
                r1 = tf.nn.relu(bn1)
            with tf.variable_scope('conv2'):
                c2 = tf.layers.conv1d(r1, 64, 3, 1, 'same', kernel_initializer=init_w,
                                      bias_initializer=
                                      init_b, name='c2', trainable=trainable)
                bn2 = tf.layers.batch_normalization(c2, trainable=trainable)

                r2 = tf.nn.relu(p0 + bn2)
                p2 = tf.layers.average_pooling1d(r2, 3, 1, 'same')
            with tf.variable_scope('fc1'):
                shp = p2.get_shape()
                dim = shp[1].value * shp[2].value
                # print(bs)
                f = tf.reshape(r2, [-1, dim])
                combine0 = tf.concat([f, p], 1)
                hidden1 = tf.layers.dense(combine0, 512, activation=tf.nn.relu,
                                          kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                          trainable=trainable)
            with tf.variable_scope('ex'):
                with tf.variable_scope('fc2'):
                    # combine0 = tf.concat([hidden1, p], 1)
                    hidden2 = tf.layers.dense(hidden1, 256, activation=tf.nn.relu,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                              trainable=trainable)
                with tf.variable_scope('fc3'):
                    hidden3 = tf.layers.dense(hidden2, 128, activation=tf.nn.relu,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                              trainable=trainable)
                with tf.variable_scope('vl'):
                    # s_in = tf.concat([hidden0, p], 1)
                    vl = tf.layers.dense(hidden3, 1, activation=tf.nn.sigmoid, kernel_initializer=init_w,
                                         bias_initializer=init_b, name='vl', trainable=trainable)
                with tf.variable_scope('va'):
                    va = tf.layers.dense(hidden3, 1, activation=tf.nn.tanh, kernel_initializer=init_w,
                                         bias_initializer=init_b, name='va', trainable=trainable)
                    # va = (va-0.5)*2
                    # scaled_a = tf.multiply(actions, 3, name='scaled_a')
                scaled_a = tf.concat([vl, va], 1)
        return scaled_a

    def _build_c(self, s, p, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.03)
            # init_w = tf.contrib.layers.xavier_initializer()
            # init_b = tf.contrib.layers.xavier_initializer()
            init_b = tf.random_normal_initializer(0., 0.01)
            with tf.variable_scope('conv0'):
                c0 = tf.layers.conv1d(s, 64, 7, 3, 'same', kernel_initializer=init_w,
                                      bias_initializer=init_b, name='c0', trainable=trainable)
                bn0 = tf.layers.batch_normalization(c0, trainable=trainable)
                r0 = tf.nn.relu(bn0)
                p0 = tf.layers.max_pooling1d(r0, 3, 1, 'same')
                # pool = tf.layers.max_pooling1d(inputs=c0, pool_size=4, strides=4, padding='same')
            with tf.variable_scope('conv1'):
                c1 = tf.layers.conv1d(p0, 64, 3, 1, 'same', kernel_initializer=init_w,
                                      bias_initializer=init_b, name='c1', trainable=trainable)
                bn1 = tf.layers.batch_normalization(c1, trainable=trainable)
                r1 = tf.nn.relu(bn1)
            with tf.variable_scope('conv2'):
                c2 = tf.layers.conv1d(r1, 64, 3, 1, 'same', kernel_initializer=init_w,
                                      bias_initializer=
                                      init_b, name='c2', trainable=trainable)
                bn2 = tf.layers.batch_normalization(c2, trainable=trainable)

                r2 = tf.nn.relu(p0 + bn2)
                p2 = tf.layers.average_pooling1d(r2, 3, 1, 'same')
            with tf.variable_scope('fc1'):
                shp = r2.get_shape()
                dim = shp[1].value * shp[2].value
                # print(bs)
                f = tf.reshape(r2, [-1, dim])
                combine0 = tf.concat([f, p, a], 1)
                hidden1 = tf.layers.dense(combine0, 512, activation=tf.nn.relu,
                                          kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                          trainable=trainable)
            with tf.variable_scope('ex'):
                with tf.variable_scope('fc2'):
                    hidden2 = tf.layers.dense(hidden1, 256, activation=tf.nn.relu,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                              trainable=trainable)
                with tf.variable_scope('fc3'):
                    hidden3 = tf.layers.dense(hidden2, 128, activation=tf.nn.relu,
                                              kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                              trainable=trainable)
                with tf.variable_scope('q'):
                    q = tf.layers.dense(hidden3, 1, kernel_initializer=init_w,
                                        bias_initializer=init_b, name='a', trainable=trainable)
        return q


###############################  training  ####################################

# env = gym.make('DDPGEnv-v0')
# env = gym.make('formDDPGEnv-v0')
env = gym.make('DDPG1Go-v0')
env.seed(2)
s_dim = 180
a_dim = 2
p_dim = nor + 3 # nor + 3  # 4
ddpg = DDPG(a_dim, s_dim, p_dim)

Test = True
TestEpisodes = 1000
t1 = time.time()
test_count = 0
success_count = 0
# env.get_all_init_states() #获取机器人初始位置
first = True
for i in trange(TestEpisodes):
    env._reset()
    ep_reward = 0.
    step = 0  # count the steps
    j = 0
    CourseGoal = [[5, -6], [5, -4], [3, -3], [-3, -2], [-2, 0], [5, 2], [5, 4]]
    goal = CourseGoal[test_count % 5]
    print("Goal is: ", goal)
    action = np.zeros([nor, 2])
    s, p, r, done = env._step(action, goal, first)
    first = False
    while (j < MAX_GOAL_STEPS and not done):
        action = ddpg.choose_action(s.reshape([nor, 1, s_dim]).transpose(0, 2, 1), p)
        #############################################################
        if 0:
            gl_dis = np.sqrt((env.listen_class.odom_list[0].pose.pose.position.x - goal[0]) ** 2 + ( env.listen_class.odom_list[0].pose.pose.position.y - goal[1]) ** 2) / env.max_range_dis
            print("gl_dis: ", gl_dis)
            head_largest = np.max(heapq.nlargest(30, s[0][60:120]))
            # print("head_largest:", head_largest)
            gl_theta = env.cal_theta(goal)
            # 目标点与机器人间无障碍
            if (head_largest < gl_dis) and (abs(gl_theta) < 0.67):
                action[0, 0] = 0.5 * gl_dis + 0.5 * action[0, 0]
                action[0, 1] = 0.5 * gl_theta + 0.5 * action[0, 1]
                pass
            else:  # 目标点与机器人间存在障碍,即先要避开当前的障碍
                # 激光观测数组从机器人的右边到左边排列,即数组低位对应右端激光.
                largest_index = Nlargest(s[0])
                direction = np.mean(largest_index)  # 270°对应180维, 求得方向决定后续打角.
                # print("The direction:",direction)
                # print("The index:",largest_index)
                if direction < 60:  # action[0, 1]角速度为负数, 则机器人右转.反之左转。
                    action[0, 1] = -abs(90 - direction) * 0.01 + np.clip(np.random.normal(-0.5, 0.1), -0.7, -0.3)
                elif direction >= 60 and direction < 120:
                    pass
                else:
                    action[0, 1] = abs(90 - direction) * 0.01 + np.clip(np.random.normal(0.5, 0.1), 0.3, 0.7)  # np.clip(np.random.normal(0.2, 0.1), 0, 0.3) 转向能力不足
                pass
        ##################################################################
        s_, p_, r, done = env._step(action, goal)
        p = p_.copy()
        s = s_.copy()
        j += 1
        if done:
            success_count += 1
            print("Success step: ",j)
        else:
            pass
    if (i % 100 == 0):
        test_count += 1
        MAX_GOAL_STEPS += 150
    else:
        pass
print("Arrival Rate: ", success_count/TestEpisodes)


