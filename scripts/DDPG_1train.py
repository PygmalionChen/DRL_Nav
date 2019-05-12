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

# sys.path.append('/home/pygmalionchen/PycharmProjects/treasure/tf_rl')
sys.path.append('/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/tf_rl')
save_path = '/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/logs/DDPG_1Go/'
import tf_rl
from gym import envs

#####################  hyper parameters  ####################

# MAX_EP_STEPS = 200
MAX_EPISODES = 20000 #20000
MAX_GOAL_STEPS = 2500  # 参考实际地图何时能走到去设置
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.95  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE =  128# 256
nor = 1  # number of robots


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
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
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
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True  # 增长式分配的方式容易造成较多的内存碎片.
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        # session = tf.Session(config=config, ...)
        self.sess = tf.Session(config=config)

        self.a_dim, self.s_dim, self.p_dim = a_dim, s_dim, p_dim,
        self.S = tf.placeholder(tf.float32, [None, s_dim, 1], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim, 1], 's_')
        self.P = tf.placeholder(tf.float32, [None, p_dim], 'p')
        self.P_ = tf.placeholder(tf.float32, [None, p_dim], 'p_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.loss_count = 0  # 用于绘制图线

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
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
        self.saver = tf.train.Saver(max_to_keep=1000)
        # self.saver.restore(self.sess, "/home/pygmalionchen/PycharmProjects/treasure/logs/pddpg/models/toPoints/model_53000.ckpt")   # 选了toPoint的预训练模型载入
        # self.saver.restore(self.sess,"/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/logs/DDPG_1Go/2019-05-07_20:31/models/model_350000.ckpt")
        # self.saver.restore(self.sess,"/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/logs/DDPG_1Go/2019-05-11/models/model_150000.ckpt")
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
        if self.loss_count % 5000 == 0: # save the model for every 5000 count
            self.saver.save(self.sess,
                            os.path.abspath(os.path.join(self.out_dir, "models/model_%d.ckpt" % self.loss_count)))

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
                                      bias_initializer=init_b, name='c2', trainable=trainable)
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
                # 网络输入 action 的原因是
                combine0 = tf.concat([f, p,  a], 1)  # p本质也是观测的一部分,但其内容不需要提取特征所以放在中间加入.
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
env = gym.make('DDPG1Go-v0')
env.seed(2)
s_dim = 180  # 均值滤波以后的激光雷达数据,原始数据720维
a_dim = 2   # 线速度 角速度
p_dim = nor + 3   # 添加距离目标点的角度 # 移动机器人的参数空间 4 [goal_dis, theta_error, linear.x, angular.z]
var = 0.71# 0.8  # control exploration
t1 = time.time()
total_step = 0
ep_count = 0
ddpg = DDPG(a_dim, s_dim, p_dim)
# env.get_all_init_states() #获取机器人初始位置
first = True
reward_list = []
course_count = 0
for i in trange(MAX_EPISODES):
    env._reset()
    ep_reward = 0.
    x1 = 0
    y1 = 0
    step = 0  # count the steps
    j = 0
    # 间隔50个 episode 调整 course 难度
    # 课程学习方式 # 初始中心点(-4,-7)
    course_goal = [[6, -5], [3, -3], [-3, -2], [5, 2],]
    c, d = course_goal[course_count % 5]
    if i % 5000 == 0:
        course_count += 1
    else:
        pass
    goal = [c, d] + np.random.rand(2) * 2
    print('goal center is x: %f, y: %f' % (c, d))
    print("Goal is: ", goal)
    action = np.zeros([nor, 2])
    s, p, r, done = env._step(action, goal, first)
    first = False
    # va_math = 0.
    var *= .9998
    # print("Step: ",step)
    # goal = goal + (np.random.rand(2) - 0.5)  # 为训练部分的目标点添加一定的随机性
    while (j < MAX_GOAL_STEPS):  # and (success_num + fail_num <= 100):
        # a = ddpg.choose_action(s/10, p)
        action = ddpg.choose_action(s.reshape([nor, 1, s_dim]).transpose(0, 2, 1), p)
        # print("Action:",a) #  list with (8,2) shape.
        action[:, 0] = np.clip(np.random.normal(action[:, 0], var), 0, 1)   # linear V
        action[:, 1] = np.clip(np.random.normal(action[:, 1], var), -1, 1)  # angular V
        ## 多控制器切换, 基本控制器有毒
        if np.random.rand() < var/2:
            print("The basic Control Law.")
            if s[0].min() <= 0.1:  # obstacle avoidance
                # 判定障碍物方位, 决定左右转
                if s.argmin() > len(s)/2:
                    # a[1] = -abs(s.min()-0.1)/0.1
                    action[0, 1] = np.clip(np.random.normal(-0.8, 0.2), -1, -0.7)
                else:
                    # a[1] = abs(s.min()-0.1) / 0.1
                    action[0, 1] = np.clip(np.random.normal(0.8, 0.2), 0.7, 1)
            elif s[0].min() <= 0.2:
                if s.argmin() > len(s)/2:
                    # a[1] = -abs(s.min()-0.1)/0.1
                    action[0, 1] = np.clip(np.random.normal(-0.15, 0.1), -0.3, 0)
                else:
                    # a[1] = abs(s.min()-0.1) / 0.1
                    action[0, 1] = np.clip(np.random.normal(0.15, 0.1), 0, 0.3)
                pass
            else:
                pass
                # # 目标导航控制器
                # # 夹角大
                # if abs(p[0][3]) > 0.6:
                #     action[0, 1] = - p[0][3] / abs(p[0][3]) * 0.7
                #     action[0, 0] = 0.1
                # #  夹角小
                # elif abs(p[0][3]) < 0.3:
                #     action[0, 1] = -p[0][3] / abs(p[0][3]) * 0.2
                #     action[0, 0] = 0.5
                # else:
                #     action[0, 1] = -p[0][3] / abs(p[0][3]) * 0.5
                #     action[0, 0] = 0.3
        else:
            print("The DDPG Law.")
            pass
        s_, p_, r, done = env._step(action, goal)
        if done:
            # 一个episode内到达则重置目标点.
            goal = [c, d] + np.random.rand(2) * 2
            print("Goal is: ", goal)
        ddpg.store_transition(s, p, action, r, s_, p_)
        total_step += 1
        if total_step == MEMORY_CAPACITY:
            with open(ddpg.out_dir + '/memory_tree.pkl', 'wb') as fn:
                pickle.dump(ddpg.memory.tree, fn, True)
            print("memory full")
        # state transition
        p = p_.copy()
        s = s_.copy()
        ep_reward += float(sum(r))
        # sum(dis)
        step += 1
        j += 1
        if j % 10 == 0 and total_step >= MEMORY_CAPACITY:  # if j % 10 == 0 and total_step >= MEMORY_CAPACITY:
            ddpg.learn(True)
        if j == MAX_GOAL_STEPS - 1:
            v2 = tf.Summary.Value(tag='ep_reward', simple_value=ep_reward)
            s2 = tf.Summary(value=[v2])
            ddpg.loss_writer.add_summary(s2, i)
            print('Episode:', i+1, 'step: ', step, ' Reward_sum: %.4f' % ep_reward, 'Explore: %.2f' % var)
            reward_list.append(ep_reward)
            break
    if i >= 20: # 20 是滑动窗口的大小. 也就是20次存一次取avg_reward
        print("reward_list[]:",reward_list)
        reward_list.remove(reward_list[0])
        if (i+1) % 20 == 0:
            smma = tf.Summary()
            # arrive_rate = sum(arrive_list) / len(arrive_list)
            avg_reward = sum(reward_list) / len(reward_list)
            # smma.value.add(tag='success_rate', simple_value=arrive_rate)
            smma.value.add(tag='avg_reward', simple_value=avg_reward)
            ddpg.loss_writer.add_summary(smma, i+1)