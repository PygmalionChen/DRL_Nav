# -*- coding:utf-8 -*- -

import rospy
import math
import time
import numpy as np

np.random.seed(1)
from nav_msgs.msg import Odometry
from tf_rl.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose, Quaternion, Vector3, Point
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
# 20181025 输入归一化，考虑局部感知域，比如10m，超过就取10m
# 20181113 考虑队形控制，修改碰撞重置机制
from sensor_msgs.msg import LaserScan
from gym.utils import seeding


class Listen_class():
    def __init__(self, nor):
        self.range_list = [[] for _ in range(nor)]
        self.odom_list = [[] for _ in range(nor)]
        for i in range(nor):
            rospy.Subscriber('/robot%d/scan' % i, LaserScan, self.scan_callback, i)
            rospy.Subscriber('/robot%d/odom' % i, Odometry, self.odom_callback, i)

    def scan_callback(self, msg, args):
        self.range_list[args] = msg.ranges

    def odom_callback(self, msg, args):
        self.odom_list[args] = msg


class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name


class GazeboMultiJackalLidarEnv8(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/pygmalionchen/PycharmProjects/treasure/tf_rl/envs/assets/launch/multi_goal.launch")
        self.vel_pub = []
        # /home/pygmalionchen/PycharmProjects/treasure/tf_rl/envs/assets/launch/one_robot_world.launch
        self.max_range_dis = 10
        self.vel_pub.append(rospy.Publisher('/robot0/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot3/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot4/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot5/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot6/cmd_vel', Twist, queue_size=5))
        self.vel_pub.append(rospy.Publisher('/robot7/cmd_vel', Twist, queue_size=5))
        self.setState = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.getState = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.nor = 8
        self.n = self.nor
        self.action_space = [2 for _ in range(self.nor)]
        self.init_pose = []
        self.modelState = ModelState()
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.listen_class = Listen_class(self.nor)
        self.pong_list = np.zeros(self.nor)  # 因为实际测试的时候可能不能因为碰撞就马上重置

    def calculate_observation(self, data1):  # determine whether there is a collision
        min_range = 0.5
        pong = False
        where_are_inf = np.isinf(data1)
        data1[where_are_inf] = self.max_range_dis  # 最大距离
        for i, item in enumerate(data1):
            if min_range > data1[i] > 0:
                pong = True
            data1[i] = min(self.max_range_dis, data1[i])
        # ranges = np.min(data1.reshape(36, 20), 1) / self.max_range_dis  # 使用36维激光，用最小值保证安全，对应网络应该可以简单点
        ranges = np.mean((np.array(data1)).reshape(180, 4), 1) / self.max_range_dis
        return ranges, pong

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action, goal, first=False):
        vcmds = []
        for i in range(self.nor):  # num of robots
            vcmds.append(Twist())
            vcmds[i].linear.x = action[i][0]
            vcmds[i].angular.z = action[i][1]
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        if first:
            rospy.sleep(0.2)
            self.get_all_init_states()
        for i in range(self.nor):
            self.vel_pub[i].publish(vcmds[i])
        rospy.sleep(0.1/10)  # 指令要持续一段时间，注意这是仿真时间，真实时间要考虑仿真速率比
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        state_list = []
        done_list = []
        reward_list = []
        param_list = []
        # pong_list = []
        dis_matrix = np.zeros([self.nor, self.nor])
        for i in range(self.nor):
            state, pong = self.calculate_observation(np.array(self.listen_class.range_list[i]))
            state_list.append(state)
            done_list.append(False)
            param_list.append([])
            for j in range(self.nor):  # 代表nor个目标点
                # pong_list.append(pong)
                goal_dis = np.sqrt((self.listen_class.odom_list[i].pose.pose.position.x - goal[j][0]) ** 2
                                   + (self.listen_class.odom_list[i].pose.pose.position.y - goal[j][1]) ** 2)
                dis_matrix[i][j] = min(goal_dis, self.max_range_dis) / self.max_range_dis  # 归一化（0,1）
                theta = math.atan2(2 * self.listen_class.odom_list[i].pose.pose.orientation.z *
                                   self.listen_class.odom_list[i].pose.pose.orientation.w,
                                   1 - 2 * (self.listen_class.odom_list[i].pose.pose.orientation.z ** 2)) - \
                        math.atan2(goal[j][1] - self.listen_class.odom_list[i].pose.pose.position.y,
                                   goal[j][0] - self.listen_class.odom_list[i].pose.pose.position.x)
                # theta = arctan(delta y/delta x)
                if theta > 3.14:
                    theta -= 6.28
                elif theta < -3.14:
                    theta += 6.28
                theta /= 3.14
                param_list[i].append(dis_matrix[i][j])
                param_list[i].append(theta)

            # if not (done_list[i] or pong):
            #     reward_list.append(-(goal_dis / 3) ** 2)

            if pong:
                reward_list.append(-2)
                self.pong_list[i] += 1
                if self.pong_list[i] >= 20:
                    self.resetState(i)
                    self.pong_list[i] = 0
            else:
                reward_list.append(0)
                self.pong_list[i] = 0
            if abs(vcmds[i].angular.z) > 0.6:
                reward_list[i] -= 0.1 * abs(vcmds[i].angular.z)
            if dis_matrix[i,:].min() < 0.3 / self.max_range_dis:
                done_list[i] = True
                reward_list[i] += 5
                # self.resetState(i)

            # param_list.append(np.array([goal_dis, theta, self.listen_class.odom_list[i].twist.twist.linear.x,
            #                             self.listen_class.odom_list[i].twist.twist.angular.z]))
        goal_reward = sum(dis_matrix.min(axis=0))/self.nor  # 每列的最小值，就是对每个目标点得到的最小距离，求和
        reward_array = np.array(reward_list) - goal_reward
        # total_toc = time.time()
        # print('time is %f'%(total_toc-total_tic))
        return np.array(state_list), np.array(param_list), reward_array, done_list, self.pong_list
        # [xt, yt, q2, q3,q0,q1], [vl, va], {}

    def get_all_init_states(self):  # 在前期调用一次，获取机器人的初始位置
        # 这个循环是为了等所有机器人都加载完
        self.init_pose = []
        for i in range(self.nor):
            self.init_pose.append(self.getState('robot%d' % i, 'world'))
        return self.init_pose

    def resetState(self, number):  # 若发生碰撞，将该robot放置到其初始位置
        self.modelState.model_name = 'robot%d' % number
        self.modelState.pose = self.init_pose[number].pose
        self.setState(self.modelState)

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        return
