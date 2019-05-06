# -*- coding:utf-8 -*- -

import rospy
import math
import time
import numpy as np
np.random.seed(1) # 固定随机数种子
from nav_msgs.msg import Odometry
from tf_rl.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose, Quaternion, Vector3, Point
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
# 20181025 输入归一化，考虑局部感知域，比如10m，超过就取10m

from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from visualization_msgs.msg import Marker

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


class DDPGEnv1Go(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/tf_rl/envs/assets/launch/OneGo.launch")
        self.vel_pub = []
        self.max_range_dis = 10
        self.vel_pub.append(rospy.Publisher('/robot0/cmd_vel', Twist, queue_size=5))
        self.setState = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.getState = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.nor = 1
        self.init_pose = []
        self.modelState = ModelState()
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.listen_class = Listen_class(self.nor)
        # self.pong_list = np.zeros(self.nor)  # 因为实际测试的时候可能不能因为碰撞就马上重置

        self.markerPub = rospy.Publisher('/model_marker', Marker, queue_size=10)
        self.init_vel()
        self.markerPub.publish(self.vel)

    def init_vel(self):
        # 此函数作用？
        vel = Marker()
        vel.header.frame_id = "odom"
        vel.header.stamp = rospy.Time.now()
        vel.ns = "bezier"
        vel.action = vel.ADD
        vel.type = Marker.CYLINDER
        vel.id = 0

        vel.pose.position.x = 1
        vel.pose.position.y = 1
        vel.pose.position.z = 0

        vel.scale.x = 0.1
        vel.scale.y = 0.2
        vel.scale.z = 0.7

        vel.pose.orientation.x = 0
        vel.pose.orientation.y = 0
        vel.pose.orientation.z = 0
        vel.pose.orientation.w = 1
        #

        vel.color.a = 1.0
        # markers.color.r = 1.0
        vel.color.g = 0.5
        vel.color.b = 0.5

        self.vel = vel

    def calculate_observation(self, data):  # determine whether there is a collision
        min_range = 0.3
        collision = False
        where_are_inf = np.isinf(data)
        data[where_are_inf] = self.max_range_dis  # 最大距离
        for i, item in enumerate(data):
            if min_range > data[i] > 0:
                collision = True
            data[i] = min(self.max_range_dis, data[i])
        # ranges = np.min(data1.reshape(36, 20), 1) / self.max_range_dis  # 使用36维激光，用最小值保证安全，对应网络应该可以简单点
        ranges = np.mean((np.array(data)).reshape(180, 4), 1) / self.max_range_dis   # 720 dim filter to 180 dim
        return ranges, collision

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cal_theta(self, goal):
        theta = math.atan2(2 * self.listen_class.odom_list[0].pose.pose.orientation.z *
                           self.listen_class.odom_list[0].pose.pose.orientation.w,
                           1 - 2 * (self.listen_class.odom_list[0].pose.pose.orientation.z ** 2)) - \
                math.atan2(goal[1] - self.listen_class.odom_list[0].pose.pose.position.y,
                           goal[0] - self.listen_class.odom_list[0].pose.pose.position.x)
        if theta > 3.14:
            theta -= 6.28
        elif theta < -3.14:
            theta += 6.28
        theta /= 3.14
        return theta

    def _step(self, action, goal, first=False):
        vcmds = []
        vcmds.append(Twist())
        vcmds[0].linear.x = action[0][0]
        vcmds[0].angular.z = action[0][1]
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        if first:
            rospy.sleep(2)
            self.get_all_init_states()
        self.vel_pub[0].publish(vcmds[0])
        rospy.sleep(0.1/10)  # 指令要持续一段时间，注意这是仿真时间，真实时间要考虑仿真速率比
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
        state_list = []
        param_list = []
        reward = []
        done_list = False
        goal_dis = np.sqrt((self.listen_class.odom_list[0].pose.pose.position.x - goal[0]) ** 2 + (self.listen_class.odom_list[0].pose.pose.position.y - goal[1]) ** 2) / self.max_range_dis  # 归一化目标距离
        goal_dis_reward = - goal_dis * 10  # 距离惩罚
        reward.append(goal_dis_reward)
        state, collision = self.calculate_observation(np.array(self.listen_class.range_list[0]))
        state_list.append(state)

        vw = self.listen_class.odom_list[0].twist.twist.angular.z
        vl = self.listen_class.odom_list[0].twist.twist.linear.x
        theta = self.cal_theta(goal)
        param_list.append(np.array([goal_dis, vw, vl, theta]))  # nor + 3
        # goal_dis = goal_matrix[i, i]
        # goal_dis = np.min(goal_matrix[i, :])  # 得到每个机器人离最近目标点的距离
        if not (done_list or collision):
            # reward_list.append(-(goal_dis/3) ** 2)
            # reward_list.append(goal_dis_sum_reward)
            pass
        elif collision:
            # 碰撞处理只处理单个持续碰撞才重置
            reward[0] -= 1
            self.resetState(0)
        if goal_dis < 0.3/self.max_range_dis:
            done_list = True
            reward[0] += 1
        return np.array(state_list), np.array(param_list), np.array(reward), done_list

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
