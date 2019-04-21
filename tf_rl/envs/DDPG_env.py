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


class DDPGEnv(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/pygmalionchen/PycharmProjects/TensorflowPrj/DRL_Nav/tf_rl/envs/assets/launch/formation.launch")
        self.vel_pub = []
        # /home/szz/tf_rl/tf_rl/envs/assets/launch/one_robot_world.launch
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
        self.init_pose = []
        self.modelState = ModelState()
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.listen_class = Listen_class(self.nor)
        self.pong_list = np.zeros(self.nor)  # 因为实际测试的时候可能不能因为碰撞就马上重置

        self.markerPub = rospy.Publisher('/model_marker', Marker, queue_size=10)
        self.init_vel()
        self.markerPub.publish(self.vel)

    def init_vel(self):
        # 此函数作用
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

    def calculate_observation(self, data1):  # determine whether there is a collision
        min_range = 0.3
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

    def reward_fun(self, dis):
        dis_sum = 0
        while(dis.size>1):
            dis_sum += (np.min(dis)/5)**2   # 为何吧最小值平方
            index = np.argmin(dis)
            r, c = index/dis.shape[0], index%dis.shape[0]
            dis = np.delete(dis, r, 0)
            dis = np.delete(dis, c, 1)
        dis_sum += (dis[0, 0]/5)**2
        return dis_sum

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cal_theta(self, goal, i, j):
        theta = math.atan2(2 * self.listen_class.odom_list[i].pose.pose.orientation.z *
                           self.listen_class.odom_list[i].pose.pose.orientation.w,
                           1 - 2 * (self.listen_class.odom_list[i].pose.pose.orientation.z ** 2)) - \
                math.atan2(goal[j][1] - self.listen_class.odom_list[i].pose.pose.position.y,
                           goal[j][0] - self.listen_class.odom_list[i].pose.pose.position.x)
        if theta > 3.14:
            theta -= 6.28
        elif theta < -3.14:
            theta += 6.28
        theta /= 3.14
        return theta

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
            rospy.sleep(2)
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
        param_list = []
        done_list = []
        goal_matrix = np.zeros([self.nor, self.nor])
        formation_list = []
        # pong_list = []
        for i in range(self.nor):
            for j in range(self.nor):  # 对目标点i，计算离它最近的机器人j的距离
                goal_matrix[i][j] = min(self.max_range_dis,
                                        np.sqrt((self.listen_class.odom_list[i].pose.pose.position.x - goal[j][0]) ** 2
                                                + (self.listen_class.odom_list[i].pose.pose.position.y - goal[j][1]) ** 2))
        goal_matrix = goal_matrix / self.max_range_dis  # 归一化（0,1）
        goal_dis_sum_reward = - self.reward_fun(goal_matrix.copy())  # 距离和，每个机器人都要加上的reward
        reward_list = [goal_dis_sum_reward for _ in range(self.nor)]
        for i in range(self.nor):
            state, pong = self.calculate_observation(np.array(self.listen_class.range_list[i]))
            state_list.append(state)
            goal_dis = goal_matrix[i, i]
            # goal_dis = np.min(goal_matrix[i, :])  # 得到每个机器人离最近目标点的距离
            done_list.append(False)
            if not (done_list[i] or pong):
                # reward_list.append(-(goal_dis/3) ** 2)
                # reward_list.append(goal_dis_sum_reward)
                self.pong_list[i] = 0
            elif pong:
                # 碰撞处理只处理单个持续碰撞才重置
                reward_list[i] -= 1
                self.pong_list[i]+=1
                if self.pong_list[i]>=20:
                    self.resetState(i)
                    self.pong_list[i] = 0

                # call set_model_state
            # reward = reward - (1/min_range-0.2)
            # if abs(vcmds[i].angular.z) > 0.6:
            #     reward_list[i] -= 0.1 * abs(vcmds[i].angular.z)
            # done threshold
            if goal_dis < 0.3/self.max_range_dis:
                done_list[i] = True
                reward_list[i] += 1
                # self.resetState(i)
            #  此部分希望各个机器人能够相互处理角度？
            if i == np.argmax(action[0,:]) or i == np.argmin(action[0,:]) or i == np.argmax(action[1,:]) or i == np.argmin(action[1,:]):
                reward_list[i] -= 0.1
            theta = self.cal_theta(goal, i, i)  # why i i ?
            param_list.append(np.array([goal_dis, theta, self.listen_class.odom_list[i].twist.twist.linear.x,
                                        self.listen_class.odom_list[i].twist.twist.angular.z]))
            if i<3:
                f_theta = self.cal_theta(goal, i, i+5)
                if f_theta >0:
                    f_theta -= 1
                else:
                    f_theta += 1
            elif i>4:
                f_theta = self.cal_theta(goal, i, i-5)
            elif i==3:
                f_theta = self.cal_theta(goal, i, i - 3)
            else:
                f_theta = self.cal_theta(goal, i, i - 2)
            formation_list.append(f_theta)
            if abs(f_theta)<0.1:
                reward_list[i] += 0.1
        # reward_res = reward_fun(goal_matrix)
        # total_toc = time.time()
        # print('time is %f'%(total_toc-total_tic))
        return np.array(state_list), np.array(param_list), reward_list, done_list   #, formation_list  # , self.pong_list
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

        # rospy.wait_for_service('/gazebo/reset_world')
        # try:
        #     self.reset_proxy()
        # except rospy.ServiceException as e:
        #     print("/gazebo/reset_simulation service call failed")
        #
        # # if first:
        # #     rospy.sleep(2)
        # #     self.get_all_init_states()
        #
        # # 随机初始化起点
        # # self.resetState(0)
        # rospy.sleep(0.1 / 10)
        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #     self.unpause()
        # except rospy.ServiceException as e:
        #     print("/gazebo/unpause_physics service call failed")
        # # read laser data
        # for i in range(self.nor):
        #     data1 = None
        #     while data1 is None:
        #         try:
        #             data1 = rospy.wait_for_message('/robot%d/scan'%i, LaserScan, timeout=1)
        #         except:
        #             pass
        #
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     # resp_pause = pause.call()
        #     self.pause()
        # except rospy.ServiceException as e:
        #     print("/gazebo/pause_physics service call failed")
        #
        #     # state, pong = self.calculate_observation(np.array(data1.ranges))
        #     # 经常reset发现读不到激光数据，所以这里用之前的方式while循环要确保
        # print('finish reset')
        # return
