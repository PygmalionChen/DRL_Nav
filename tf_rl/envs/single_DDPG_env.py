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


class SingleDDPGEnv(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/pygmalionchen/PycharmProjects/treasure/tf_rl/envs/assets/launch/one_robot_world.launch")
        self.vel_pub = []
        # /home/pygmalionchen/PycharmProjects/treasure/tf_rl/envs/assets/launch/one_robot_world.launch
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
        self.pong_list = np.zeros(self.nor)  # 因为实际测试的时候可能不能因为碰撞就马上重置

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

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action, goal):
        goal_dis_old = np.sqrt((self.listen_class.odom_list[0].pose.pose.position.x - goal[0][0]) ** 2
                               + (self.listen_class.odom_list[0].pose.pose.position.y - goal[0][
            1]) ** 2) / self.max_range_dis
        vel_cmd = Twist()
        # if action<5:
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        for i in range(self.nor):
            self.vel_pub[i].publish(vel_cmd)
        rospy.sleep(0.1 / 10)  # 指令要持续一段时间，注意这是仿真时间，真实时间要考虑仿真速率比
        # 更正，这个就是实际时间，在仿真中就是实际时间X仿真倍率，郁闷，所以手动调整吧
        # 目前就按10Hz的控制频率来吧，那么10倍速率就是0.01s one step
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
        # dis_matrix = np.zeros([self.nor, self.nor])
        for i in range(self.nor):
            state, pong = self.calculate_observation(np.array(self.listen_class.range_list[i]))
            state_list.append(state)
            done_list.append(False)
            param_list.append([])
            for j in range(self.nor):  # 代表nor个目标点
                # pong_list.append(pong)
                goal_dis = np.sqrt((self.listen_class.odom_list[i].pose.pose.position.x - goal[j][0]) ** 2
                                   + (self.listen_class.odom_list[i].pose.pose.position.y - goal[j][
                    1]) ** 2) / self.max_range_dis
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
                param_list[i].append(goal_dis)
                param_list[i].append(theta)
                param_list[i].append(self.listen_class.odom_list[i].twist.twist.linear.x)
                param_list[i].append(self.listen_class.odom_list[i].twist.twist.angular.z)

            # if not (done_list[i] or pong):
            reward_list.append((goal_dis_old-goal_dis)*10-0.002)

            if pong:
                reward_list[i] -= 1
            # if abs(vel_cmd.angular.z) > 0.6:
            #     reward_list[i] -= 0.1 * abs(vel_cmd.angular.z)
            # if dis_matrix[i,:].min() < 0.85 / self.max_range_dis:
            if goal_dis < 0.8 / self.max_range_dis:
                done_list[i] = True
                reward_list[i] += 1

        return np.array(state_list[0]), np.array(param_list[0]), reward_list[0], done_list[0], pong  # , [vl, va]
        # [xt, yt, q2, q3,q0,q1], [vl, va], {}

    def get_all_init_states(self):  # 在前期调用一次，获取机器人的初始位置
        # 这个循环是为了等所有机器人都加载完
        self.init_pose = []
        for i in range(self.nor):
            self.init_pose.append(self.getState('robot%d' % i, 'world'))
        return self.init_pose

    def resetState(self, number):  # 若发生碰撞，将该robot放置到其初始位置
        self.modelState.model_name = 'robot%d' % number
        self.init_pose[number].pose.position.x, self.init_pose[number].pose.position.y = (np.random.rand(2) - 0.5) * 11
        self.modelState.pose = self.init_pose[number].pose
        self.setState(self.modelState)

    def _reset(self, first=False):
        # Resets the state of the environment and returns an initial observation.

        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        if first:
            rospy.sleep(2)
            self.get_all_init_states()

        # 随机初始化起点
        # self.resetState(0)
        rospy.sleep(0.1/10)
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        # read laser data
        data1 = None
        while data1 is None:
            try:
                data1 = rospy.wait_for_message('/robot0/scan', LaserScan, timeout=1)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        state, pong = self.calculate_observation(np.array(data1.ranges))
        # 经常reset发现读不到激光数据，所以这里用之前的方式while循环要确保
        return state, self.init_pose[0].pose.position.x, self.init_pose[0].pose.position.y
