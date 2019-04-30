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
from geometry_msgs.msg import Pose
import tf

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

# This env for the multi-DDPG with the
class formDDPGEnv(gazebo_env.GazeboEnv):
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
        # self.vel_pub.append(rospy.Publisher('/robot4/cmd_vel', Twist, queue_size=5))
        # self.vel_pub.append(rospy.Publisher('/robot5/cmd_vel', Twist, queue_size=5))
        # self.vel_pub.append(rospy.Publisher('/robot6/cmd_vel', Twist, queue_size=5))
        # self.vel_pub.append(rospy.Publisher('/robot7/cmd_vel', Twist, queue_size=5))
        self.setState = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.getState = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.nor = 4  # 8
        self.init_pose = []
        self.modelState = ModelState()
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.listen_class = Listen_class(self.nor)
        self.InitLenList = []
        self.YawList = []  # 转换函数得到的欧拉角是弧度制表示的
        self.form_mat = np.zeros([4, 4])  # The form_mat is built for the form observation.
        self.pong_list = np.zeros(self.nor)  # 因为实际测试的时候可能不能因为碰撞就马上重置

        self.markerPub = rospy.Publisher('/model_marker', Marker, queue_size=10)
        # self.init_vel()
        # self.markerPub.publish(self.vel)

    # def init_vel(self):
    #
    #     vel = Marker()
    #     vel.header.frame_id = "odom"
    #     vel.header.stamp = rospy.Time.now()
    #     vel.ns = "bezier"
    #     vel.action = vel.ADD
    #     vel.type = Marker.CYLINDER
    #     vel.id = 0
    #
    #     vel.pose.position.x = 1
    #     vel.pose.position.y = 1
    #     vel.pose.position.z = 0
    #
    #     vel.scale.x = 0.1
    #     vel.scale.y = 0.2
    #     vel.scale.z = 0.7
    #
    #     vel.pose.orientation.x = 0
    #     vel.pose.orientation.y = 0
    #     vel.pose.orientation.z = 0
    #     vel.pose.orientation.w = 1
    #     #
    #
    #     vel.color.a = 1.0
    #     # markers.color.r = 1.0
    #     vel.color.g = 0.5
    #     vel.color.b = 0.5
    #
    #     self.vel = vel

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
        ranges = np.mean((np.array(data1)).reshape(180, 4), 1) / self.max_range_dis # 720*1 to 180*1,间隔4维度求平均使用.相当于对激光数据简单均值滤波以后再用.
        return ranges, pong

    def reward_fun(self, dis):
        # 小正代码
        dis_sum = 0
        while(dis.size>1):
            dis_sum += (np.min(dis)/5)**2
            index = np.argmin(dis)
            r, c = index/dis.shape[0], index%dis.shape[0]
            dis = np.delete(dis, r, 0)
            dis = np.delete(dis, c, 1)
        dis_sum += (dis[0, 0]/5)**2
        return dis_sum

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def CalDist(self, position1, position2):
        LenDist = math.sqrt((position1.x - position2.x) * (position1.x - position2.x) + (position1.y - position2.y) * (position1.y - position2.y))
        return LenDist

    def GetLenList(self):
        # 将任务环境定义为指定出事队形后,在保持队形的前提下避障运行到指定地点.
        # 各个机器人朝向角的误差和，保持在一个统一的范围,即根据角度偏差给Reward.
        # FormDist_reward = [sum of point[i]: abs_sum(point i to other points - The init distance) for i in range(self.nor)]
        # 保持队型的核心,在于各个智能体连边比例保持不变.这样才能训练出动态缩放的蔽障可能.
        Len_list = []
        for i in range(self.nor-1):
            for j in range(i+1, self.nor):
                Len = self.CalDist(self.listen_class.odom_list[i].pose.pose.position, self.listen_class.odom_list[j].pose.pose.position)
                Len_list.append(Len)
        return Len_list # 1*6排列组合得到的连边集合

    def GetEuler(self, RobotNum):   # 位姿信息roll, pitch, yaw
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([self.listen_class.odom_list[RobotNum].pose.pose.orientation.x, self.listen_class.odom_list[RobotNum].pose.pose.orientation.y, self.listen_class.odom_list[RobotNum].pose.pose.orientation.z, self.listen_class.odom_list[RobotNum].pose.pose.orientation.w])
        return roll, pitch, yaw

    def GetFormMat(self):
        # Init the form_mat, get the distance between each robot.
        for i in range(self.nor):
            for j in range(self.nor):
                if i == j:
                    self.form_mat[i][j] = 0
                else:
                    self.form_mat[i][j] = self.CalDist(self.listen_class.odom_list[i].pose.pose.position, self.listen_class.odom_list[j].pose.pose.position) / self.max_range_dis  # 归一化的好处？
        # print("self.form_mat:",self.form_mat)
        return self.form_mat

    def CalTheta(self, goal, i, j):
        # theta为机器人智能体与某个目标点的夹角
        theta = math.atan2(2 * (self.listen_class.odom_list[i].pose.pose.orientation.z * self.listen_class.odom_list[i].pose.pose.orientation.w + self.listen_class.odom_list[i].pose.pose.orientation.x * self.listen_class.odom_list[i].pose.pose.orientation.y), 1 - 2 * (self.listen_class.odom_list[i].pose.pose.orientation.x ** 2 + self.listen_class.odom_list[i].pose.pose.orientation.z ** 2)) - math.atan2(goal[j][1] - self.listen_class.odom_list[i].pose.pose.position.y, goal[j][0] - self.listen_class.odom_list[i].pose.pose.position.x)
        if theta > 3.14:
            theta -= 6.28
        elif theta < -3.14:
            theta += 6.28
        theta /= 3.14
        return theta

    def _step(self, action, goal, first=False):
        # 单下划线的方式便于跳转到函数.不加下划线能够直接调用,因为当前类的成员函数能够覆盖同名的父类成员函数.
        # print("Here we use DRL_Nav formation env.")
        vcmds = []
        for i in range(self.nor):  # num of robots
            vcmds.append(Twist())   # 简易差分驱动机器人,只考虑x方向线速度和z方向角速度.
            vcmds[i].linear.x = action[i][0]
            vcmds[i].angular.z = action[i][1]
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        if first: # Init step for robots.
            rospy.sleep(0.1)
            self.get_all_init_states()
            self.InitLenList = self.GetLenList() # 获取初始队形连边比
            print("InitLenList:", self.InitLenList)
            for i in range(self.nor):
                _, _, Yaw = self.GetEuler(i)
                self.YawList.append(Yaw)
            # print("Yawlist:", self.YawList)

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
        reward_list = []
        pong_list = []
        # pong_count = 0
        # Restart = False
        dis_matrix = np.zeros([self.nor, self.nor])     # The distance mat to each goal point.
        for i in range(self.nor):
            state, pong = self.calculate_observation(np.array(self.listen_class.range_list[i]))
            state_list.append(state)
            done_list.append(False)
            # pong_list.append(pong)
            for j in range(self.nor):  # 代表nor个目标点
                # 计算当前机器人距离目标点的位置距离.
                goal_dis = np.sqrt((self.listen_class.odom_list[i].pose.pose.position.x - goal[j][0]) ** 2
                                   + (self.listen_class.odom_list[i].pose.pose.position.y - goal[j][1]) ** 2)
                # The distance between robot i and goal j.
                dis_matrix[i][j] = min(goal_dis, self.max_range_dis) / self.max_range_dis  # 归一化（0,1）后存入matrix
            # 差分机器人只用了这两个量在控制.
            vw = self.listen_class.odom_list[i].twist.twist.angular.z
            vl = self.listen_class.odom_list[i].twist.twist.linear.x
            # # # 队形保持Reward, sum|各连边比值-初始比值|
            FormReward = np.sum(abs(np.array(self.GetLenList()) / np.array(self.InitLenList) - 1))
            self.form_mat = self.GetFormMat()
            theta = self.CalTheta(goal, i, i) # 计算第i个智能体到第i个目标点的夹角.

            param_list.append(np.array([vw, vl, theta]))   # 3
            param_list[i] = np.append(param_list[i], self.form_mat[i,:])  # nor +3
            # 所有Reward的处理放在碰撞检测之后.
            if not done_list[i] and not pong:
                reward_list.append(0)
                self.pong_list[i] = 0
            elif pong:
                # 碰撞处理只处理单个持续碰撞才重置
                reward_list.append(-5)
                self.pong_list[i] += 1
                if self.pong_list[i] >= 10:
                    # self.resetState(i)  # 重置出问题的那个
                    self._reset()   # 集体重置
                    self.pong_list[i] = 0
            else:
                self._reset()
            #  防止机器人以极低的速度运行
            if abs(vcmds[i].linear.x) < 0.5:
                reward_list[i] -= 0.3
            if abs(vcmds[i].angular.z) > 0.6:   # 转向太快的惩罚？
                # print("vcmds[i].angular.z:", vcmds[i].angular.z)
                reward_list[i] -= 2 * abs(vcmds[i].angular.z)
            if FormReward < 1.2:    # 6个连边,平均每个比例误差0.2
                reward_list[i] += 0.8
            else:
                reward_list[i] -= 1
                # Restart = True
            # 方向保持Reward,对于过分偏离队伍大方向机器人进行惩罚.
            AvgYaw = np.sum(self.YawList) / self.nor
            _, _, yaw = self.GetEuler(i)
            self.YawList[i] = yaw
            if abs(yaw - AvgYaw) > 0.17453 * 2:   # pi/180 * 10 偏离队伍均值10°以上就扣分
                reward_list[i] -= 0.3
            else:
                reward_list[i] += 0.3
            # 根据距离信息粗略给定一个动态Reward.
            if dis_matrix[i].min() < 2 and dis_matrix[i].min() >= 1.5:
                reward_list[i] -= 0.5
            elif dis_matrix[i].min() < 1.5 and dis_matrix[i].min() >= 1:
                reward_list[i] -= 1
            elif dis_matrix[i].min() < 1 and dis_matrix[i].min() >= 0.5:
                reward_list[i] -= 1.5
            elif dis_matrix[i].min() < 0.5:
                reward_list[i] -= 2

            #  到达检测
            if dis_matrix[i, i] < 0.03:
                done_list[i] = True
                reward_list[i] += 3
            # goal_reward = sum(dis_matrix.min(axis=0)) / self.nor
        # The parameter 5 is the distance amplify factor.
        # sum(dis_matrix.min(axis=0))/self.nor about [0.3,0.8]
        #   reward_list[i] += (2.3 - sum(dis_matrix[i]) / self.nor)
            reward_list[i] -= sum(dis_matrix[i]) / self.nor # 距离越远惩罚越大  # sum(dis_matrix[i]) / self.nor(2+2+2.828)/3 = 2.3 为到达状态
        # print("goal_reward: ",goal_reward)
        # reward_array = np.array(reward_list) + (1 - goal_reward) * 3
        # reward_array = np.array(reward_list)
        # total_toc = time.time()
        # print('time is %f'%(total_toc-total_tic))
        return np.array(state_list), np.array(param_list) , np.array(reward_list), done_list

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