import gym
import rospy
import roslaunch
import time
import math
import numpy as np
import cv2

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties, SetModelState
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboTurtlebotMazeColorEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "MazeColor.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=11)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        self.name_target = 'Hint'

        self.target_pos = [[-0.25, -2], [6.5, -1.75], [6.5, 1.75], [3.25, 3.5], [8.25, 1.5]]
        self.check_point = [False] * len(self.target_pos)
        self.set_model = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        
        self.set_target()

        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
        
        rospy.wait_for_service('gazebo/get_model_state')
        try:
            self.turtlebot_state = self.model_state(self.name_model, "world")
        except rospy.ServiceException, e:
            print ("/gazebo/get_model_state service call failed")

        self.channel = 3
        self.width = 84
        self.height = 84
        self.num_action = 11
        # self.num_state = [[84, 84, 3], 100, 2]
        self.num_state = [100]

        self._seed()

    def set_target(self):
        for i, pos in enumerate(self.target_pos):
            rospy.wait_for_service('gazebo/set_model_state')
            try:
                state = ModelState()
                state.model_name = self.name_target + str(i)
                state.reference_frame = "world"
                state.pose.position.x = pos[0]
                state.pose.position.y = pos[1]
                self.set_model(state)

            except rospy.ServiceException, e:
                print ("/gazebo/set_model_state service call failed")

    def calculate_observation(self, data, pos):
        min_range = 0.21
        min_range_target = 0.41
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True

        return done

    def calculate_distance(self, x, y, a, b):
        return math.sqrt((x-a)*(x-a) +(y-b) * (y-b))

    def calculate_reward(self, done, pos):

        min_range = 0.3
        min_range_target = 0.4
        reward = 1
        if(done):
            reward = -200
        else:
            for i, tpos in enumerate(self.target_pos):
                dist2target = self.calculate_distance(pos.x, pos.y, tpos[0], tpos[1])
                if self.check_point[i] == False and dist2target < min_range_target:
                    reward = 100
                    self.check_point[i] = True

        return reward

    def discretize_observation(self,data):
        image_data = None
        cv_image = None
        n = 0
        while image_data is None:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            except:
                n += 1
                if n == 10:
                    print ("Camera error")
                    state = []
                    done = True
                    return state
        cv_image = cv2.resize(cv_image, (self.height, self.width))
        observation = []
        mod = len(data.ranges)/100
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    observation.append(21)
                elif np.isnan(data.ranges[i]):
                    observation.append(0)
                else:
                    observation.append(data.ranges[i])
        return cv_image, observation


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")

        #######Action defination######
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        max_ang_speed = 0.3
        ang_vel = (action-5)*max_ang_speed*0.5 
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        ########Getting state#########
        pos = self.model_state(self.name_model, "world").pose.position
        done = self.calculate_observation(data, pos)
        image, laser = self.discretize_observation(data)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")
        

        ########Reward calculation#####
        reward = self.calculate_reward(done, pos)
        # print reward

        # return np.reshape(np.array([image, np.asarray(laser), np.asarray([pos.x, pos.y], dtype = np.float32)]), [1, 3]), reward, done, {}
        return np.asarray(laser), reward, done, {}

    def _reset(self):
        # print("Resetting ...")
        rospy.wait_for_service('gazebo/set_model_state')
        try:
            state = ModelState()
            state.model_name = self.name_model
            state.reference_frame = "world"
            state.pose= self.turtlebot_state.pose
            state.twist = self.turtlebot_state.twist

            self.set_model(state)
        except rospy.ServiceException, e:
            print ("/gazebo/set_model_state service call failed")

        self.check_point = [False] * len(self.target_pos)

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        pos = self.model_state(self.name_model, "world").pose.position
        done = self.calculate_observation(data, pos)
        image, laser = self.discretize_observation(data)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")

        # return np.reshape(np.array([image, np.asarray(laser), np.asarray([pos.x, pos.y], dtype = np.float32)]), [1, 3])
        return np.asarray(laser)