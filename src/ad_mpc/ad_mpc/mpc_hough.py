# Vivek Alumootil
# Last updated: 10/15/24

# INSTRUCTIONS
# source install/setup.bash (from agronav_ws_temp)
# ros2 run ad_mpc mpc

# DETAILS
# [a, b, c] <=> ax+by+c=0

# SETTINGS
# MPC SETTINGS:
hor = 7 # horizon length 
ts = 0.10 # timestep

import rclpy
import sys

# adding path to this directory

from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
sys.path.append('/home/agribot2/Documents/Scout_Robot/scout2_ws/src/ad_mpc/ad_mpc')

import time
from os.path import isfile, join

import cv2
import numpy as np

# mpc libraries
import math
from controller import Robot
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# if these imports fail, please check the environment variable PYTHONPATH

# Initialize Robot
robot = Robot(ts, hor, np.array([0, 0, np.pi/2]))

# Homography matrix
H = np.load('/home/agribot2/camera-info/homography.npy')  

# from the line endpoint coordinates in the image system, generate two pixel coordinates that lay on the ground
def get_ground_level_points(x0, y0, x1, y1):
    return np.array([[x0*0.95+x1*0.05, y0*0.95+y1*0.05], [x0*0.9+x1*0.1, y0*0.9+y1*0.1]])

def line_from_points(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return np.array([y1-y2, x2-x1, x1*(y1+y2)-y1*(x1+x2)])

def get_midline(L1, L2):
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    coeff_matrix = np.array([[A1, B1], [A2, B2]])
    constants = np.array([-C1, -C2])
    
    intersection = np.linalg.solve(coeff_matrix, constants)
    x1, y1 = intersection
    slope1 = np.arctan(-A1/B1)
    slope2 = np.arctan(-A2/B2)
    if (slope1 < 0):
        slope1 += np.pi
    if (slope2 < 0):
        slope2 += np.pi
    slope_new = np.tan((slope1+slope2)/2)
    L3 = np.array([slope_new, -1, y1-slope_new*x1])
    return L3/slope_new
    
def get_xy(imc):
    y = np.dot(H, np.append(imc, [1]))
    y = y/y[2]
    return y[:2]

def unique_line(lines, l):
    for line in lines:
        if (abs(line[0][1]-l[0][1]) < 0.2):
            return False
    return True 

# camera feed of the ROS topic  
camera_name = '/webcam_image'

# output feed topic name
output_name = '/agronav'
visual_name = '/agronav_visual'

class Hough(Node):
    def __init__(self):
        super().__init__('model')
        self.subscription = self.create_subscription(
            Image,
            camera_name,
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, output_name, 1)
        self.publisher_aud = self.create_publisher(Twist, '/cmd_vel', 1)
        # instrs to send to controller
        self.m_queue = []
        # adjustable (should be the same as MPC timestep)
        timer_period = ts
        self.timer = self.create_timer(timer_period, self.mpc_callback)
        self.bridge = CvBridge()
        self.camera_feed = None

    def mpc_callback(self):

        time1 = time.time()
        cv_image = self.camera_feed
        if (cv_image is None):
            return

        msg = Twist()
        mov = np.array([0.0, 0.0])
        if (len(self.m_queue) > 0):
            mov = self.m_queue.pop(0)
        else:
            self.get_logger().info(f'Error: Queue is empty')
        msg.linear.x = float(mov[0])
        msg.angular.z = float(2.56*mov[1])
        # self.publisher_aud.publish(msg)
        self.get_logger().info(f'Publishing: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

        cv2.imwrite('raw_image.jpg', cv_image)

        median_image = cv2.medianBlur(cv_image, 3)
        cv2.imwrite('median.jpg', median_image)

        canny = cv2.Canny(median_image, 50, 200, None, 3)
        cv2.imwrite('canny.jpg', canny)

        cdst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        height, width = canny.shape
        # canny[0:int(height/5), :] = 0

        lines = cv2.HoughLines(canny, 3, np.pi / 180, 150, None, 0, 0)
        true_lines = []
        true_lines_hough = []
        slopes = []

        first = 0;
        if lines is not None:
            for i in range(0, len(lines)):
                if (len(true_lines) == 2):
                    break
                l = lines[i]
                if (unique_line(true_lines_hough, l)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    # pt1[0], -pt1[1]
                    # pt2[0], -pt2[1]
                    slope = 0
                    if (pt2[0] == pt1[0]):
                        slope = 10000
                    else:
                        slope = (-pt2[1]+pt1[1])/(pt2[0]-pt1[0])
                    if (abs(slope) < 0.2 or abs(slope) > 10):
                        continue
                    if (len(true_lines) == 1 and abs(slopes[0]/slope) == slopes[0]/slope):
                        continue
                    true_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                    true_lines_hough.append(lines[i])
                    slopes.append(slope)
                    cv2.line(canny, pt1, pt2, (127,255,255), 2, cv2.LINE_AA)

        cv2.imwrite('hough-no.jpg', canny)

        if (len(true_lines) != 2):
            self.get_logger().info(f'Inference error: {len(true_lines)} lines were detected')
            return
        else:
            self.get_logger().info(f'Inference detected 2 lines')

        x0, y0, x1, y1 = true_lines[0]
        x2, y2, x3, y3 = true_lines[1]
      
        # self.get_logger().info(f'Image coordinates of lines: ({x0}, {y0}), ({x1}, {y1}) and ({x2}, {y2}), ({x3}, {y3})')
        
        const0 = (160-y1)/(y0-y1)
        const1 = (200-y1)/(y0-y1)
        const2 = (160-y3)/(y2-y3)
        const3 = (200-y3)/(y2-y3)

        p0 = np.array([x0, y0])
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([x3, y3])
       
        q0 = p0*const0+p1*(1-const0)
        q1 = p0*const1+p1*(1-const1)
        q2 = p2*const2+p3*(1-const2)
        q3 = p2*const3+p3*(1-const3)

        cv2.circle(canny, (int(q0[0]), int(q0[1])), radius=10, color=(128, 255, 0), thickness=-1)
        cv2.circle(canny, (int(q1[0]), int(q1[1])), radius=10, color=(255, 128, 0), thickness=-1)
        cv2.circle(canny, (int(q2[0]), int(q2[1])), radius=10, color=(128, 128, 0), thickness=-1)
        cv2.circle(canny, (int(q3[0]), int(q3[1])), radius=10, color=(0, 128, 0), thickness=-1)

        W1 = get_xy(q0)
        W2 = get_xy(q1)
        W3 = get_xy(q2)
        W4 = get_xy(q3)

        print(q0)
        print(q1)
        print(q2)
        print(q3)

        cv2.imwrite('hough.jpg', canny)
        self.get_logger().info(f'World Coordinates (XY plane): {W1}, {W2}, {W3}, {W4}')
        L1 = line_from_points(W1, W2)
        L2 = line_from_points(W3, W4)

        self.get_logger().info(f'World Lines (XY plane): {L1}, {L2}')

        goal = get_midline(L1, L2)
        self.get_logger().info(f'Center Line (XY plane): {goal}')

        initial_pose = np.array([0, 0, np.pi/2])
        # forward = robot.forward(initial_pose, mov)  
        forward = robot.forward(initial_pose, np.array([0, 0]))  
        opt = robot.optimize(initial_pose, goal)
        
        opt_sol = opt.x.reshape((hor, 2))
        self.m_queue = []
        for i in range(3):
            self.m_queue.append(opt_sol[i])

        time2 = time.time()
        self.get_logger().info(f'Time difference: {time2-time1}')
        
        msg = self.bridge.cv2_to_imgmsg(cdst, 'bgr8')
        self.publisher_.publish(msg)

    def listener_callback(self, msg):
        try:
            self.camera_feed = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error('CvBridge Error: {}'.format(e))

def main(args=None):

    rclpy.init(args=args)
    ag_model = Hough()
    rclpy.spin(ag_model)
    ag_model.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
