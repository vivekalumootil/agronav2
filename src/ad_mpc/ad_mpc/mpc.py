# Last updated: 5/2/25

# HYPERPARAMETERS
hor = 20 # horizon length 
ts = 0.2 # timestep

import rclpy
import sys

from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# necessary to get certain python packages
# this is necessary to get certain python packages and modules
sys.path.append('/home/agribot2/Documents/Agronav/venv/lib/python3.8/site-packages')
sys.path.append('/home/agribot2/scout2/src/ad_mpc/ad_mpc')
sys.path.append('/home/agribot2/Documents/Agronav/agronav/lineDetection')
sys.path.append('/home/agribot2/Documents/Agronav/agronav')

import time
from os.path import isfile, join
import PIL

import cv2
import numpy as np
import torch
import torch.optim
from torchvision import transforms
import yaml
import scipy.linalg as lin
from skimage.measure import label, regionprops

# mpc libraries
import math
from controller import Robot
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

# if these imports fail, please check the environment variable PYTHONPATH
from dataloader import get_loader
from logger import Logger
from model.network import Net
from utils import reverse_mapping, edge_align, get_boundary_point, visualize

# root_path is the path to the agronav directory
root_path = '/home/agribot2/Documents/Agronav/agronav/'

# model_path is the path to the DHT checkpoint
model_path = '/home/agribot2/checkpoints/chewy200-jan26th.pth'

CONFIGS = yaml.full_load(open(root_path+'lineDetection/config.yml'))

# Initialize Robot
robot = Robot(ts, hor, np.array([0, 0, 0]))

# Homography matrix
# This assumes that the robot is flat (not tilted upwards or downwards)
H = np.load('/home/agribot2/camera-info/homography.npy')  

# unnecessary code -- will be removed eventually
def get_ground_level_points(x0, y0, x1, y1):
    return np.array([[x0*0.95+x1*0.05, y0*0.95+y1*0.05], [x0*0.9+x1*0.1, y0*0.9+y1*0.1]])

# in general, the line [a, b, c] represents ax+by+c=0
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
   
# this converts an image coordate [x, y, 1] (OpenCV convention) to the coordinate (x, y) on the 2D ground plane relative to the robot
def get_xy(imc):
    y = np.dot(H, np.append(imc, [1]))
    y = y/y[2]
    return y[:2]

# load model
model = Net(numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], backbone=CONFIGS["MODEL"]["BACKBONE"])
model = model.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])
checkpoint = torch.load(model_path)
if 'state_dict' in checkpoint.keys():
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
transform = transforms.Compose([
transforms.Resize((400, 400)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# camera feed of the ROS topic  
camera_name = '/webcam_image'

# output feed topic name
output_name = '/agronav'

class Model(Node):
    def __init__(self):
        super().__init__('model')
        self.subscription = self.create_subscription(
            Image,
            camera_name,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Image, output_name, 1)
        self.publisher_aud = self.create_publisher(Twist, 'cmd_vel', 1)
        # movement instructions to send to controller
        # [v1, v2], v1 is linear velocity, v2 is angular velocity (metric)
        self.u_queue = np.array([0, 0])
        self.m_queue = []
        timer_period = ts
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()

    def timer_callback(self):
        msg = Twist()
        mov = np.array([0.0, 0.0])
        if (len(self.m_queue) > 0):
            mov = self.m_queue.pop(0)
        else:
            self.get_logger().info(f'Error: Queue is empty')
        # need to convert metric units to units the robot understands
        msg.linear.x = float(mov[0])
        msg.angular.z = float(2.56*mov[1])

        self.publisher_aud.publish(msg)
        self.get_logger().info(f'NOTE: Publishing linear.x={msg.linear.x}, angular.z={msg.angular.z}')

    def listener_callback(self, msg):
        try:    
            with torch.no_grad():
                time0 = time.time()
                # Convert ROS Image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.get_logger().info(f'NOTE: Writing camera output to camera.jpg')
                cv2.imwrite('camera.jpg', cv_image)
                img_height, img_width, img_channels = cv_image.shape
                size = [img_height, img_width]
                
                # Inference
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

                img_pil = PIL.Image.fromarray(cv_image)
                img_pil = transform(img_pil)
                images = torch.stack([img_pil])
                images = images.cuda(device=CONFIGS["TRAIN"]["GPU_ID"])

                key_points = model(images)
                key_points = torch.sigmoid(key_points)

                binary_kmap = key_points.squeeze().cpu().numpy() > CONFIGS['MODEL']['THRESHOLD']
                kmap_label = label(binary_kmap, connectivity=1)
                props = regionprops(kmap_label)
                plist = []
                for prop in props:
                    plist.append(prop.centroid)

                size = (size[0], size[1])
                b_points = reverse_mapping(plist, numAngle=CONFIGS["MODEL"]["NUMANGLE"], numRho=CONFIGS["MODEL"]["NUMRHO"], size=(400, 400))
                scale_w = size[1] / 400
                scale_h = size[0] / 400
                for i in range(len(b_points)):
                    y1 = int(np.round(b_points[i][0] * scale_h))
                    x1 = int(np.round(b_points[i][1] * scale_w))
                    y2 = int(np.round(b_points[i][2] * scale_h))
                    x2 = int(np.round(b_points[i][3] * scale_w))
                    if x1 == x2:
                        angle = -np.pi / 2
                    else:
                        angle = np.arctan((y1-y2) / (x1-x2))
                    (x1, y1), (x2, y2) = get_boundary_point(y1, x1, angle, size[0], size[1])
                    b_points[i] = (y1, x1, y2, x2)

                time1 = time.time()
                self.get_logger().info(f'NOTE: Inference took: {time1-time0}s')
                cv2_image_mapping = visualize(b_points, cv_image)
        
                self.get_logger().info(f'NOTE: Writing mapping to dht.jpg')
                cv2.imwrite('dht.jpg', cv2_image_mapping)

                if (len(b_points) < 2):
                    # model failure
                    self.get_logger().info(f'ERROR: {len(b_points)} lines were detected')
                    return
                elif (len(b_points) > 2):
                    # model failure, but persist
                    self.get_logger().info(f'WARNING: {len(b_points)} lines were detected')
                    b_points = b_points[:2]
                else:
                    self.get_logger().info(f'SUCCESS: {len(b_points)} lines were detected')

                y0, x0, y1, x1 = b_points[0]
                y2, x2, y3, x3 = b_points[1]
              
                # self.get_logger().info(f'Image coordinates of lines: ({x0}, {y0}), ({x1}, {y1}) and ({x2}, {y2}), ({x3}, {y3})')

                g0, g1 = 0, 0
                if (y1 < y0):
                    g0 = get_ground_level_points(x0, y0, x1, y1)
                else:
                    g0 = get_ground_level_points(x1, y1, x0, y0)

                if (y3 < y2):
                    g1 = get_ground_level_points(x2, y2, x3, y3)
                else:
                    g1 = get_ground_level_points(x3, y3, x2, y2)
              
                W1 = get_xy(g0[0])
                W2 = get_xy(g0[1])
                W3 = get_xy(g1[0])
                W4 = get_xy(g1[1])

                # self.get_logger().info(f'World Coordinates (XY plane): {W1}, {W2}, {W3}, {W4}')
                L1 = line_from_points(W1, W2)
                L2 = line_from_points(W3, W4)

                # self.get_logger().info(f'World Lines (XY plane): {L1}, {L2}')

                goal = get_midline(L1, L2)
                # self.get_logger().info(f'Center Line (XY plane): {goal}')

                # adjustable (but requires recalibration or math)
                initial_pose = np.array([0, 0, np.pi/2])
                # print(f'goal is: {goal} and initial pose is: {initial_pose}')
                opt = robot.optimize(initial_pose, goal)
                # print(opt)

                opt_sol = opt.x.reshape((hor, 2))
                print(opt_sol[:4]) 
                # self.u_queue = opt_sol[0]
                self.m_queue = []
                for i in range(hor):
                    self.m_queue.append(opt_sol[i])

                # ax + by + c = 0, y = -a/b -c/b
                # msg = self.bridge.cv2_to_imgmsg(cv2_image_mapping, 'bgr8')
                # self.publisher_.publish(msg)
        
        except CvBridgeError as e:
            self.get_logger().error('CvBridge Error: {}'.format(e))

def main(args=None):

    rclpy.init(args=args)
    ag_model = Model()
    rclpy.spin(ag_model)
    ag_model.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
