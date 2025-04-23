import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class WebcamSubscriber(Node):
    def __init__(self):
        super().__init__('webcam_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'webcam_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite('hello.jpg', cv_image)
            self.get_logger().info('Received image with dimensions: {}x{}'.format(cv_image.shape[1], cv_image.shape[0]))
        except CvBridgeError as e:
            self.get_logger().error('CvBridge Error: {}'.format(e))

def main(args=None):
    rclpy.init(args=args)
    webcam_subscriber = WebcamSubscriber()
    rclpy.spin(webcam_subscriber)
    webcam_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
