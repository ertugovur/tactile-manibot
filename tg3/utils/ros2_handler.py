# ROS2 handler for publishing tactile data
# add source /opt/ros/humble/setup.bash to ~/.bashrc
# to see published data: ros2 topic echo tactile/<topic>

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayLayout, MultiArrayDimension


_node = None
_namespace = "tactile"   # default

def set_namespace(namespace: str):
    global _namespace
    _namespace = namespace


class TactilePubNode(Node):
    def __init__(self):
        super().__init__(f'TactilePubNode_{_namespace}')
        self.contact = self.create_publisher(Bool, f'{_namespace}/contact', 10)
        self.ssim    = self.create_publisher(Float32, f'{_namespace}/ssim', 10)
        self.pose    = self.create_publisher(Float32MultiArray, f'{_namespace}/pose', 10)

def _ensure_node():
    global _node
    if _node is None:
        if not rclpy.ok():
            rclpy.init(args=None)
        _node = TactilePubNode()

def publish_pose(pose_dict):
    # Publish pose_predictions with internal ROS node
    _ensure_node()
    pose_labels, num = list(pose_dict.keys()), len(pose_dict)
    pose = [p.item() for _, p in pose_dict.items()]
    layout = MultiArrayLayout(
        dim=[MultiArrayDimension(label=", ".join(pose_labels), size=num, stride=num)],
        data_offset=0
    )
    _node.pose.publish(Float32MultiArray(layout=layout, data=pose))

def publish_contact(contact, ssim):
    # Publish contact and ssim values with internal ROS node
    _ensure_node()
    _node.contact.publish(Bool(data=bool(contact)))
    _node.ssim.publish(Float32(data=float(ssim)))

def shutdown():
    # Clean shutdown
    global _node
    if _node:
        _node.destroy_node()
        rclpy.shutdown()
        _node = None
