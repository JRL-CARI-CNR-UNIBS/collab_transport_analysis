from bag_to_csv.info_extractor import InfoExtractor
import numpy as np


class PoseArrayExtractor(InfoExtractor):
    def __init__(self):
        super().__init__("geometry_msgs/msg/PoseArray")

    def extract_info_from_msg(self, msg, msg_type):
        self._check_msg_type(msg_type)
        msg_info = {}
        msg_info['time'] = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        msg_info['frame_id'] = msg.header.frame_id

        for idx,pose in enumerate(msg.poses):
          msg_info[f'position_x_{idx}'] =    pose.position.x
          msg_info[f'position_y_{idx}'] =    pose.position.y
          msg_info[f'position_z_{idx}'] =    pose.position.z
          msg_info[f'orientation_x_{idx}'] = pose.orientation.x
          msg_info[f'orientation_y_{idx}'] = pose.orientation.y
          msg_info[f'orientation_z_{idx}'] = pose.orientation.z
          msg_info[f'orientation_w_{idx}'] = pose.orientation.w

        return msg_info
