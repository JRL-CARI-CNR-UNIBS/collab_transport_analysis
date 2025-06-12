from bag_to_csv.info_extractor import InfoExtractor
import numpy as np


class TwistExtractor(InfoExtractor):
    def __init__(self):
        super().__init__("geometry_msgs/msg/Twist")

    def extract_info_from_msg(self, msg, msg_type):
        self._check_msg_type(msg_type)
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        linear_z = msg.linear.z
        angular_x = msg.angular.x
        angular_y = msg.angular.y
        angular_z = msg.angular.z
        normalized_linear = np.linalg.norm([linear_x, linear_y, linear_z])
        normalized_angular = np.linalg.norm([angular_x, angular_y, angular_z])

        return {
            "linear_x": linear_x,
            "linear_y": linear_y,
            "linear_z": linear_z,
            "angular_x": angular_x,
            "angular_y": angular_y,
            "angular_z": angular_z,
            "normalized_linear": normalized_linear,
            "normalized_angular": normalized_angular
        }
