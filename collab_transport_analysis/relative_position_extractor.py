from bag_to_csv.info_extractor import InfoExtractor
import numpy as np
from scipy.spatial.transform import Rotation as R


class RelativePositionExtractor(InfoExtractor):
    def __init__(self):
        super().__init__("geometry_msgs/msg/PoseStamped")

    def extract_info_from_msg(self, msg, msg_type):
        self._check_msg_type(msg_type)
        msg_header = msg.header
        time = msg_header.stamp.sec + msg_header.stamp.nanosec * 1e-9
        frame_id = msg_header.frame_id
        position_x = msg.pose.position.x
        position_y = msg.pose.position.y
        position_z = msg.pose.position.z
        orientation_x = msg.pose.orientation.x
        orientation_y = msg.pose.orientation.y
        orientation_z = msg.pose.orientation.z
        orientation_w = msg.pose.orientation.w
        normalized_position = np.linalg.norm([position_x, position_y, position_z])
        normalized_orientation = np.linalg.norm([orientation_x, orientation_y, orientation_z, orientation_w])
        r = R.from_quat([orientation_x, orientation_y, orientation_z, orientation_w])
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler_angles = r.as_euler('xyz', degrees=True)
        roll, pitch, yaw = euler_angles
        rotation_normalized = np.linalg.norm([roll, pitch, yaw])

        return {
            "time": time,
            "frame_id": frame_id,
            "position_x": position_x,
            "position_y": position_y,
            "position_z": position_z,
            "orientation_x": orientation_x,
            "orientation_y": orientation_y,
            "orientation_z": orientation_z,
            "orientation_w": orientation_w,
            "normalized_position": normalized_position,
            "normalized_orientation": normalized_orientation,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "rotation_normalized": rotation_normalized
        }