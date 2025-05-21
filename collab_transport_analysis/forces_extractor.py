from bag_to_csv.info_extractor import InfoExtractor
import numpy as np


class ForcesExtractor(InfoExtractor):
    def __init__(self):
        super().__init__("geometry_msgs/msg/WrenchStamped")

    def extract_info_from_msg(self, msg, msg_type):
        self._check_msg_type(msg_type)
        msg_header = msg.header
        time = msg_header.stamp.sec + msg_header.stamp.nanosec * 1e-9
        frame_id = msg_header.frame_id
        force_x = msg.wrench.force.x
        force_y = msg.wrench.force.y
        force_z = msg.wrench.force.z
        torque_x = msg.wrench.torque.x
        torque_y = msg.wrench.torque.y
        torque_z = msg.wrench.torque.z
        normalized_force = np.linalg.norm([force_x, force_y, force_z])
        normalized_torque = np.linalg.norm([torque_x, torque_y, torque_z])

        return {
            "time": time,
            "frame_id": frame_id,
            "force_x": force_x,
            "force_y": force_y,
            "force_z": force_z,
            "torque_x": torque_x,
            "torque_y": torque_y,
            "torque_z": torque_z,
            "normalized_force": normalized_force,
            "normalized_torque": normalized_torque
        }