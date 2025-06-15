# %%
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import List

from ament_index_python import get_package_share_path

# Forward kinematics
import pinocchio
import xacro

# Read from Bags
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
# from tf2_py import BufferCore
from tf2_ros.buffer import Buffer
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, PoseStamped
from std_msgs.msg import Header
import tf2_geometry_msgs.tf2_geometry_msgs
import transforms3d.affines as affine
import transforms3d.quaternions as quaternion
import importlib

PATH_SRC= Path.home() / 'projects/lampo_ws/src/BagFile'
TEST_TYPE='linear'
BAG_TO_LOAD='linear'
CSV_FILE_TO_LOAD='linear_1_exported_aruco_poses.csv'
OMRON_URDF_PATH = get_package_share_path('omron_imm_description') / 'urdf' / 'system.urdf.xacro'
AZRAEL_URDF_PATH = get_package_share_path('azrael_description') / 'urdf' / 'system.urdf.xacro'

# Called Affine but in fact are Isometries
def transform_to_affine(t: TransformStamped):
  T = affine.compose([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
    quaternion.quat2mat([t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z]),
    np.ones(3))
  q = [t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z]
  return T

def pose_to_affine(p: Pose):
  return affine.compose([p.position.x, p.position.y, p.position.z],
    quaternion.quat2mat([p.orientation.w,p.orientation.x,p.orientation.y,p.orientation.z]),
    np.ones(3))

def affine_to_transform(a: np.ndarray, frame_id: str="", child_frame_id: str="") -> TransformStamped:
  t = TransformStamped()
  T, R, _, _ = affine.decompose(a)
  Q = quaternion.mat2quat(R)
  t.transform.translation.x = T[0]
  t.transform.translation.y = T[1]
  t.transform.translation.z = T[2]
  t.transform.rotation.w = Q[0]
  t.transform.rotation.x = Q[1]
  t.transform.rotation.y = Q[2]
  t.transform.rotation.z = Q[3]
  t.header.frame_id = frame_id
  t.child_frame_id = child_frame_id
  return t

def invert_affine(a: np.ndarray):
  P, R, _, _ = affine.decompose44(a)
  return affine.compose(-R.T @ P,  R.T, np.ones(3))

# %%
# Robot models
# ----------------------------------------------------------------------

# omron = pinocchio.buildModelFromUrdf(OMRON_URDF_PATH)
# azrael = pinocchio.buildModelFromUrdf(AZRAEL_URDF_PATH)

# %%
# Get data from bags
# ------------------------------------------------------------------------

# 2. Create a tf2 buffer (with a cache window you choose)
# tf_buffer = BufferCore(nanoseconds=60e9) # 60 seconds
tf_buffer = Buffer(cache_time=Duration(seconds=60))
bag_paths = [Path() / PATH_SRC / TEST_TYPE / f'{BAG_TO_LOAD}_{idx}' for idx in range(10)]

omron_jnts: List[JointState] = []
azrael_jnts: List[JointState]= []
aruco_poses: List[PoseArray] = []

def process_bag(path: str):
    global omron_jnts
    global azrael_jnts
    global aruco_poses
    global tf_stamps
    global tf_buffer

    storage_opts = StorageOptions(uri=path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )

    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    reader.set_filter(StorageFilter(
        topics=[
            '/tf',
            '/tf_static',
            '/omron/joint_states',
            '/azrael/joint_states',
            '/aruco_poses',
        ])
    )

    topic_types = {
        info.name: info.type
        for info in reader.get_all_topics_and_types()
    }

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        # get the msg class
        type_str = topic_types[topic]             # e.g. "tf2_msgs/msg/TFMessage"
        pkg,_, msg = type_str.split('/')            # ["tf2_msgs", "msg", "TFMessage"]
        module = importlib.import_module(f'{pkg}.msg')
        cls    = getattr(module, msg)

        # deserialize
        msg_obj = deserialize_message(data, cls)

        if topic == '/tf':
            tf_msg: TFMessage = msg_obj
            for t in tf_msg.transforms:
                t: TransformStamped
                tf_buffer.set_transform(t, 'bag')
        elif topic == '/tf_static':
          tf_msg: TFMessage = msg_obj
          for t in tf_msg.transforms:
              t: TransformStamped
              tf_buffer.set_transform_static(t, 'bag')
        elif topic == '/omron/joint_states':
            omron_jnts.append(msg_obj)
        elif topic == '/azrael/joint_states':
            azrael_jnts.append(msg_obj)
        elif topic == '/aruco_poses':
            aruco_poses.append(msg_obj)

# %%
# Get distance from aruco
# ------------------------------------------------------------------------

ts_azbase_marker: TransformStamped = TransformStamped()
ts_azbase_marker.header.stamp.sec = 0
ts_azbase_marker.header.stamp.nanosec = 1
ts_azbase_marker.child_frame_id = 'azrael_marker'
ts_azbase_marker.header.frame_id = 'azrael/base_link'
ts_azbase_marker.transform.translation.x = -0.23995416494481422
ts_azbase_marker.transform.translation.y = -0.3160449659404627
ts_azbase_marker.transform.translation.z = 0.23998565857824383
ts_azbase_marker.transform.rotation.x = 0.5106091522645703
ts_azbase_marker.transform.rotation.y = 0.49233048549432906
ts_azbase_marker.transform.rotation.z = -0.5016744194861883
ts_azbase_marker.transform.rotation.w = -0.49518861407525405
tf_buffer.set_transform(ts_azbase_marker, 'script')

fig, axs = plt.subplots(3,1,sharex=True)

for jdx, bag in enumerate(bag_paths):

  omron_jnts = []
  azrael_jnts = []
  aruco_poses = []

  dist_bases = []
  dist_tool = []
  time = []
  time2 = []

  process_bag(str(bag))

  # NOTE: manca un pezzo: trasformazione tra azrael/base_link (del marker) e azrael/base_footprint

  tp = tf_buffer.get_latest_common_time('omron/base_footprint', 'camera_link')
  ts_omron_base_camera = tf_buffer.lookup_transform('omron/base_footprint', 'camera_link', tp)
  tp = tf_buffer.get_latest_common_time('azrael/base_footprint', 'azrael/base_link')
  ts_azrael_base_mount = tf_buffer.lookup_transform('azrael/base_footprint', 'azrael/base_link', tp)
  for idx, pa in enumerate(aruco_poses):

    T_camera_marker = pose_to_affine(pa.poses[0])
    T_azrael_base_mount = transform_to_affine(ts_azrael_base_mount)
    T_azrael_mount_marker = transform_to_affine(ts_azbase_marker)
    T_azrael_base_marker = T_azrael_base_mount @ T_azrael_mount_marker
    T_marker_azrael_base = invert_affine(T_azrael_base_marker)
    T_omron_base_camera = transform_to_affine(ts_omron_base_camera)
    T_omron_base_azrael_base = T_omron_base_camera @ T_camera_marker @ T_marker_azrael_base
    dist_bases.append(np.linalg.norm(T_omron_base_azrael_base[0:3, -1]))
    time.append(pa.header.stamp.sec + 1e-9 * pa.header.stamp.nanosec)

    try:
      ts_omron_base_tool = tf_buffer.lookup_transform('omron/base_footprint', 'omron/flange', Time().from_msg(pa.header.stamp))
    except:
      continue
    T_omron_base_tool = transform_to_affine(ts_omron_base_tool)
    T_omron_tool_base = invert_affine(T_omron_base_tool)

    try:
      ts_azrael_base_tool = tf_buffer.lookup_transform('azrael/base_footprint', 'azrael/tool0', Time().from_msg(pa.header.stamp))
    except:
      continue
    T_azrael_base_tool = transform_to_affine(ts_omron_base_tool)

    T_omron_tool_azrael_tool = T_omron_tool_base @ T_omron_base_azrael_base @ T_azrael_base_tool
    dist_tool.append(np.linalg.norm(T_omron_tool_azrael_tool[0:3, -1]))
    time2.append(time[idx])


  dist_bases = np.array(dist_bases)
  dist_tool = np.array(dist_tool)
  time = np.array(time)
  time = time - time[0]
  if time2:
    time2 = np.array(time2)
    time2 = time2 - time2[0]
    axs[1].plot(time2, dist_tool, label=f'{jdx}')
  axs[0].plot(time, dist_bases, label=f'{jdx}')

  # From localization

  dist_tool__loc = []
  time__loc = []

  dt = 1e-1
  t0 = (aruco_poses[0].header.stamp.sec + 1e-9 * aruco_poses[0].header.stamp.nanosec)
  T = (aruco_poses[-1].header.stamp.sec + 1e-9 * aruco_poses[-1].header.stamp.nanosec)
  nT = (T-t0) / dt
  sp = np.linspace(t0, T, round(nT))
  for idx in range(len(sp)):
    t = sp[idx]
    try:
      ts_omron_azrael_tool__loc = tf_buffer.lookup_transform('omron/flange', 'azrael/tool0', Time(seconds=int(t), nanoseconds=(t - int(t))))
    except:
      continue
    T_omron_azrael_tool__loc = transform_to_affine(ts_omron_azrael_tool__loc)
    dist_tool__loc.append(np.linalg.norm(T_omron_azrael_tool__loc[0:3, -1]))
    time__loc.append(t)
  if time__loc:
    time__loc = time__loc - time__loc[0]
    axs[2].plot(time__loc, dist_tool__loc, label=f'{jdx}')

axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[0].legend(); axs[0].set_title('Bases')
axs[1].legend(); axs[1].set_title('Tools')
axs[2].legend(); axs[2].set_title('Tools (localization)')
fig.suptitle('Distances between robots')
plt.show()
