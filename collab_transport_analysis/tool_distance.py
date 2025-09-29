# %%
%matplotlib
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
import tempfile

# Read from Bags
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
# from tf2_py import BufferCore
from tf2_ros.buffer import Buffer
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, PoseStamped, WrenchStamped
from std_msgs.msg import Header
import tf2_geometry_msgs.tf2_geometry_msgs
import transforms3d.affines as affine
import transforms3d.quaternions as quaternion
import importlib
import bisect

PATH_SRC= Path.home() / 'projects/lampo_ws/src/BagFile'
TEST_TYPE='linear'
BAG_TO_LOAD=TEST_TYPE
CSV_FILE_TO_LOAD='linear_1_exported_aruco_poses.csv'
OMRON_URDF_PATH = get_package_share_path('omron_imm_description') / 'urdf' / 'system.urdf.xacro'
AZRAEL_URDF_PATH = get_package_share_path('azrael_description') / 'urdf' / 'system.urdf.xacro'

# Called Affine but in fact are Isometries
def transform_to_affine(t: TransformStamped):
  return affine.compose([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
    quaternion.quat2mat([t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z]),
    np.ones(3))

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

def stamp_to_float(s) -> float:
  return s.sec + s.nanosec * 1e-9

# Robot models
# ----------------------------------------------------------------------

omron_urdf = xacro.process(OMRON_URDF_PATH, mappings=None)
tmp_name: str = ""
with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp:
  tmp_name = tmp.name
  tmp.write(omron_urdf)
omron = pinocchio.buildModelFromUrdf(tmp_name)
omron_data = pinocchio.Data(omron)

azrael_urdf = xacro.process(AZRAEL_URDF_PATH, mappings=None)
with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as tmp:
  tmp_name = tmp.name
  tmp.write(azrael_urdf)
azrael = pinocchio.buildModelFromUrdf(tmp_name)
azrael_data = pinocchio.Data(azrael)

# Get data from bags
# ------------------------------------------------------------------------

# 2. Create a tf2 buffer (with a cache window you choose)
# tf_buffer = BufferCore(nanoseconds=60e9) # 60 seconds
tf_buffer = Buffer(cache_time=Duration(seconds=60))
bag_paths = [Path() / PATH_SRC / TEST_TYPE / f'{BAG_TO_LOAD}_{idx}' for idx in range(5,6)]

omron_jnts: list[dict[str, float]] = []
azrael_jnts: List[dict[str, float]]= []
aruco_poses: List[PoseArray] = []
wrenches: List[dict[str, float]] = []

def process_bag(path: str):
  global df_omron_jnts
  global azrael_jnts
  global aruco_poses
  global tf_buffer
  global wrenches
  global path_msgs

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
        '/joint_states',
        '/aruco_poses',
        '/force_torque_sensor_broadcaster/wrench',
        '/omron/plan'
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
      new_row = { name: q for name, q in zip(msg_obj.name, msg_obj.position) }
      new_row['time'] = stamp_to_float(msg_obj.header.stamp)
      omron_jnts.append(new_row)
    elif topic == '/joint_states':
      new_row = { name: q for name, q in zip(msg_obj.name, msg_obj.position) }
      new_row['time'] = stamp_to_float(msg_obj.header.stamp)
      azrael_jnts.append(new_row)
    elif topic == '/aruco_poses':
      aruco_poses.append(msg_obj)
    elif topic == '/force_torque_sensor_broadcaster/wrench':
      w = {}
      w['time'] = stamp_to_float(msg_obj.header.stamp)
      w['force.x'] =  msg_obj.wrench.force.x
      w['force.y'] =  msg_obj.wrench.force.y
      w['force.z'] =  msg_obj.wrench.force.z
      w['torque.x'] = msg_obj.wrench.torque.x
      w['torque.y'] = msg_obj.wrench.torque.y
      w['torque.z'] = msg_obj.wrench.torque.z
      wrenches.append(w)
    elif topic == '/omron/plan':
      path_msgs.append(msg_obj)


# Get distance from aruco
# ------------------------------------------------------------------------

ts_azrael_mount_marker: TransformStamped = TransformStamped()
ts_azrael_mount_marker.header.stamp.sec = 0
ts_azrael_mount_marker.header.stamp.nanosec = 1
ts_azrael_mount_marker.header.frame_id = 'azrael/base_link'
ts_azrael_mount_marker.child_frame_id = 'azrael_marker'
ts_azrael_mount_marker.transform.translation.x = -0.23995416494481422
ts_azrael_mount_marker.transform.translation.y = -0.3160449659404627
ts_azrael_mount_marker.transform.translation.z = 0.23998565857824383
ts_azrael_mount_marker.transform.rotation.x = 0.5106091522645703
ts_azrael_mount_marker.transform.rotation.y = 0.49233048549432906
ts_azrael_mount_marker.transform.rotation.z = -0.5016744194861883
ts_azrael_mount_marker.transform.rotation.w = -0.49518861407525405
tf_buffer.set_transform(ts_azrael_mount_marker, 'script')


for bag in bag_paths:

  fig_aruco, ax_aruco = plt.subplots()
  fig, axs = plt.subplots(3,1,sharex=True)
  fig_wr, ax_wr = plt.subplots()

  omron_jnts = []
  azrael_jnts = []
  aruco_poses = []
  wrenches = []
  path_msgs = []

  dist_bases = []
  dist_tool = []
  dist_marker = []
  time = []
  time2 = []

  process_bag(str(bag))


  df_omron_jnts = pd.DataFrame(omron_jnts, columns=['time','omron/joint_1','omron/joint_2','omron/joint_3','omron/joint_4','omron/joint_5','omron/joint_6', 'omron/finger_joint'])
  df_omron_jnts.sort_values(by='time', axis='index')
  df_azrael_jnts = pd.DataFrame(azrael_jnts, columns=['time', 'azrael/shoulder_pan_joint', 'azrael/shoulder_lift_joint', 'azrael/elbow_joint', 'azrael/wrist_1_joint', 'azrael/wrist_2_joint', 'azrael/wrist_3_joint'])
  df_azrael_jnts.sort_values(by='time', axis='index')
  df_wrench = pd.DataFrame(wrenches, columns=['time', 'force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z'])
  df_wrench.sort_values(by='time', axis='index')

  # start_path = path_msgs[0].header.stamp.sec + path_msgs[0].header.stamp.nanosec * 1e-9

  # print(start_path - df_wrench['time'][0])

  # df_wrench = df_wrench[df_wrench['time'] > start_path]

  tp = tf_buffer.get_latest_common_time('omron/base_footprint', 'camera_color_optical_frame')
  ts_omron_base_camera = tf_buffer.lookup_transform('omron/base_footprint', 'camera_color_optical_frame', tp)
  T_omron_base_camera = transform_to_affine(ts_omron_base_camera)

  tp = tf_buffer.get_latest_common_time('azrael/base_footprint', 'azrael/base_link')
  ts_azrael_base_mount = tf_buffer.lookup_transform('azrael/base_footprint', 'azrael/base_link', tp)
  T_azrael_base_mount = transform_to_affine(ts_azrael_base_mount)
  T_azrael_mount_marker = transform_to_affine(ts_azrael_mount_marker)
  T_azrael_base_marker = T_azrael_base_mount @ T_azrael_mount_marker
  T_marker_azrael_base = invert_affine(T_azrael_base_marker)

  for idx, pa in enumerate(aruco_poses):
    T_camera_marker = pose_to_affine(pa.poses[0])
    T_omron_base_azrael_base = T_omron_base_camera @ T_camera_marker @ T_marker_azrael_base
    dist_marker.append(np.linalg.norm(T_camera_marker[:3,-1]))
    dist_bases.append (np.linalg.norm(T_omron_base_azrael_base[:3,-1]))
    # dist_bases.append(np.linalg.norm(T_omron_base_azrael_base[0:3,-1]))
    time.append(stamp_to_float(pa.header.stamp))

    ## USE: Forward Kinematics from models
    omron_q = df_omron_jnts.loc[df_omron_jnts['time'] >= stamp_to_float(pa.header.stamp), ~df_omron_jnts.columns.isin(['time','omron/finger_joint'])].iloc[0].to_numpy()
    pinocchio.forwardKinematics(omron, omron_data, omron_q)
    pinocchio.updateFramePlacements(omron, omron_data)
    T_omron_base_tool = np.asarray(omron_data.oMf[omron.getFrameId('omron/tcp')])
    T_omron_tool_base = invert_affine(T_omron_base_tool)

    azrael_q = df_azrael_jnts.loc[df_azrael_jnts['time'] >= stamp_to_float(pa.header.stamp), df_azrael_jnts.columns != 'time'].iloc[0].to_numpy()
    pinocchio.forwardKinematics(azrael, azrael_data, azrael_q)
    pinocchio.updateFramePlacements(azrael, azrael_data)
    T_azrael_base_tool = np.asarray(azrael_data.oMf[azrael.getFrameId('azrael/tool0')])
    ## END

    ## USE: tf_tree for respective base-tool
    # try:
    #   ts_omron_base_tool = tf_buffer.lookup_transform('omron/base_footprint', 'omron/tcp', Time().from_msg(pa.header.stamp))
    # except:
    #   continue
    # T_omron_base_tool = transform_to_affine(ts_omron_base_tool)
    # T_omron_tool_base = invert_affine(T_omron_base_tool)

    # try:
    #   ts_azrael_base_tool = tf_buffer.lookup_transform('azrael/base_footprint', 'azrael/tool0', Time().from_msg(pa.header.stamp))
    # except:
    #   continue
    # T_azrael_base_tool = transform_to_affine(ts_azrael_base_tool)
    ## END

    ## USE: tf_tree for base-base distance
    try:
      ts_omron_base_azrael_base_from_tf = tf_buffer.lookup_transform('omron/base_footprint', 'azrael/base_footprint', Time().from_msg(pa.header.stamp))
    except:
      continue
    T_omron_base_azrael_base_from_tf = transform_to_affine(ts_omron_base_azrael_base_from_tf)
    ##

    T_omron_tool_azrael_tool = T_omron_tool_base @ T_omron_base_azrael_base @ T_azrael_base_tool
    # dist_tool.append(np.linalg.norm(T_omron_tool_azrael_tool[:3, -1]))
    dist_tool.append(np.array([
      np.linalg.norm(T_omron_tool_azrael_tool[:3, -1]), # Aruco
      np.linalg.norm((T_omron_tool_base @ T_omron_base_azrael_base_from_tf @ T_azrael_base_tool)[:3,-1])  # Localization
    ]))
    time2.append(time[idx])

  dist_bases = np.array(dist_bases)
  dist_tool = np.array(dist_tool)
  dist_tool = dist_tool - dist_tool[0]
  time = np.array(time)
  time = time - time[0]
  if time2:
    time2 = np.array(time2)
    time2 = time2 - time2[0]
    axs[1].plot(time2, dist_tool, label=[f'{bag.name}-aruco',f'{bag.name}-localization'])
  axs[0].plot(time, dist_bases, label=f'{bag.name}-bases-from-marker')

  ax_aruco.plot(time, dist_marker-dist_marker[0], label=f'{bag.name}-marker')
  ax_aruco.plot(time, dist_bases-dist_bases[0], label=f'{bag.name}-bases-from-marker')
  ax_aruco.legend()
  ax_aruco.grid(True)

  # From localization

  dist_tool_from_tf = []
  time_from_tf = []

  dt = 1e-2
  t0 = stamp_to_float(aruco_poses[0].header.stamp)
  T = stamp_to_float(aruco_poses[-1].header.stamp)
  nT = (T-t0) / dt
  sp = np.linspace(t0, T, round(nT))
  for idx in range(len(sp)):
    t = sp[idx]
    try:
      ts_omron_azrael_tool_from_tf = tf_buffer.lookup_transform('omron/flange', 'azrael/tool0', Time(seconds=int(t), nanoseconds=int((t - int(t)) * 1e9) ) )
    except:
      continue
    T_omron_azrael_tool_from_tf = transform_to_affine(ts_omron_azrael_tool_from_tf)
    dist_tool_from_tf.append(np.linalg.norm(T_omron_azrael_tool_from_tf[0:3, -1]))
    # dist_tool_from_tf.append(T_omron_azrael_tool_from_tf[0:3, -1])
    time_from_tf.append(t)
  if time_from_tf:
    time_from_tf = time_from_tf - time_from_tf[0]
    axs[2].plot(time_from_tf, dist_tool_from_tf, label=f'{bag.name}')

  axs[0].grid(True)
  axs[1].grid(True)
  axs[2].grid(True)
  axs[0].legend(); axs[0].set_title('Bases')
  axs[1].legend(); axs[1].set_title('Tools (FK)')
  axs[2].legend(); axs[2].set_title('Tools (TF)')
  fig.suptitle('Distances between robots')

  df_wrench.plot(x='time',y='force.x', ax=ax_wr, grid=True)

# ['force.x','force.y','force.z']
print('Plotting...')
plt.show()
