# %%
# %matplotlib
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass

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

# Analysis
import tsfel
from tsfel.utils.signal_processing import merge_time_series
import bootstrapped.bootstrap as bootstrap
import bootstrapped.stats_functions as bs_stats

PATH_SRC= Path.home() / 'projects/lampo_ws/src/BagFile'
TEST_TYPE='obstacle'
BAG_TO_LOAD=TEST_TYPE
OMRON_URDF_PATH = get_package_share_path('omron_imm_description') / 'urdf' / 'system.urdf.xacro'
AZRAEL_URDF_PATH = get_package_share_path('azrael_description') / 'urdf' / 'system.urdf.xacro'

START_TEST = 2
NUM_OF_TESTS = 1

sns.set_theme() # seaborn default theme
sns.set(font_scale=2)

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
bag_paths = [Path() / PATH_SRC / TEST_TYPE / f'{BAG_TO_LOAD}_{idx}' for idx in range(START_TEST, START_TEST + NUM_OF_TESTS)]


class ProcessBag:
  def __init__(self):
    self.omron_jnts : List[float]  = []
    self.azrael_jnts: List[float]  = []
    self.aruco_poses: List[Pose]  = []
    self.wrenches   : List[dict[str, float]]  = []
    self.path_msgs  : List[Path]  = []
    self.amcl_poses : List[dict[str, float]]  = []
    self.omron_poses: List[dict[str, float]]  = []
    self.time_data  : List[dict[str, float]]  = []

  def clear(self):
    self.omron_jnts.clear()
    self.azrael_jnts.clear()
    self.aruco_poses.clear()
    self.wrenches.clear()
    self.path_msgs.clear()
    self.amcl_poses.clear()
    self.omron_poses.clear()
    self.time_data.clear()



def process_bag(path: str) -> ProcessBag:

  pb: ProcessBag = ProcessBag()

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
        '/omron/plan',
        '/amcl_pose',
        '/omron/odom',
        '/omron/time_data'
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
      pb.omron_jnts.append(new_row)
    elif topic == '/joint_states':
      new_row = { name: q for name, q in zip(msg_obj.name, msg_obj.position) }
      new_row['time'] = stamp_to_float(msg_obj.header.stamp)
      pb.azrael_jnts.append(new_row)
    elif topic == '/aruco_poses':
      pb.aruco_poses.append(msg_obj)
    elif topic == '/force_torque_sensor_broadcaster/wrench':
      w = {}
      w['time'] = stamp_to_float(msg_obj.header.stamp)
      w['force.x'] =  msg_obj.wrench.force.x
      w['force.y'] =  msg_obj.wrench.force.y
      w['force.z'] =  msg_obj.wrench.force.z
      w['torque.x'] = msg_obj.wrench.torque.x
      w['torque.y'] = msg_obj.wrench.torque.y
      w['torque.z'] = msg_obj.wrench.torque.z
      pb.wrenches.append(w)
    elif topic == '/omron/plan':
      pb.path_msgs.append(msg_obj)
    elif topic == '/amcl_pose':
      w = {}
      w['time'] = stamp_to_float(msg_obj.header.stamp)
      w['pose.linear.x'] = msg_obj.pose.pose.position.x
      w['pose.linear.y'] = msg_obj.pose.pose.position.y
      w['pose.linear.z'] = msg_obj.pose.pose.position.z
      w['pose.angular.x'] = msg_obj.pose.pose.orientation.x
      w['pose.angular.y'] = msg_obj.pose.pose.orientation.y
      w['pose.angular.z'] = msg_obj.pose.pose.orientation.z
      w['pose.angular.w'] = msg_obj.pose.pose.orientation.w
      euler = Rotation.from_quat([msg_obj.pose.pose.orientation.x, msg_obj.pose.pose.orientation.y, msg_obj.pose.pose.orientation.z, msg_obj.pose.pose.orientation.w], scalar_first=False).as_euler(seq='ZYX')
      w['pose.angular.roll'] = euler[2]
      w['pose.angular.pitch'] = euler[1]
      w['pose.angular.yaw'] = euler[0]
      w['std.x'] =   np.sqrt(msg_obj.pose.covariance[0])
      w['std.y'] =   np.sqrt(msg_obj.pose.covariance[7])
      w['std.yaw'] = np.sqrt(msg_obj.pose.covariance[35])
      pb.amcl_poses.append(w)
    elif topic == '/omron/odom':
      w = {}
      w['time'] = stamp_to_float(msg_obj.header.stamp)
      w['pose.linear.x'] = msg_obj.pose.pose.position.x
      w['pose.linear.y'] = msg_obj.pose.pose.position.y
      w['pose.linear.z'] = msg_obj.pose.pose.position.z
      w['pose.angular.x'] = msg_obj.pose.pose.orientation.x
      w['pose.angular.y'] = msg_obj.pose.pose.orientation.y
      w['pose.angular.z'] = msg_obj.pose.pose.orientation.z
      w['pose.angular.w'] = msg_obj.pose.pose.orientation.w
      euler = Rotation.from_quat([msg_obj.pose.pose.orientation.x, msg_obj.pose.pose.orientation.y, msg_obj.pose.pose.orientation.z, msg_obj.pose.pose.orientation.w], scalar_first=False).as_euler(seq='ZYX')
      w['pose.angular.roll'] = euler[2]
      w['pose.angular.pitch'] = euler[1]
      w['pose.angular.yaw'] = euler[0]
      pb.omron_poses.append(w)
    elif topic == '/omron/time_data':
      w = {}
      w['merging_time'] = msg_obj.data[0]
      pb.time_data.append(w)
  return pb



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

fig, axs = plt.subplots(4,1,sharex=True)
# plt.get_current_fig_manager().full_screen_toggle()
fig_wr, ax_wr = plt.subplots()
# plt.get_current_fig_manager().full_screen_toggle()

cfg = tsfel.get_features_by_domain()
stat_columns = ['rms', 'rms_tf', 'rms_tool', 'rms_tool_tf', 'wrench']
stat_df = pd.DataFrame(columns=stat_columns)

times_ = {}

for bag in bag_paths:

  dist_bases = []
  dist_bases_tf = []
  dist_tool = []
  dist_tool_from_tf = []
  dist_marker = []
  time = []
  time2 = []

  pb = process_bag(str(bag))

  start_path = pb.path_msgs[0].header.stamp.sec + pb.path_msgs[0].header.stamp.nanosec * 1e-9

  df_omron_jnts = pd.DataFrame(pb.omron_jnts, columns=['time','omron/joint_1','omron/joint_2','omron/joint_3','omron/joint_4','omron/joint_5','omron/joint_6', 'omron/finger_joint'])
  df_omron_jnts.sort_values(by='time', axis='index')
  df_azrael_jnts = pd.DataFrame(pb.azrael_jnts, columns=['time', 'azrael/shoulder_pan_joint', 'azrael/shoulder_lift_joint', 'azrael/elbow_joint', 'azrael/wrist_1_joint', 'azrael/wrist_2_joint', 'azrael/wrist_3_joint'])
  df_azrael_jnts.sort_values(by='time', axis='index')
  df_wrench = pd.DataFrame(pb.wrenches, columns=['time', 'force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z'])
  df_wrench.sort_values(by='time', axis='index')
  # start_path_rel = start_path - df_wrench['time'][0]
  # df_wrench['time'] = df_wrench['time'] - df_wrench['time'][0]
  df_wrench['time'] = df_wrench['time'] - start_path
  df_wrench['force_norm'] = np.sqrt(df_wrench['force.x']**2 + df_wrench['force.y']**2 + df_wrench['force.z']**2)
  df_wrench_cut = df_wrench[df_wrench['time'] > 0]

  df_amcl = pd.DataFrame(pb.amcl_poses)
  df_amcl.sort_values(by='time', axis='index')
  df_amcl['time'] = df_amcl['time'] - start_path
  df_amcl_cut = df_amcl[df_amcl['time'] > 0]

  df_omron = pd.DataFrame(pb.omron_poses)
  df_omron.sort_values(by='time', axis='index')
  df_omron['time'] = df_omron['time'] - start_path
  df_omron_cut = df_omron[df_omron['time'] > 0]

  times_[str(bag)] = pb.time_data

  # merged_df = merge_time_series(
    # data={'omron.linear.x' : df_omron_cut['pose.linear.x'],
      # 'omron.linear.y' : df_omron_cut['pose.linear.y'],
      # 'omron.angular.yaw' : df_omron_cut['pose.angular.yaw'],
      # 'azrael.linear.x' : df_amcl_cut['pose.linear.x'],
      # 'azrael.linear.y' : df_amcl_cut['pose.linear.y'],
      # 'azrael.angular.yaw' : df_amcl_cut['pose.angular.yaw'],
    # },
    # fs_resample=50.0,
    # time_unit=1e-9
  # )


  tp = tf_buffer.get_latest_common_time('omron/base_footprint', 'camera_color_optical_frame')
  ts_omron_base_camera = tf_buffer.lookup_transform('omron/base_footprint', 'camera_color_optical_frame', tp)
  T_omron_base_camera = transform_to_affine(ts_omron_base_camera)

  tp = tf_buffer.get_latest_common_time('azrael/base_footprint', 'azrael/base_link')
  ts_azrael_base_mount = tf_buffer.lookup_transform('azrael/base_footprint', 'azrael/base_link', tp)
  T_azrael_base_mount = transform_to_affine(ts_azrael_base_mount)
  T_azrael_mount_marker = transform_to_affine(ts_azrael_mount_marker)
  T_azrael_base_marker = T_azrael_base_mount @ T_azrael_mount_marker
  T_marker_azrael_base = invert_affine(T_azrael_base_marker)

  df_azrael_fk = pd.DataFrame(columns=['time','x','y','z','roll','pitch','yaw'])

  initial_dist_bases_length: int = len(pb.aruco_poses)

  for idx, pa in enumerate(pb.aruco_poses):
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
    df_azrael_fk.loc[len(df_azrael_fk)] = ([stamp_to_float(pa.header.stamp) , T_azrael_base_tool[0,-1], T_azrael_base_tool[1,-1], T_azrael_base_tool[2,-1], *Rotation.from_matrix(T_azrael_base_tool[:3,:3]).as_euler('ZYX')])

    try:
      ts_omron_base_azrael_base_from_tf = tf_buffer.lookup_transform('omron/base_footprint', 'azrael/base_footprint', Time().from_msg(pa.header.stamp))
    except:
      dist_bases.pop()
      continue
    T_omron_base_azrael_base_from_tf = transform_to_affine(ts_omron_base_azrael_base_from_tf)
    dist_bases_tf.append(np.linalg.norm(T_omron_base_azrael_base_from_tf[:3,-1]))

    T_omron_tool_azrael_tool = T_omron_tool_base @ T_omron_base_azrael_base @ T_azrael_base_tool
    T_omron_tool_azrael_tool_from_tf = T_omron_tool_base @ T_omron_base_azrael_base_from_tf @ T_azrael_base_tool
    # dist_tool.append(np.linalg.norm(T_omron_tool_azrael_tool[:3, -1]))
    dist_tool.append(np.array(
      np.linalg.norm(T_omron_tool_azrael_tool[:3, -1])))
    dist_tool_from_tf.append(np.array(
      np.linalg.norm(T_omron_tool_azrael_tool_from_tf[:3, -1])
    ))
    time2.append(time[idx])

  print(f'dist_bases: original: {initial_dist_bases_length}, reduced: {len(dist_bases)}')
  time_from_tf = []

  dt = 1e-2
  t0 = stamp_to_float(pb.aruco_poses[0].header.stamp)
  T = stamp_to_float(pb.aruco_poses[-1].header.stamp)
  nT = (T-t0) / dt
  sp = np.linspace(t0, T, round(nT))

  dist_bases = np.array(dist_bases)
  dist_tool = np.array(dist_tool)
  dist_tool_from_tf = np.array(dist_tool_from_tf)
  time = np.array(time)
  time = time - time[0]

  time2 = np.array(time2)
  # time2 = time2 - time2[0]
  time2 = time2 - start_path

  dist_df: pd.DataFrame = pd.DataFrame({'time' : time2, 'dist_bases' : dist_bases, 'dist_bases_tf' : dist_bases_tf, 'dist_tool' : dist_tool, 'dist_tool_tf' : dist_tool_from_tf}, columns=['time', 'dist_bases', 'dist_bases_tf', 'dist_tool', 'dist_tool_tf'])
  dist_df['base_tf_error'] = dist_df['dist_bases_tf'] - dist_df['dist_bases_tf'][0]
  dist_df['base_error'] = dist_df['dist_bases'] - dist_df['dist_bases'][0]
  dist_df_cut = dist_df[dist_df['time'] > 0]

  # dist_df.plot(x='time', y='dist_bases', label=f'{bag.name}-bases-from-marker', grid=True, ax=axs[0])
  #dist_df_cut.plot(x='time', y='dist_bases', label=f'{bag.name}-bases-from-marker', grid=True, ax=axs[0])
  # dist_df.plot(x='time', y='dist_tool', grid=True, ax=axs[1], label=f'dist_tool_{bag.name}')
  #dist_df_cut.plot(x='time', y='dist_tool', grid=True, ax=axs[1], label=f'dist_tool_{bag.name}')
  sns.lineplot(ax=axs[3], data=dist_df_cut, x='time', y='dist_tool')
  axs[3].set_ylabel('Tools\nAruco\n[m]')
  sns.lineplot(ax=axs[2], data=dist_df_cut, x='time', y='dist_tool_tf')
  axs[2].set_ylabel('Tools\nAMCL\n[m]')
  sns.lineplot(ax=axs[1], data=dist_df_cut, x='time', y='base_error')
  axs[1].set_ylabel('Error\nAruco\n[m]')
  sns.lineplot(ax=axs[0], data=dist_df_cut, x='time', y='base_tf_error')
  axs[0].set_ylabel('Error\nAMCL\n[m]')
  axs[2].set_xlabel('time [s]')
  # plt.get_current_fig_manager().full_screen_toggle()

  _, axd = plt.subplots()
  sns.lineplot(data=dist_df_cut, x='time', y='base_error', ax=axd, label='Aruco')
  sns.lineplot(data=dist_df_cut, x='time', y='base_tf_error', ax=axd, label='Localization')
  axd.set_ylabel('distance [m]')
  axd.set_xlabel('time [s]')
  # plt.get_current_fig_manager().full_screen_toggle()

  axs[0].grid(True)
  axs[1].grid(True)
  # axs[2].grid(True)
  #axs[0].legend(); axs[0].set_title('Bases')
  #axs[1].legend(); axs[1].set_title('Tools from Aruco')
  # fig.suptitle('Distances')

  # df_wrench.plot(x='time', y='force.x', ax=ax_wr, grid=True)
  sns.lineplot(data=df_wrench_cut, x='time', y='force_norm', ax=ax_wr, label='')

  _, ax_amcl = plt.subplots(3,1, sharex=True)
  sns.lineplot(data=df_amcl_cut, x='time', y='pose.linear.x', ax=ax_amcl[0])
  ax_amcl[0].set_ylabel('x [m]')
  ax_amcl[0].fill_between(df_amcl_cut['time'], df_amcl_cut['pose.linear.x'] - 2 * df_amcl_cut['std.x'], df_amcl_cut['pose.linear.x'] + 2 * df_amcl_cut['std.x'], alpha=0.3 )
  sns.lineplot(data=df_amcl_cut, x='time', y='pose.linear.y', ax=ax_amcl[1])
  ax_amcl[1].set_ylabel('y [m]')
  ax_amcl[1].fill_between(df_amcl_cut['time'], df_amcl_cut['pose.linear.y'] - 2 * df_amcl_cut['std.y'], df_amcl_cut['pose.linear.y'] + 2 * df_amcl_cut['std.y'], alpha=0.3 )
  sns.lineplot(data=df_amcl_cut, x='time', y='pose.angular.yaw', ax=ax_amcl[2])
  ax_amcl[2].set_ylabel('yaw [rad]')
  ax_amcl[2].fill_between(df_amcl_cut['time'], df_amcl_cut['pose.angular.yaw'] - 2 * df_amcl_cut['std.yaw'], df_amcl_cut['pose.angular.yaw'] + 2 * df_amcl_cut['std.yaw'], alpha=0.3)
  ax_amcl[2].set_xlabel('time [s]')
  # plt.get_current_fig_manager().full_screen_toggle()

  df_azrael_fk['time'] = df_azrael_fk['time'] - start_path
  df_azrael_fk_cut = df_azrael_fk[df_azrael_fk['time'] > 0]
  _, ax_jnt = plt.subplots(3,1, sharex=True)
  sns.lineplot(data=df_azrael_fk_cut, x='time', y='x', ax=ax_jnt[0])
  sns.lineplot(data=df_azrael_fk_cut, x='time', y='y', ax=ax_jnt[1])
  sns.lineplot(data=df_azrael_fk_cut, x='time', y='z', ax=ax_jnt[2])
  # sns.lineplot(data=df_azrael_fk_cut, x='time', y='roll',  ax=ax_jnt[0, 1])
  # sns.lineplot(data=df_azrael_fk_cut, x='time', y='pitch', ax=ax_jnt[1, 1])
  # sns.lineplot(data=df_azrael_fk_cut, x='time', y='yaw',   ax=ax_jnt[2, 1])
  ax_jnt[0].set_ylabel('x [m]')
  ax_jnt[1].set_ylabel('y [m]')
  ax_jnt[2].set_ylabel('z [m]')
  ax_jnt[2].set_xlabel('time [s]')
  # plt.get_current_fig_manager().full_screen_toggle()

  stat_tmp_df = pd.DataFrame(
  [[
   tsfel.rms(dist_df_cut['base_error']   ),      # ['rms']
   tsfel.rms(dist_df_cut['base_tf_error']),   # ['rms_tf']
   tsfel.rms(dist_df_cut['dist_tool']    ),                 # ['rms_tool']
   tsfel.rms(dist_df_cut['dist_tool_tf'] ),                 # ['rms_tool_tf']
   tsfel.rms(df_wrench_cut['force_norm'] )       # ['wrench']
  ]],
  columns=stat_columns)

  stat_df = pd.concat([stat_df, stat_tmp_df], ignore_index=True)



_, ax = plt.subplots(1,2)
# sns.violinplot(data=stat_df, ax=ax, alpha=.3)
# sns.boxplot(data=stat_df, ax=ax, fill=False, linewidth=0.5)
print(stat_df)
# sns.stripplot(data=stat_df[['rms', 'rms_tool','rms_tf', 'rms_tool_tf']], ax=ax, size=10)
sns.boxplot(data=stat_df[['rms','rms_tf']], ax=ax[0], fill=True, linewidth=0.5)
ax[0].set_xticks(['rms','rms_tf'])
ax[0].set_xticklabels(['Aruco\nBases', 'Localization\nBases'])
ax[0].set_ylabel('RMS [m]')
ax[0].set_title(['Mobile base distances'])
sns.boxplot(data=stat_df[['rms_tool', 'rms_tool_tf']], ax=ax[1], fill=True, linewidth=0.5)
ax[1].set_xticks(['rms_tool', 'rms_tool_tf'])
ax[1].set_xticklabels(['Aruco\nTool', 'Localization\nTool'])
ax[1].set_ylabel('RMS [m]')
ax[1].set_title(['Tool distances'])
# plt.get_current_fig_manager().full_screen_toggle()

_, ax_w = plt.subplots()
# sns.violinplot(data=stat_df, ax=ax, alpha=.3)
sns.boxplot(data=stat_df[['wrench']], ax=ax_w, fill=True, linewidth=0.5)
# sns.swarmplot(data=stat_df[['wrench']], ax=ax_w, size=10)
# sns.stripplot(data=stat_df[['wrench']], ax=ax_w, jitter=False)
ax_w.set_xticks(['wrench'])
ax_w.set_xticklabels(['Force'])
ax_w.set_ylabel('Wrench [N]')
# plt.get_current_fig_manager().full_screen_toggle()


merging_times: dict[str, pd.Series] = {}
merge_times_tab = []
# df_merge_times_stat = pd.DataFrame(columns=['mean','min','max'])
_, ax_times = plt.subplots()
for b,c in times_.items():
  merging_times[b] = pd.Series(np.array([t['merging_time'] for t in c]))
  merge_times_tab.append([np.mean(merging_times[b]) * 1e-6, # converted in sec
    (np.mean(merging_times[b]) - np.min(merging_times[b])) * 1e-6,
    (np.max(merging_times[b]) - np.mean(merging_times[b])) * 1e-6,
    np.sqrt(np.var(merging_times[b])) * 2 * 1e-6]
  )

df_merge_times = pd.DataFrame(merging_times)
df_merge_times_stat = pd.DataFrame(np.array(merge_times_tab), columns=['mean', 'min', 'max', '2 * std'])
df_merge_times_stat.set_index(pd.Index(merging_times.keys()))

# sns.pointplot(data=df_merge_times, linestyle='none', errorbar='ci', estimator=np.mean)

print(df_merge_times_stat)

# ['force.x','force.y','force.z']
print('Plotting...')
plt.show()
