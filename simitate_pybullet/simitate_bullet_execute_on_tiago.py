import os.path
import time
import math
from datetime import datetime
import csv
import argparse
import numpy as np
import sys
import yaml
import pybullet as p
# import cv2
import trajectory_loader
import zipfile

import requests

# imports for execution on tiago
import rosparam
from homer_robot import robot as lisa

def download_and_unzip_data_file(url="https://agas.uni-koblenz.de/simitate/data/simitate/data/simitate_pybullet_data.zip"):
    print('Beginning file download with simitate data files...')
    dest_filename = "simitate_pybullet_data.zip"
    # wget.download(url, dest_filename)
    r = requests.get(url)
    with open(dest_filename, 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile(dest_filename, 'r') as zip_ref:
        zip_ref.extractall(".")


if not os.path.exists("data"):
    download_and_unzip_data_file()

parser = argparse.ArgumentParser()
parser.add_argument('-gt', action='store_true', help="use ground truth")
parser.add_argument('-config', default="config_tiago.yaml", help="Config file")
parser.add_argument('csvfile')
args = parser.parse_args()
print ("gt is " + str(args.gt))
print ("csvfile is " + str(args.csvfile))

if len(args.csvfile) <= 1:
    print("Call like: simitate_bullet.py <trajectory_filename>")
    sys.exit()

## Read config
with open(args.config, "r") as config_file:
    try: 
        config = yaml.safe_load(config_file)
    except yaml.YAMLError as error: 
        print("error", error)


# def load_trajectory_from_csv(filename):
    # trajectory = []
    # cnt = 0 
    # with open(filename, 'r') as csvfile:
        # csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        # first = True
        # for row in csvreader:
            # print (', '.join(row))
            # cnt += 1
            # if first:
                # first = False
                # continue
            # print(row, cnt)
            # trajectory_point = {"timestep": row[0], "x": float(row[5]), "y": float(row[6]), "z": float(row[7])}
            # if row[4] == "hand":
                # print ("found hand")
                # trajectory.append(trajectory_point)
    # return trajectory


show_gt = args.gt  # using this parameter, the robot shows the behaviour with ground-truth input data this can be seen as current best case scenario

clid = p.connect(p.SHARED_MEMORY)
if (clid<0):
    p.connect(p.GUI)

video_filename = os.path.basename(args.csvfile)[:-4]+".mp4"
# print("Log video to: ")
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)
p.loadURDF(config["environment"]["model"],[0,0,0], globalScaling=0.01)



jd = config["jd"]
ll = config["ll"]
ul = config["ul"]
jr = config["jr"]
rest_pose = config["rest_pose"]
robot_name = config["robot"]["name"]
robot_origin_pos = config["robot"]["origin_pos"]
robot_origin_orientation = config["robot"]["origin_orientation"]
robotId = p.loadURDF(config["robot"]["model"], robot_origin_pos, robot_origin_orientation)
robotEndeffectorIndex = config["robot"]["endeffector_index"]

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.resetDebugVisualizerCamera(cameraDistance=2.0,
                             cameraYaw=180.0,
                             cameraPitch=0.0,
                             cameraTargetPosition=[0,0,1.6])
# p.resetBasePositionAndOrientation(sawyerId,[0,0,0],[0,0,0,1])

#bad, get it from name! sawyerEndEffectorIndex = 18
# numJoints = p.getNumJoints(sawyerId)
numJoints = p.getNumJoints(robotId)
#joint damping coefficents
# jd=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]
p.setGravity(0,0,0)
t=0.
prevPose=[0,0,0]
prev_pose=[0,0,0]
prevPose2=[0,0,0]
prev_pose_gt=[0,0,0]
hasPrevPose = 0

useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 1000


trajectory_file = args.csvfile# sys.argv[1]

tl = trajectory_loader.SimitateTrajectoryLoader()
tl.load_trajectories(trajectory_file)
trajectory = tl.trajectories["hand"]

# trajectory = load_trajectory_from_csv(filename=trajectory_file)
# a.load_trajectories_from_file("../data/eval/circle_2018-08-23-17-55-04.pkl")
baseline_trajectory_file = trajectory_file[:-3]+"pkl"
if os.path.isfile(baseline_trajectory_file):
    print("Found a baseline dump, now loading")
    tl.load_trajectories_from_file(baseline_trajectory_file)
    baseline_trajectory = tl.trajectories["baseline_trajectory"]
else:
    print("no baseline found, still processing, but approaching gt values")
# print(a.get_transform_to_world())
# sys.exit



# cv2.namedWindow('kinect_image',cv2.WINDOW_NORMAL)

# print(a.trajectories["hand"])

def bullet_24_to_ros_24(bullet_24):
    return bullet_24[15:22] + [bullet_24[23]] + [bullet_24[22]] + bullet_24[13:15] + [bullet_24[12]] + [bullet_24[3]] + [bullet_24[1]] + bullet_24[10:12] + bullet_24[6:8] + bullet_24[8:10] + bullet_24[4:6] + [bullet_24[2]] + [bullet_24[0]]

ros_trajectory = []

step_cnt = 0
while 1:
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second/60.)*2.*math.pi
        print (t)
    else:
        t=t+0.01
        time.sleep(0.01)
        # time.sleep(0.02)
        # time.sleep(0.3)

    for i in range (1):
        # pos = [1.0,0.2*math.cos(t),0.+0.2*math.sin(t)]
        # pos = [t,0,0]

        # print step_cnt, len(trajectory), trajectory[step_cnt]
        pos = trajectory[step_cnt,1:4]
        # print pos
        # pos = np.array([trajectory[step_cnt]["x"],
                        # trajectory[step_cnt]["y"],
                        # trajectory[step_cnt]["z"]])
        if not show_gt:
            if "baseline_trajectory" in tl.trajectories.keys():
                if len(baseline_trajectory) > step_cnt:
                    index = step_cnt
                else:
                    index = len(baseline_trajectory) - 1
                    sys.exit()
                print(index)
                pos_open_pose = tl.transform_to_world([baseline_trajectory[index][1:]][0])[:3]
                print("baseline pos", pos_open_pose)


        # img_filename = os.path.dirname(sys.argv[1])+"/_kinect2_qhd_image_color_rect_compressed/frame"+str(trajectory[step_cnt]["timestep"])+".jpg"
        # print(img_filename)
        # if os.path.isfile(img_filename):
            # print("found associate image")
            # img=cv2.imread('/home/jeff/Downloads/iphone.png', 1)
            # cv2.imshow('kinect_image',img)
            # cv2.waitKey(0)


        # joint_poses_open_pose = p.calculateInverseKinematics(sawyerId,sawyerEndEffectorIndex,pos,jointDamping=jd)
        # joint_poses_open_pose = p.calculateInverseKinematics(robotId,robotEndeffectorIndex,pos,lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rest_pose)


        if show_gt:
            joint_poses_gt = p.calculateInverseKinematics(robotId,
                                                          robotEndeffectorIndex,
                                                          pos,
                                                          lowerLimits=ll,
                                                          upperLimits=ul,
                                                          jointRanges=jr,
                                                          restPoses=rest_pose)
            for i in range (numJoints):
                # jointInfo = p.getJointInfo(sawyerId, i)
                jointInfo = p.getJointInfo(robotId, i)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    # p.resetJointState(sawyerId,i,joint_poses_open_pose[qIndex-7])
                    p.resetJointState(robotId,i,joint_poses_gt[qIndex-7])

            ls_gt = p.getLinkState(robotId,robotEndeffectorIndex)
            
            # transform pose to ros format and store
            ros_trajectory.append(bullet_24_to_ros_24(joint_poses_gt))
        else:
            #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            joint_poses_open_pose = p.calculateInverseKinematics(robotId,
                                                                 robotEndeffectorIndex,
                                                                 pos_open_pose,
                                                                 lowerLimits=ll,
                                                                 upperLimits=ul,
                                                                 jointRanges=jr,
                                                                 restPoses=rest_pose)
            for i in range (numJoints):
                # jointInfo = p.getJointInfo(sawyerId, i)
                jointInfo = p.getJointInfo(robotId, i)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    # p.resetJointState(sawyerId,i,joint_poses_open_pose[qIndex-7])
                    p.resetJointState(robotId,i,joint_poses_open_pose[qIndex-7])
            ls = p.getLinkState(robotId,robotEndeffectorIndex)
            
            # transform pose to ros format and store
            ros_trajectory.append(bullet_24_to_ros_24(joint_poses_open_pose))

    if (hasPrevPose):

        if show_gt:
            p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
            p.addUserDebugLine(prev_pose_gt,ls_gt[4],[1,0,1],1,trailDuration)
            tl.add_point_to_trajectory("baseline_trajectory_ik_gt_"+robot_name,
                                       np.float64(str(trajectory[step_cnt][0])),
                                       # np.float64(str(trajectory[step_cnt]["timestep"])),
                                       [float(ls_gt[4][0]),
                                        float(ls_gt[4][1]),
                                        float(ls_gt[4][2])
                                       ])
            print(pos)
        else:
            p.addUserDebugLine(prev_pose,ls[4],[1,0,0],1,trailDuration)
            tl.add_point_to_trajectory("baseline_trajectory_ik_open_pose_"+robot_name,
                                       np.float64(str(baseline_trajectory[index][0])),
                                       [float(ls[4][0]),
                                        float(ls[4][1]),
                                        float(ls[4][2])
                                       ])
            p.addUserDebugLine(prevPose2,pos_open_pose,[1,1,0],1,trailDuration)
            prevPose2=pos_open_pose
            print(pos_open_pose)

    prevPose=pos
    if show_gt:
        prev_pose_gt=ls_gt[4]
    else:
        prev_pose=ls[4]
    hasPrevPose = 1
    tl.save_trajectory_to_file(os.path.basename(baseline_trajectory_file)[:-4]+"_"+robot_name+".pkl")
    if step_cnt >= len(trajectory)-1:
        step_cnt = 0
        print("Sequence finished")
        break
    step_cnt += 1
# p.stopStateLogging()
# cv2.destroyAllWindows()

# Execute trajectory on tiago:
print("Executing now on tiago...")
robot = lisa.Robot()

repr_name = "/last_simitate_trajectory"
rosparam.set_param_raw(repr_name, ros_trajectory)
robot.play_arm_motion(repr_name)

print ("finished execution")

