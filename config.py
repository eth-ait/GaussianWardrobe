import torch
import numpy as np
import math
import os

device = torch.device('cuda:0')

# SMPL related
cano_smpl_pose = np.zeros(75, dtype = np.float32)
cano_smpl_pose[3+3*1+2] = math.radians(25)
cano_smpl_pose[3+3*2+2] = math.radians(-25)
# meshavatar template setting
# cano_smpl_pose[3+3*1+2] = torch.pi / 6
# cano_smpl_pose[3+3*2+2] = -torch.pi / 6
cano_smpl_pose[3+3*1+57] = -torch.pi / 2
cano_smpl_pose[3+3*1+60] = -torch.pi / 2
cano_smpl_pose = torch.from_numpy(cano_smpl_pose)
cano_smpl_transl = cano_smpl_pose[:3]
cano_smpl_global_orient = cano_smpl_pose[3:6]
cano_smpl_body_pose = cano_smpl_pose[6:69]

# fist pose
left_hand_pose = torch.tensor([0.09001956135034561, 0.1604590266942978, -0.3295670449733734, 0.12445037066936493, -0.11897698789834976, -1.5051144361495972, -0.1194705069065094, -0.16281449794769287, -0.6292539834976196, -0.27713727951049805, 0.035170216113328934, -0.5893177390098572, -0.20759613811969757, 0.07492011040449142, -1.4485805034637451, -0.017797302454710007, -0.12478633224964142, -0.7844052314758301, -0.4157009720802307, -0.5140947103500366, -0.2961726784706116, -0.7421528100967407, -0.11505582183599472, -0.7972996830940247, -0.29345276951789856, -0.18898937106132507, -0.6230823397636414, -0.18764786422252655, -0.2696149945259094, -0.5542467832565308, -0.47717514634132385, -0.12663133442401886, -1.2747308015823364, -0.23940050601959229, -0.1586960405111313, -0.7655659914016724, 0.8745182156562805, 0.5848557353019714, -0.07204405218362808, -0.5052485466003418, 0.1797526329755783, 0.3281439244747162, 0.5276764035224915, -0.008714836090803146, -0.4373648762702942], dtype = torch.float32)
right_hand_pose = torch.tensor([0.034751810133457184, -0.12605343759059906, 0.5510415434837341, 0.19454114139080048, 0.11147838830947876, 1.4676157236099243, -0.14799435436725616, 0.17293521761894226, 0.4679432511329651, -0.3042353689670563, 0.007868679240345955, 0.8570928573608398, -0.1827319711446762, -0.07225851714611053, 1.307037591934204, -0.02989627793431282, 0.1208646297454834, 0.7142824530601501, -0.3403030335903168, 0.5368582606315613, 0.3839572072029114, -0.9722614884376526, 0.17358140647411346, 0.911861002445221, -0.29665058851242065, 0.21779759228229523, 0.7269846796989441, -0.15343312919139862, 0.3083758056163788, 0.7146623730659485, -0.5153037309646606, 0.1721675992012024, 1.2982604503631592, -0.2590428292751312, 0.12812566757202148, 0.7502076029777527, 0.8694817423820496, -0.5263001322746277, 0.06934576481580734, -0.4630220830440521, -0.19237111508846283, -0.25436165928840637, 0.5972414612770081, -0.08250168710947037, 0.5013565421104431], dtype = torch.float32)


# project
PROJ_DIR = os.path.dirname(os.path.realpath(__file__))

# rgb_label
SURFACE_LABEL = ['skin', 'upper', 'lower', 'hair', 'shoe', 'outer']
SURFACE_LABEL_COLOR = torch.tensor([[128, 128, 128], [0, 128, 255], [255, 0, 128], [180, 50, 50], [50, 180, 50], [255, 128, 0]]).float().to(device) / 255.
RGB_LABEL = dict(zip(SURFACE_LABEL, SURFACE_LABEL_COLOR))

# smplx joint names
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]
JOINT_MAPPER = {}
for i, j in enumerate(JOINT_NAMES):
    JOINT_MAPPER[j] = i
def load_global_opt(path):
    import yaml
    global opt
    opt = yaml.load(open(path, encoding = 'UTF-8'), Loader = yaml.FullLoader)

