import numpy as np
import json
from scipy.spatial.transform import Rotation

import root_file_io as fio

def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    extrinsics = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                transform_matrix = [camera_id, qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2]]
                extrinsics[image_name] = transform_matrix
    return extrinsics


def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                # assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = [width, height, params.tolist()]
    return cameras


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def create_transformation_matrix(qw, qx, qy, qz, tx, ty, tz):
    """Create 4x4 transformation matrix from quaternion and translation."""
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def quat_trans_to_matrix(qw, qx, qy, qz, tx, ty, tz):
    # Normalize quaternion
    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*q[2]**2 - 2*q[3]**2,     2*q[1]*q[2] - 2*q[3]*q[0],     2*q[1]*q[3] + 2*q[2]*q[0]],
        [    2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2,     2*q[2]*q[3] - 2*q[1]*q[0]],
        [    2*q[1]*q[3] - 2*q[2]*q[0],     2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T


if __name__ == "__main__":
    tag = 'camb'
    scenes = ['scene_GreatCourt', 'scene_KingsCollege', 'scene_OldHospital', 'scene_ShopFacade', 'scene_StMarysChurch']
    folds = ['train_full_byorder_85', 'test_full_byorder_59']
    scene_tag = 3
    fold_tag = 1

    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'raw_data', tag, scenes[scene_tag], folds[fold_tag]])
    sparse_extrinsic_file_path = fio.createPath(fio.sep, [data_dir, 'sparse', '0'], 'images.txt')
    sparse_intrinsic_file_path = fio.createPath(fio.sep, [data_dir, 'sparse', '0'], 'cameras.txt')
    if fio.file_exist(sparse_extrinsic_file_path) == False:
        exit()
    if fio.file_exist(sparse_intrinsic_file_path) == False:
        exit()

    extrinsic_dict = read_extrinsics_text(sparse_extrinsic_file_path)
    intrinsic_dict = read_intrinsics_text(sparse_intrinsic_file_path)

    asd_json = {
        "camera_model": "pinhole",
        "image_size": [1024, 576],
        "fps": 30.0,
        "source": {
            "type": "interpolation",
            "interpolation": "kochanek-bartels",
            "keyframes": [],
            "default_fov": 75.0,
            "default_appearance": [],
            "is_cycle": False,
            "tension": 0.0,
            "default_transition_duration": 2.0
        },
        "frames": [],
        "format": "nerfbaselines-v1"
    }

    for image_id, content in extrinsic_dict.items():
        [camera_id, qvec0, qvec1, qvec2, qvec3, tvec0, tvec1, tvec2] = content
        [width, height, params] = intrinsic_dict[camera_id]

        qvec = [qvec0, qvec1, qvec2, qvec3]
        tvec = [tvec0, tvec1, tvec2]

        # transform = create_transform_matrix(qvec0, qvec1, qvec2, qvec3, tvec0, tvec1, tvec2)
        # transformation_matrix = create_transformation_matrix(qvec0, qvec1, qvec2, qvec3, tvec0, tvec1, tvec2)
        transformation_matrix = quat_trans_to_matrix(qvec0, qvec1, qvec2, qvec3, tvec0, tvec1, tvec2)

        # R = qvec2rotmat(qvec)
        # pose = np.eye(4)
        # pose[:3, :3] = R
        # pose[:3, 3] = tvec
        # pose_flat = pose[:3].flatten().tolist()

        pose = transformation_matrix.flatten().tolist()
        pose_flat = pose[0:12]

        keyframes = {
            "pose": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            "fov": [],
            "transition_duration": []
        }

        asd_json["source"]["keyframes"].append(keyframes)


        frame = {
            "pose": pose_flat,
            "intrinsics": [
                params[0],  # fx
                params[0],  # fy
                params[1],  # cx
                params[2]  # cy
            ],
            "appearance_weights": [1.0, 0.0, 0.0]
        }
        asd_json["frames"].append(frame)

    with open('output_file.json', 'w') as out:
        json.dump(asd_json, out, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else x)