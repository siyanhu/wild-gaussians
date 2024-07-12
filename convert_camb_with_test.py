import json
import numpy as np
import math

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
                cameras[camera_id] = [width, height, model, params.tolist()]
    return cameras


if __name__ == "__main__":
    tag = 'camb'
    scenes = ['scene_GreatCourt', 'scene_KingsCollege', 'scene_OldHospital', 'scene_ShopFacade', 'scene_StMarysChurch']
    folds = ['train_full_byorder_85', 'test_full_byorder_59']

    for scn_tag in scenes:
        data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'raw_data', tag, scn_tag])
        train_sparse_cameras_pth = fio.createPath(fio.sep, [data_dir, folds[0], 'sparse', '0'], 'cameras.txt')
        train_sparse_images_pth = fio.createPath(fio.sep, [data_dir, folds[0], 'sparse', '0'], 'images.txt')
        test_sparse_cameras_pth = fio.createPath(fio.sep, [data_dir, folds[1], 'sparse', '0'], 'cameras.txt')
        test_sparse_images_pth = fio.createPath(fio.sep, [data_dir, folds[1], 'sparse', '0'], 'images.txt')

        if not (fio.file_exist(train_sparse_cameras_pth) & fio.file_exist(train_sparse_images_pth) & fio.file_exist(test_sparse_cameras_pth) 
        & fio.file_exist(test_sparse_images_pth)):
            continue

        
