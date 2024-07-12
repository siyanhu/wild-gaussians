import json
import math
import random

import root_file_io as fio


def create_info_json(name, scene, downscale_factor=8):
    return {
        "loader": "colmap",
        "name": name,
        "scene":scene,
        "downscale_factor": downscale_factor
    }

def get_train_image_names(data_dir_path):
    image_dir = fio.createPath(fio.sep, [data_dir_path], 'images')
    seqs_dir = fio.traverse_dir(image_dir, full_path=True, towards_sub=False)
    seqs_dir = fio.filter_folder(seqs_dir, filter_out=False, filter_text='seq')
    rslt = []
    for seq_dir_pth in seqs_dir:
        (seqd, seqname, seqe) = fio.get_filename_components(seq_dir_pth)
        image_paths = fio.traverse_dir(seq_dir_pth, full_path=True, towards_sub=False)
        image_names = [fio.get_filename_components(x)[1] for x in image_paths]
        seq_image_names = [fio.sep.join([seqname, x]) + '.png' for x in image_names]
        rslt += seq_image_names
    return rslt
    


if __name__ == "__main__":
    tag = 'camb'
    scenes = ['scene_GreatCourt', 'scene_KingsCollege', 'scene_OldHospital', 'scene_ShopFacade', 'scene_StMarysChurch']
    folds = ['train_full_byorder_85', 'test_full_byorder_59']
    scene_tag = 2
    fold_tag = 0

    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'raw_data', tag, scenes[scene_tag], folds[fold_tag]])
    if fio.file_exist(data_dir) == False:
        exit()
    info_dict = create_info_json(tag, scenes[scene_tag])
    spth_info_dict = fio.createPath(fio.sep, [data_dir], 'info.json')
    with open(spth_info_dict, 'w') as json_file:
        json.dump(info_dict, json_file, indent=4)

    train_images_list = get_train_image_names(data_dir)
    num_to_select = math.ceil(len(train_images_list) * 0.05)
    val_images_list = random.sample(train_images_list, num_to_select)    

    spth_train_list = fio.createPath(fio.sep, [data_dir], 'train_list.txt')
    with open(spth_train_list, 'w') as file:
        for string in train_images_list:
            file.write(string + '\n')
    
    spth_val_list = fio.createPath(fio.sep, [data_dir], 'test_list.txt')
    with open(spth_val_list, 'w') as file:
        for string in val_images_list:
            file.write(string + '\n')
