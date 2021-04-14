import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import torch
import trimesh
from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
from sunrgbd.sunrgbd_detection_dataset import MEAN_COLOR_RGB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

from utils import setup_logger
from models import GroupFreeDetector
from models import parse_predictions

from eval_avg import parse_option, get_model, load_checkpoint
from utils import pc_util

def load_pointcloud(path_to_input):
    point_cloud = trimesh.load(path_to_input)
    if not isinstance(point_cloud, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh but got {type(point_cloud)}")
    
    return point_cloud

def transform(point_cloud, num_points: int, use_colors: bool, use_height: bool):
    """point_cloud N x 6 (x, y, z, r, g, b)
    """
    point_cloud = np.array(point_cloud.vertices)

    if not use_colors:
        point_cloud = point_cloud[:, :3]
    else:
        point_cloud = point_cloud[:, :6]
        point_cloud[:, 3:] -= MEAN_COLOR_RGB

    if use_height:
        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)


    point_cloud = pc_util.random_sampling(point_cloud, num_points, return_choices=False)

    ret_dict = {}
    ret_dict['point_clouds'] = point_cloud.astype(np.float32)
    
    return ret_dict    


def main(args, avg_times=5):
    DATASET_CONFIG = SunrgbdDatasetConfig()
    model, _ = get_model(args, DATASET_CONFIG)
    logger.info(str(model))
    save_path = load_checkpoint(args, model)
    model.cuda()

if __name__ == '__main__':
    opt = parse_option()

    logger = setup_logger(output=opt.dump_dir, name="eval")

    main(opt, opt.avg_times)