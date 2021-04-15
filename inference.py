import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import pdb
import torch
import trimesh
import open3d as op3d
from trimesh import exchange
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
    with open(path_to_input, "rb") as ply_file_obj:
        point_cloud_data = exchange.ply.load_ply(ply_file_obj)

    point_cloud = trimesh.PointCloud(**point_cloud_data)

    if not isinstance(point_cloud, trimesh.PointCloud):
        raise ValueError(f"Expected Trimesh but got {type(point_cloud)}")
    
    return point_cloud

def transform(trimesh_point_cloud, num_points: int, use_colors: bool, use_height: bool):
    """point_cloud N x 6 (x, y, z, r, g, b)
    """
    if len(trimesh_point_cloud.colors) == 0:
        raise ValueError("cannot find color in an input point cloud")

    point_cloud = np.array(trimesh_point_cloud.vertices)
    
    colors = np.array(trimesh_point_cloud.colors, dtype=point_cloud.dtype)
    colors /= 255.0

    point_cloud = np.hstack((point_cloud, colors))


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
    ret_dict['point_clouds'] = torch.unsqueeze(torch.from_numpy(point_cloud.astype(np.float32)), 0)
    
    return ret_dict    

@torch.no_grad()
def main(args, avg_times=5):
    point_cloud = load_pointcloud(args.in_ply)

    DATASET_CONFIG = SunrgbdDatasetConfig()

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': True, 'nms_iou': args.nms_iou,
                   'use_old_type_nms': args.use_old_type_nms, 'cls_nms': True,
                   'per_class_proposal': True,
                   'conf_thresh': args.conf_thresh, 'dataset_config': DATASET_CONFIG}

    input_data = transform(point_cloud, args.num_point, args.use_color, args.use_height)

    input_data["point_clouds"] = input_data["point_clouds"].cuda()

    model, _ = get_model(args, DATASET_CONFIG)
    model.eval()
    save_path = load_checkpoint(args, model)
    model.cuda()
    end_points = model(input_data)

    end_points["point_clouds"] = input_data["point_clouds"]

    if args.num_decoder_layers > 0:
        if args.dataset == 'sunrgbd':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            prefixes = _prefixes.copy() + ['all_layers_']
        elif args.dataset == 'scannet':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            prefixes = _prefixes.copy() + ['last_three_'] + ['all_layers_']
    else:
        prefixes = ['proposal_']  # only proposal
        _prefixes = prefixes

    if args.num_decoder_layers >= 3:
        last_three_prefixes = ['last_', f'{args.num_decoder_layers - 2}head_', f'{args.num_decoder_layers - 3}head_']
    elif args.num_decoder_layers == 2:
        last_three_prefixes = ['last_', '0head_']
    elif args.num_decoder_layers == 1:
        last_three_prefixes = ['last_']
    else:
        last_three_prefixes = []

    batch_pred_map_cls_dict = {k: [] for k in prefixes}

    for prefix in prefixes:
        if prefix == 'last_three_':
            end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                                        for ppx in last_three_prefixes], 1)
            end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                                for ppx in last_three_prefixes], 1)
            end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                    for ppx in last_three_prefixes], 1)
            end_points[f'{prefix}size_scores'] = torch.cat([end_points[f'{ppx}size_scores']
                                                            for ppx in last_three_prefixes], 1)
            end_points[f'{prefix}size_residuals'] = torch.cat([end_points[f'{ppx}size_residuals']
                                                                for ppx in last_three_prefixes], 1)
            end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                for ppx in last_three_prefixes], 1)
            end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                    for ppx in last_three_prefixes], 1)

        elif prefix == 'all_layers_':
            end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                                        for ppx in _prefixes], 1)
            end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                                for ppx in _prefixes], 1)
            end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                    for ppx in _prefixes], 1)
            end_points[f'{prefix}size_scores'] = torch.cat([end_points[f'{ppx}size_scores']
                                                            for ppx in _prefixes], 1)
            end_points[f'{prefix}size_residuals'] = torch.cat([end_points[f'{ppx}size_residuals']
                                                                for ppx in _prefixes], 1)
            end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                for ppx in _prefixes], 1)
            end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                    for ppx in _prefixes], 1)

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix)
        batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)

    threshold = 0.25

    proposals = batch_pred_map_cls_dict["last_"][0][0]

    visualier = o3d.visualization.Visualizer()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    for i, proposal in enumerate(proposals):
        if proposal[-1] >= threshold:
            box_points = o3d.utility.Vector3dVector(proposal[1])
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(box_points)

    


if __name__ == '__main__':
    opt = parse_option()
    main(opt, opt.avg_times)