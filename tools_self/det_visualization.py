import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path
import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from datasets.torchvision_datasets import DetectionTest
from datasets.coco import make_coco_transforms

# from datasets.PCB.calibration_layer import PrototypicalCalibrationBlock


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)

    # test args
    parser.add_argument('--root', default='', help='abs path of dir / .txt /.jpg, each in .txt is a abs path of a image ')
    parser.add_argument('--save_dir', default='exps/vis_test', help='path where to save, empty for no saving')
    parser.add_argument('--score_thresh', default=0.35, type=float, help='score thresh')
    # PCB args 
    parser.add_argument('--pcb_enable', default=False, action='store_true', help='weather use pcb during eval')
    parser.add_argument('--pcb_model_path', default='surgery_model/resnet101-5d3b4d8f.pth',type=str, help='PCB resnet model imagenet pretrain weight path') 
    parser.add_argument('--pcb_model_type', default='resnet',type=str, help='PCB model type, now only support resnet')
    parser.add_argument('--pcb_upper', default=1.0, type=float, help='TODO')
    parser.add_argument('--pcb_lower', default=0.05, type=float, help='TODO')
    parser.add_argument('--pcb_alpha', default=0.5, type=float, help='TODO')
    parser.add_argument('--pcb_batch_size', default=2, type=int, help='batch size in pcb build proto with resnet101')

    
    # stephen add argumens:
    parser.add_argument('--dataset_name', default='coco_base', type=str, help='coco_base, coco_all, coco_{novel / all}_seed_{s}_{k}_shot')
    '''
    使用dataset_name 来决定是否过滤相应类别
        coco_base  : only fetch 60 base class from tranvalno5k.json
        coco_novel_seed_{s}_{k}_shot : only fetch 20 novel class from coco_novel_seed_{s}_{k}_shot.json
        coco_all_seed_{s}_{k}_shot : all 80 class , no filter.
    '''
    parser.add_argument('--num_classes', default='60', type=int)
    parser.add_argument('--eval_dataset', default='coco_base', type = str, help = 'coco_base, coco_all, coco_novel')
    parser.add_argument('--filter_kind', default=None, type=str, help='filter dataset')
    parser.add_argument('--freeze_transformer', default = False, action='store_true')


    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float) # if  > 0, train backbone. else not train backbone
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser




def main():
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval() # model.eval() 的位置有要求吗？比如说在加载checkpoint之后？

    
    dataset_val = DetectionTest(args.root, transforms=make_coco_transforms('val'))
    print(len(dataset_val))
    # sample, target = dataset_val[0]
    # print(sample.shape)
    # print(target)
    # return 
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    dataloader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    # save_dir = Path(args.save_dir)

    ### load checkpoint 
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # '''
        print('resume from {} done !'.format(args.resume))
    
    ### forward

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    ### stephen add for PCB !!!
    pcb = None
    if args is not None and args.pcb_enable :
        pcb = PrototypicalCalibrationBlock(args)


    for samples, targets in dataloader_val:
        samples = samples.to(device)
        paths = [target['path'] for target in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'path'} for t in targets]
        # paths = [target['path'] for target in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        ### insert PCB module !!!
        if pcb is not None:
            aug_size = torch.stack([t["size"] for t in targets], dim=0)
            scale = aug_size / orig_target_sizes
            results = pcb.execute_calibration(samples.tensors, results, scale)

        results = [[v.cpu() for k, v in t.items()] for t in results]
    
        # save res for visulization
        score_thresh = args.score_thresh


        for i in range(len(results)):
            scores, labels, boxes = results[i]
            # print(scores[:10])
            # print(boxes[:10])
            image_path =paths[i]
            img = cv2.imread(image_path)
            j = 0
            while scores[j] > score_thresh:
                cv2.rectangle(img, (int(boxes[j][0]), int(boxes[j][1])), (int(boxes[j][2]), int(boxes[j][3])), (0, 0, 255), thickness=1) 
                j = j + 1
            save_path = os.path.join(args.save_dir, image_path.split('/')[-1])
            cv2.imwrite(save_path, img)
            print('save det res in {}'.format(save_path))



if __name__ == "__main__":
    main()