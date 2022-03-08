#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.MA import (
    add_joint_estimation_config,
    register_all_cityscapes_joint,
    register_all_sceneflow,
    register_all_sceneflow_flying3d,
    register_all_kitti_2015,
    register_all_kitti360,
    JointDeeplabDatasetMapper,
    JointEvaluator,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.config import configurable

'''
import itertools
import torch.utils.data as torchdata
from detectron2.config import configurable
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import (
    InferenceSampler,
)
'''

from detectron2.data.build import (
    filter_images_with_only_crowd_annotations,
    filter_images_with_few_keypoints,
    load_proposals_into_dataset,
    print_instances_class_histogram,
    get_detection_dataset_dicts,
    build_batch_data_loader,
    trivial_batch_collator,
    worker_init_reset_seed,
)
from detectron2.data.common import DatasetFromList, MapDataset
import torch.utils.data as torchdata
from detectron2.data.samplers import InferenceSampler


# TODO: the meaning is not clear
import torch.distributed as dist

dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        '''
        与模板相比, 这里对模板中的build_evaluator进行了合并, 本质上还是相同的, 没有大的改变
        '''
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            # evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
            evaluator_list.append(JointEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                    torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            #evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            #evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
            pass    # todo: has been changed
        if evaluator_type == "coco_panoptic_seg":
            pass    # todo: has been changed
            '''
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )
            '''
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        _root_autodl = "/root/autodl-tmp"
        register_all_cityscapes_joint(_root)
        register_all_sceneflow(_root)
        register_all_sceneflow_flying3d(_root)
        register_all_kitti_2015(_root)
        register_all_kitti360(_root)
        mapper = JointDeeplabDatasetMapper(cfg) # TODO: changes
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        """
        与default相比, 这里更改了参数设置, 并且使用了不同的优化器. 
        所有的优化器都在相同的目录里, 按照需要调用即可
        """

        '''
        index_backbone = []
        index_panoptic = []
        index_disparity = []
        index_disparity_4 = []
        index_disparity_8 = []
        index_disparity_16 = []
        for i, p in enumerate(list(model.state_dict())):
            if p.split('.')[0] == 'backbone':
                index_backbone.append(i)
            if p.split('.')[0] in ['sem_seg_head', 'ins_embed_head']:
                index_panoptic.append(i)
            if p.split('.')[0] == 'dis_embed_head':
                index_disparity.append(i)
            if p.split('.')[0] == 'dis_embed_head' and (p.split('.')[2] == '1/4' or p.split('.')[2] == 'res2'):
                index_disparity_4.append(i)
            if p.split('.')[0] == 'dis_embed_head' and (p.split('.')[2] == '1/8' or p.split('.')[2] == 'res3'):
                index_disparity_8.append(i)
            if p.split('.')[0] == 'dis_embed_head' and (p.split('.')[2] == '1/16' or p.split('.')[2] == 'res5'):
                index_disparity_16.append(i)
        '''

        params = model.named_parameters()
        for i, (name, param) in enumerate(params):
            if cfg.SOLVER.FREEZE_BACKBONE and name.split('.')[0] == 'backbone':
                param.requires_grad = False
            if cfg.SOLVER.FREEZE_PANOPTIC and name.split('.')[0] in ['sem_seg_head', 'ins_embed_head']:
                param.requires_grad = False
            if (cfg.MODEL.DIS_EMBED_HEAD.FUSION_MODEL=="multi" or (not cfg.MODEL.MODE.FEATURE_FUSION)) and (name.split('.')[0] == 'dis_embed_head' and name.split('.')[3] == 'fusion_block'):
                param.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY and name.split('.')[0] == 'dis_embed_head':
                param.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY_4 and (name.split('.')[0] == 'dis_embed_head' and (name.split('.')[2] == '1/4' or name.split('.')[2] == 'res2') and name.split('.')[3] != 'fusion_block'):
                param.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY_8 and (name.split('.')[0] == 'dis_embed_head' and (name.split('.')[2] == '1/8' or name.split('.')[2] == 'res3') and name.split('.')[3] != 'fusion_block'):
                param.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY_16 and (name.split('.')[0] == 'dis_embed_head' and (name.split('.')[2] == '1/16' or name.split('.')[2] == 'res5') and name.split('.')[3] != 'fusion_block'):
                param.requires_grad = False


        '''      
        for i, p in enumerate(model.parameters()):
            if cfg.SOLVER.FREEZE_BACKBONE and i in index_backbone:
                p.requires_grad = False
            if cfg.SOLVER.FREEZE_PANOPTIC and i in index_panoptic:
                p.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY and i in index_disparity:
                p.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY_4 and i in index_disparity_4:
                p.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY_8 and i in index_disparity_8:
                p.requires_grad = False
            if cfg.SOLVER.FREEZE_DISPARITY_16 and i in index_disparity_16:
                p.requires_grad = False
        '''

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        # mapper = DatasetMapper(cfg, False)
        mapper = JointDeeplabDatasetMapper(cfg, isVal=True)
    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS}


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0, collate_fn=None):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        num_workers (int): number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    return torchdata.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_joint_estimation_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        '''
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        '''
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=cfg.RESUME)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
