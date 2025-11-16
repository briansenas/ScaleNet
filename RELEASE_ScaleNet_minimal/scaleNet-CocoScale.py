import datetime
import argparse
import inspect
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from dataset_coco_pickle_eccv import COCO2017ECCV
from dataset_coco_pickle_eccv import my_collate
from dataset_cvpr import my_collate_SUN360
from dataset_cvpr import SUN360Horizon
from maskrcnn_rui.config import cfg as CFG
from maskrcnn_rui.data.transforms import build_transforms_maskrcnn
from maskrcnn_rui.data.transforms import build_transforms_yannick
from maskrcnn_rui.utils.comm import get_rank, synchronize
from models.model_RCNNOnly_combine_indeptPointnet_maskrcnnPose_discount import (
    RCNNOnly_combine,
)
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from train_batch_combine_RCNNOnly_v5_pose_multiCat import train_batch_combine
from utils.checkpointer import DetectronCheckpointer
from utils.data_utils import make_data_loader
from utils.eval_save_utils_combine_RCNNONly import check_eval_COCO
from utils.eval_save_utils_combine_RCNNONly import check_save
from utils.logger import printer as PRINTER
from utils.logger import setup_logger
from utils.model_utils import get_bins_combine
from utils.train_utils import reduce_loss_dict
from utils.utils_misc import colored


def train(rank, opt):
    opt.debug = False
    opt.checkpoints_folder = "checkpoint"
    config_file = opt.config_file
    CFG.merge_from_file(config_file)
    CFG.merge_from_list(["MODEL.DEVICE", "cuda"])
    CFG.merge_from_list(opt.opts)
    CFG.freeze()
    opt.cfg = CFG
    device = "cuda"
    opt.device = device
    opt.local_rank = rank
    opt.rank = rank
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30)
    )
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    torch.cuda.set_device(rank)

    summary_path = "./summary/" + opt.task_name
    writer = SummaryWriter(summary_path)

    logger = setup_logger(
        "logger:train",
        summary_path,
        get_rank(),
        filename="logger_maskrcn-style.txt",
    )
    logger.info(colored("==[RANK %s]==" % rank, "white", "on_blue"))
    logger.info(colored("==[GPUS TOTAL %s]==" % num_gpus, "white", "on_blue"))
    logger.info(colored("==[config]== opt", "white", "on_blue"))
    logger.info(colored("==[config]== cfg", "white", "on_blue"))
    logger.info(
        colored(
            f"==[config]== Loaded configuration file {opt.config_file}",
            "white",
            "on_blue",
        ),
    )
    printer = PRINTER(get_rank(), debug=opt.debug)

    modules_not_build = []
    if not opt.train_cameraCls:
        modules_not_build.append("classifier_heads")
    if not opt.train_roi_h:
        modules_not_build.append("roi_h_heads")
    if not opt.est_bbox and not opt.est_kps:
        modules_not_build.append("roi_bbox_heads")
    sys.path.insert(0, "models/pointnet")
    model = RCNNOnly_combine(
        opt,
        logger,
        printer,
        num_layers=opt.num_layers,
        modules_not_build=modules_not_build,
    )

    model.to(rank)
    model.init_restore()
    model.turn_on_all_params()
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    # NOTE: to find unused parameters ourselves:
    # used = set()
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         p.register_hook(lambda grad, n=name: used.add(n))

    optimizer = optim.Adam(
        model.parameters(),
        lr=CFG.SOLVER.BASE_LR,
        betas=(opt.beta1, 0.999),
        eps=1e-5,
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10, cooldown=5)
    earlystop = 15

    opt.checkpoints_path_task = os.path.join(opt.checkpoints_folder, opt.task_name)
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        opt,
        model,
        optimizer,
        scheduler,
        opt.checkpoints_folder,
        opt.checkpoints_path_task,
        save_to_disk,
        logger=logger,
        if_print=False,
    )
    tid_start = 0
    epoch_start = 0
    if opt.resume != "NoCkpt":
        try:
            checkpoint_restored, _, _ = checkpointer.load(task_name=opt.task_name)
        except (ValueError, FileNotFoundError):
            checkpoint_restored, _, _ = checkpointer.load(
                f=os.path.join(opt.checkpoints_folder, opt.resume),
            )
        if "iteration" in checkpoint_restored:
            tid_start = checkpoint_restored["iteration"]
        if "epoch" in checkpoint_restored:
            epoch_start = checkpoint_restored["epoch"]
        logger.info(
            colored(
                "Restoring from epoch %d - iter %d" % (epoch_start, tid_start),
                "white",
                "on_blue",
            ),
        )

    train_trnfs_maskrcnn = build_transforms_maskrcnn(CFG, True)
    eval_trnfs_maskrcnn = build_transforms_maskrcnn(CFG, False)
    train_trnfs_yannick = build_transforms_yannick(CFG, True)
    eval_trnfs_yannick = build_transforms_yannick(CFG, False)

    ds_train_coco_vis = COCO2017ECCV(
        transforms_yannick=train_trnfs_yannick,
        transforms_maskrcnn=train_trnfs_maskrcnn,
        split="train",
        shuffle=False,
        logger=logger,
        opt=opt,
    )
    ds_eval_coco_vis = COCO2017ECCV(
        transforms_yannick=eval_trnfs_yannick,
        transforms_maskrcnn=eval_trnfs_maskrcnn,
        split="val",
        shuffle=False,
        logger=logger,
        opt=opt,
    )
    training_loader_coco_vis = make_data_loader(
        CFG,
        ds_train_coco_vis,
        is_train=True,
        is_distributed=opt.distributed,
        start_iter=tid_start,
        logger=logger,
        collate_fn=my_collate,
        batch_size_override=-1,
    )
    eval_loader_coco_vis = make_data_loader(
        CFG,
        ds_eval_coco_vis,
        is_train=False,
        is_distributed=opt.distributed,
        logger=logger,
        collate_fn=my_collate,
        batch_size_override=-1,
    )

    ds_train_SUN360 = SUN360Horizon(
        transforms=train_trnfs_maskrcnn,
        train=True,
        logger=logger,
    )
    train_loader_SUN360 = make_data_loader(
        CFG,
        ds_train_SUN360,
        is_train=True,
        is_distributed=opt.distributed,
        start_iter=tid_start,
        logger=logger,
        collate_fn=my_collate_SUN360,
        batch_size_override=-1,
    )
    results_path = "train_results"
    Path(results_path).mkdir(exist_ok=True)
    task_name = opt.resume
    task_name_appendix = "-phaseOne"
    task_name += task_name_appendix

    write_folder = os.path.join(results_path, task_name)
    for subfolder in ["", "png", "npy", "pickle", "results"]:
        Path(os.path.join(write_folder, subfolder)).mkdir(parents=True, exist_ok=True)

    if_vis = False
    opt.zero_pitch = False
    if_vis = False

    bins = get_bins_combine(device)
    tid = 0

    epoch = 0
    epochs_evalued = []

    loss_func = torch.nn.L1Loss()
    if opt.distributed:
        rank = dist.get_rank()
    else:
        rank = 0
    patience = 0
    logger.info(
        f"Starting at iteration {tid_start} to complete {opt.iter} for a total of {opt.iter - tid_start}.",
    )
    model.train()
    synchronize()
    for i, coco_data, sun360_data in tqdm(
        zip(
            range(0, opt.iter - tid_start),
            training_loader_coco_vis,
            train_loader_SUN360,
        ),
        total=opt.iter,
        initial=tid_start,
    ):
        optimizer.zero_grad()
        (
            _,
            inputCOCO_Image_maskrcnnTransform_list,
            W_batch_array,
            H_batch_array,
            yc_batch,
            bboxes_batch_array,
            bboxes_length_batch_array,
            v0_batch,
            f_pixels_yannick_batch,
            im_filename,
            im_file,
            target_maskrcnnTransform_list,
            labels_list,
        ) = coco_data
        (
            _,
            inputSUN360_Image_yannickTransform_list,
            horizon_dist_gt,
            pitch_dist_gt,
            roll_dist_gt,
            vfov_dist_gt,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            W_list,
            H_list,
            _,
            _,
            _,
            _,
        ) = sun360_data
        tid = i + tid_start
        is_better = False
        input_dict = {
            "inputCOCO_Image_maskrcnnTransform_list": inputCOCO_Image_maskrcnnTransform_list,
            "W_batch_array": W_batch_array,
            "H_batch_array": H_batch_array,
            "yc_batch": yc_batch,
            "bboxes_batch_array": bboxes_batch_array,
            "bboxes_length_batch_array": bboxes_length_batch_array,
            "v0_batch": v0_batch,
            "f_pixels_yannick_batch": f_pixels_yannick_batch,
            "im_filename": im_filename,
            "im_file": im_file,
            "bins": bins,
            "target_maskrcnnTransform_list": target_maskrcnnTransform_list,
            "labels_list": labels_list,
            "horizon_dist_gt": horizon_dist_gt,
            "pitch_dist_gt": pitch_dist_gt,
            "roll_dist_gt": roll_dist_gt,
            "vfov_dist_gt": vfov_dist_gt,
            "W_list": W_list,
            "H_list": H_list,
            "inputSUN360_Image_yannickTransform_list": inputSUN360_Image_yannickTransform_list,
        }
        bins = input_dict["bins"]
        loss_dict, _ = train_batch_combine(
            input_dict,
            model,
            device,
            opt,
            is_training=True,
            tid=tid,
            loss_func=loss_func,
            rank=rank,
            if_SUN360=True,
            if_vis=if_vis,
        )
        loss_vt = loss_dict.get("loss_vt", 0.0)
        loss_person = loss_dict.get("loss_person", 0.0)
        loss_detection = (
            loss_dict["loss_bbox_cls"] + loss_dict["loss_bbox_reg"]
            if opt.est_bbox
            else 0.0
        )
        loss_kp = loss_dict["loss_kp"] if opt.est_kps else 0.0
        calib_loss = (
            (
                loss_dict["loss_horizon"]
                + loss_dict["loss_pitch"]
                + loss_dict["loss_roll"]
                + loss_dict["loss_vfov"]
            )
            if opt.train_cameraCls
            else 0.0
        )
        total_loss = sum(loss for loss in loss_dict.values())
        total_loss.backward()
        # after loss.backward():
        # NOTE: When attempting to find unused parameters in DDP
        # print(
        #     "UNUSED:",
        #     [
        #         n
        #         for n, p in model.named_parameters()
        #         if p.requires_grad and n not in used
        #     ],
        # )
        synchronize()
        optimizer.step()
        if tid % opt.summary_every_iter == 0:
            logger.info(
                f"Epoch {epoch} Iter {tid}: Loss VT = {loss_vt:.4f} "
                + f"Loss Person = {loss_person:.4f} Calib Loss = {calib_loss:.4f} "
                + f"Detection Loss = {loss_detection:.4f} Kps Loss = {loss_kp:.4f} "
                + f"Total Loss = {total_loss:.4f} ",
            )
        if opt.save_every_iter != 0 and tid % opt.save_every_iter == 0 and i > 0:
            logger.info(f"Checking to save at {tid}")
            check_save(
                rank=rank,
                tid=tid,
                epoch_save=epoch,
                epoch_total=epoch,
                opt=opt,
                checkpointer=checkpointer,
                checkpoints_folder=opt.checkpoints_folder,
                logger=logger,
                is_better=is_better,
            )
        # After computing loss_dict and other stats for this tid/epoch
        if i != 0 and tid % (len(training_loader_coco_vis) - 1) == 0:
            print("Evaluate the model")
            epoch += 1
            is_better = check_eval_COCO(
                tid=tid,
                epoch=epoch,
                rank=rank,
                opt=opt,
                model=model,
                eval_loader=eval_loader_coco_vis,
                writer=writer,
                device=device,
                bins=bins,
                logger=logger,
                scheduler=scheduler,
                epochs_evalued=epochs_evalued,
            )

            check_save(
                rank=rank,
                tid=tid,
                epoch_save=epoch,
                epoch_total=epoch,
                opt=opt,
                checkpointer=checkpointer,
                checkpoints_folder=opt.checkpoints_folder,
                logger=logger,
                is_better=is_better,
            )

            if not is_better:
                patience += 1
            else:
                patience = 0
            if patience >= earlystop:
                logger.info(f"[EarlyStopping] No improvements over {patience} epochs")
                logger.info(
                    f"Epoch {epoch} Iter {tid}: Loss VT = {loss_vt:.4f} "
                    + f"Loss Person = {loss_person:.4f} Calib Loss = {calib_loss:.4f} "
                    + f"Detection Loss = {loss_detection:.4f} Kps Loss = {loss_kp:.4f} "
                    + f"Total Loss = {total_loss:.4f} ",
                )
                patience = 0
                break

    if opt.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    print(current_dir)
    sys.path.insert(0, current_dir)
    torch_version = torch.__version__
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    seed = 140421
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(
        description="Rui's Scale Estimation Network Training"
    )
    # Training
    parser.add_argument("--task_name", type=str, default="tmp", help="resume training")
    parser.add_argument(
        "--workers",
        type=int,
        help="number of data loading workers",
        default=8,
    )
    parser.add_argument(
        "--save_every_iter",
        type=int,
        default=100,
        help="set to 0 to save ONLY at the end of each epoch",
    )
    parser.add_argument("--summary_every_iter", type=int, default=100, help="")
    parser.add_argument(
        "--iter",
        type=int,
        default=10000,
        help="number of iterations to train for",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 for adam. default=0.5",
    )
    parser.add_argument(
        "--not_val",
        action="store_true",
        help="Do not validate duruign training",
    )
    parser.add_argument("--not_vis", action="store_true", help="")
    parser.add_argument("--not_vis_SUN360", action="store_true", help="")
    parser.add_argument(
        "--save_every_epoch",
        type=int,
        default=1,
        help="save checkpoint every ? epoch",
    )
    parser.add_argument(
        "--vis_every_epoch", type=int, default=5, help="vis every ? epoch"
    )
    # Model
    parser.add_argument(
        "--accu_model",
        action="store_true",
        help="Use accurate model with theta instead of Derek's approx.",
    )
    parser.add_argument("--argmax_val", action="store_true", help="")
    parser.add_argument(
        "--direct_camH",
        action="store_true",
        help="direct preidict one number for camera height ONLY, instead of predicting a distribution",
    )
    parser.add_argument(
        "--direct_v0",
        action="store_true",
        help="direct preidict one number for v0 ONLY, instead of predicting a distribution",
    )
    parser.add_argument(
        "--direct_fmm",
        action="store_true",
        help="direct preidict one number for fmm ONLY, instead of predicting a distribution",
    )

    # Pre-training
    parser.add_argument(
        "--resume",
        type=str,
        help="resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp)",
        default="NoCkpt",
    )
    parser.add_argument(
        "--feature_only",
        action="store_true",
        help="restore only features (remove all classifiers) from checkpoint",
    )
    parser.add_argument(
        "--reset_scheduler",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument("--reset_lr", action="store_true", help="")  # NOT working yet

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--master_port", type=str, default="8914")

    # DEBUG
    parser.add_argument("--debug", action="store_true", help="Debug eval")
    parser.add_argument("--debug_memory", action="store_true", help="Debug eval")

    # Mask R-CNN
    # Modules
    parser.add_argument(
        "--train_cameraCls",
        action="store_true",
        help="Disable camera calibration network",
    )
    parser.add_argument("--train_roi_h", action="store_true", help="")
    parser.add_argument(
        "--est_bbox",
        action="store_true",
        help="Enable estimating bboxes instead of using GT bboxes",
    )
    parser.add_argument(
        "--est_kps",
        action="store_true",
        help="Enable estimating keypoints",
    )
    parser.add_argument("--if_discount", action="store_true", help="")
    parser.add_argument("--discount_from", type=str, default="pred")  # ('GT', 'pred')

    # Losses
    parser.add_argument(
        "--loss_last_layer",
        action="store_true",
        help="Using loss of last layer only",
    )
    parser.add_argument(
        "--loss_person_all_layers",
        action="store_true",
        help="Using loss of last layer only",
    )
    parser.add_argument(
        "--not_rcnn",
        action="store_true",
        help="Disable Mask R-CNN person height bbox head",
    )
    parser.add_argument("--no_kps_loss", action="store_true", help="")

    # Archs
    parser.add_argument("--pointnet_camH", action="store_true", help="")
    parser.add_argument("--pointnet_camH_refine", action="store_true", help="")
    parser.add_argument("--pointnet_personH_refine", action="store_true", help="")
    parser.add_argument(
        "--pointnet_roi_feat_input",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument(
        "--pointnet_roi_feat_input_person3",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument(
        "--pointnet_fmm_refine",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument(
        "--pointnet_v0_refine",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument(
        "--not_pointnet_detach_input",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fit_derek", action="store_true", help="")
    # weights
    parser.add_argument(
        "--weight_SUN360",
        type=float,
        default=1.0,
        help="weight for Yannick's losses. default=1.",
    )
    parser.add_argument(
        "--weight_kps",
        type=float,
        default=10,
        help="weight for Yannick's losses. default=1.",
    )

    # debug
    parser.add_argument("--zero_pitch", action="store_true", help="")  # NOT working yet

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    opt = parser.parse_args(
        (
            "--task_name SUN360RCNN "
            "--est_bbox --est_kps "
            + "--train_cameraCls "
            + "--train_roi_h "
            + "--accu_model "
            + "--loss_person_all_layers "
            + "--num_layers 3 "
            + "--pointnet_camH "
            + "--pointnet_camH_refine --pointnet_personH_refine "
            + "--config-file config/coco_config_small_synBN1108_kps.yaml  "
            + "--resume checkpointer_epoch0055_iter0136785.pth "
        ).split(),
    )
    num_gpus = torch.cuda.device_count()
    world_size = int(os.environ["WORLD_SIZE"])
    opt.distributed = num_gpus > 1
    # // Only if I don't use tasks-per-node
    # if opt.distributed:
    # 	torch.multiprocessing.spawn(train, nprocs=world_size, args=[opt])
    # else:
    #
    rank = int(os.environ["SLURM_PROCID"])
    print("My slurm local rank is: ", rank)
    train(rank, opt)
