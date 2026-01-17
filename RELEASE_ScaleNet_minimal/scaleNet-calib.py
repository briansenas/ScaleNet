import datetime
from statistics import mean
import argparse
import inspect
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from dataset_cvpr import my_collate_SUN360
from dataset_cvpr import SUN360Horizon
from maskrcnn_rui.config import cfg as CFG
from maskrcnn_rui.data.transforms import build_transforms_maskrcnn
from maskrcnn_rui.utils.comm import get_rank, synchronize
from models.model_RCNN_only import RCNN_only
from eval_epoch_cvpr_RCNN import eval_epoch_cvpr_RCNN
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.checkpointer import DetectronCheckpointer
from utils.data_utils import make_data_loader
from utils.eval_save_utils_combine_RCNNONly import check_eval_SUN360, check_save
from utils.logger import printer as PRINTER
from utils.logger import setup_logger
from utils.train_utils import reduce_loss_dict
from utils.model_utils import oneLargeBboxList
from utils.train_utils import process_sun360_losses
from utils.utils_misc import colored
from collections import defaultdict


def logger_report(epoch, tid, toreport, logger):
    loss_horizon = toreport["horizon"]
    loss_pitch = toreport["pitch"]
    loss_vfov = toreport["vfov"]
    loss_roll = toreport["roll"]
    loss_reduced = toreport["loss"]
    logger.info(
        f"Epoch {epoch} Iter {tid}: Loss horizon = {loss_horizon:.4f} "
        + f"Loss pitch = {loss_pitch:.4f} "
        + f"Loss vfov = {loss_vfov:.4f} "
        + f"Loss roll = {loss_roll:.4f} "
        + f"Calib Loss = {loss_reduced:.4f} "
    )


def train(rank, opt):
    opt.checkpoints_folder = "checkpoint"
    config_file = opt.config_file
    CFG.merge_from_file(config_file)
    CFG.merge_from_list(["MODEL.DEVICE", "cuda"])
    CFG.merge_from_list(opt.opts)
    CFG.freeze()
    opt.cfg = CFG
    device = "cuda"
    opt.device = device
    opt.rank = rank
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30)
    )
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    torch.cuda.set_device(rank)

    summary_path = "./summary/" + opt.task_name
    if rank == 0:
        writer = SummaryWriter(summary_path)
    else:
        # NOTE: this will make the code fail if not checked properly
        writer = None

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
    model = RCNN_only(
        CFG,
        opt,
        logger,
        printer,
        rank=rank,
    )

    model.to(rank)
    model.init_restore()
    model = DDP(
        model,
        device_ids=[rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=CFG.SOLVER.BASE_LR,
        betas=(opt.beta1, 0.999),
        eps=1e-5,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=20, cooldown=10
    )
    earlystop = 30

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
        if_reset_scheduler=opt.reset_scheduler,
        if_reset_lr=opt.reset_lr,
    )
    tid_start = 0
    epoch_start = 0
    if opt.resume != "NoCkpt":
        try:
            checkpoint_restored, _, _ = checkpointer.load(task_name=opt.resume)
        except (ValueError, FileNotFoundError):
            checkpoint_restored, _, _ = checkpointer.load(
                f=os.path.join(opt.checkpoints_folder, opt.resume),
            )
        if not opt.reset_iter:
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
    ds_train_SUN360 = SUN360Horizon(
        transforms=train_trnfs_maskrcnn,
        train=True,
        logger=logger,
        json_name=opt.calib_file,
        debug=opt.debug,
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
    ds_eval_SUN360 = SUN360Horizon(
        transforms=eval_trnfs_maskrcnn,
        train=False,
        logger=logger,
        json_name=opt.calib_file,
        debug=opt.debug,
    )
    eval_loader_SUN360 = make_data_loader(
        CFG,
        ds_eval_SUN360,
        is_train=False,
        is_distributed=opt.distributed,
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

    opt.zero_pitch = False
    tid = 0

    epoch = 0
    ctx = defaultdict(list)
    if opt.distributed:
        rank = dist.get_rank()
    else:
        rank = 0
    patience = 0
    evaluate_at_every = (
        len(train_loader_SUN360.batch_sampler.batch_sampler)
        if opt.evaluate_every == -1
        else int(opt.evaluate_every)
    )
    logger.info(
        f"Starting at iteration {0}, batch skipping first {tid_start % evaluate_at_every},  to complete {opt.iter} for a total of {opt.iter - tid_start} actual trained batches.",
    )
    if not opt.not_val:
        logger.info(
            f"Evaluating at every {evaluate_at_every} iteration",
        )
    loss_func = nn.CrossEntropyLoss()
    model.train()
    synchronize()
    train_bar = tqdm(
        total=opt.iter,
        initial=0,
        desc="Training",
    )
    eval_bar = tqdm(
        total=evaluate_at_every,
        initial=0,
        desc="Epoch",
        position=1,
    )
    epochs_evalued = []
    skip_for = tid_start % evaluate_at_every
    for i, pano_data in zip(
        range(0, opt.iter),
        train_loader_SUN360,
    ):
        train_bar.update(1)
        eval_bar.update(1)
        tid = i
        if i < skip_for:
            # SequentialBatch logic -- must have the seed set correctly
            # Skip until we find the same batch as the last iteration (only within an epoch)
            # Worst case are the last iterations of an epoch -- O(n)
            continue
        optimizer.zero_grad()
        is_better = False
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
            idx1,
            idx2,
            idx3,
            idx4,
        ) = pano_data
        horizon_dist_gt, pitch_dist_gt, roll_dist_gt, vfov_dist_gt = (
            horizon_dist_gt.to(device),
            pitch_dist_gt.to(device),
            roll_dist_gt.to(device),
            vfov_dist_gt.to(device),
        )
        horizon_idx_gt, pitch_idx_gt, roll_idx_gt, vfov_idx_gt = (
            idx1.to(device),
            idx2.to(device),
            idx3.to(device),
            idx4.to(device),
        )
        list_of_oneLargeBbox_list_cpu = oneLargeBboxList(
            W_list,
            H_list,
        )
        list_of_oneLargeBbox_list = [
            bbox_list_array.to(device)
            for bbox_list_array in list_of_oneLargeBbox_list_cpu
        ]

        input_dict_misc = {
            "rank": rank,
            "data": "SUN360",
            "device": device,
            "tid": tid,
        }
        output_RCNN = model(
            input_dict_misc=input_dict_misc,
            image_batch_list=inputSUN360_Image_yannickTransform_list,
            list_of_oneLargeBbox_list=list_of_oneLargeBbox_list,
        )
        loss_horizon = loss_func(output_RCNN["output_horizon"], horizon_idx_gt)
        loss_pitch = loss_func(output_RCNN["output_pitch"], pitch_idx_gt)
        loss_roll = loss_func(output_RCNN["output_roll"], roll_idx_gt)
        loss_vfov = loss_func(output_RCNN["output_vfov"], vfov_idx_gt)
        loss_dict = {
            "loss_horizon": loss_horizon,
            "loss_pitch": loss_pitch,
            "loss_roll": loss_roll,
            "loss_vfov": loss_vfov,
        }
        loss_dict_reduced = reduce_loss_dict(
            loss_dict,
            mark=i,
            logger=logger,
        )
        loss_reduced = sum(loss for loss in loss_dict_reduced.values())
        toreport = {
            "loss": loss_reduced.item(),
            "horizon": loss_horizon.item(),
            "pitch": loss_pitch.item(),
            "roll": loss_roll.item(),
            "vfov": loss_vfov.item(),
        }
        train_bar.set_postfix(**toreport)
        train_bar.update()
        loss_reduced.backward()
        synchronize()
        optimizer.step()
        if tid % opt.summary_every_iter == 0:
            ctx = process_sun360_losses(
                tid,
                loss_dict,
                ctx,
                logger,
            )
            for name in ["horizon", "pitch", "vfov", "roll"]:
                writer.add_scalar(
                    f"loss_train/train_{name}_loss",
                    mean(ctx[f"loss_{name}"]),
                    tid,
                )
            writer.add_scalar(
                "loss_train/train_calib_total_loss",
                mean(ctx["total_loss"]),
                tid,
            )
            logger_report(epoch, tid, toreport, logger)
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
        if i != 0 and tid % (evaluate_at_every) == 0:
            print("Evaluate the model")
            epoch += 1
            is_better = check_eval_SUN360(
                tid=tid,
                epoch=epoch,
                rank=rank,
                opt=opt,
                model=model,
                eval_loader=eval_loader_SUN360,
                writer=writer,
                device=device,
                logger=logger,
                scheduler=scheduler,
                epochs_evalued=epochs_evalued,
            )
            model.train()
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
                logger_report(epoch, tid, toreport, logger)
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
    torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
    seed = 140421
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(
        description="Rui's Scale Estimation Network Training"
    )
    # Training
    parser.add_argument("--task-name", type=str, default="tmp", help="resume training")
    parser.add_argument(
        "--workers",
        type=int,
        help="number of data loading workers",
        default=8,
    )
    parser.add_argument(
        "--save-every-iter",
        type=int,
        default=1000,
        help="set to 0 to save ONLY at the end of each epoch",
    )
    parser.add_argument("--summary-every-iter", type=int, default=100, help="")
    parser.add_argument("--evaluate-every", type=int, default=5000, help="")
    parser.add_argument(
        "--iter",
        type=int,
        default=90000,
        help="number of iterations to train for",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 for adam. default=0.5",
    )
    parser.add_argument(
        "--not-val",
        action="store_true",
        help="Do not validate during training",
    )
    parser.add_argument(
        "--save-every-epoch",
        type=int,
        default=1,
        help="save checkpoint every ? epoch",
    )
    parser.add_argument(
        "--vis-every-epoch", type=int, default=5, help="vis every ? epoch"
    )
    parser.add_argument(
        "--est_bbox",
        action="store_true",
        help="Enable estimating bboxes instead of using GT bboxes",
    )

    # Pre-training
    parser.add_argument(
        "--resume",
        type=str,
        help="resume training; can be full path (e.g. tmp/checkpoint0.pth.tar) or taskname (e.g. tmp)",
        default="1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug",
    )
    parser.add_argument(
        "--feature-only",
        action="store_true",
        help="restore only features (remove all classifiers) from checkpoint",
    )
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="",
    )  # NOT working yet
    parser.add_argument("--reset-lr", action="store_true", help="")  # NOT working yet
    parser.add_argument("--reset-iter", action="store_true", help="")  # NOT working yet

    # DEBUG
    parser.add_argument("--debug", action="store_true", help="Debug eval")

    parser.add_argument(
        "--config-file",
        default="config/coco_config_small_RCNNOnly.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--calib-file",
        default="config/pano360_crops_dataset_cvpr_myDistWider_train.json",
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

    opt = parser.parse_args()
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
