from tqdm import tqdm
import logging
import ntpath
from utils.utils_misc import white_blue 
from utils.utils_misc import magenta 
from utils.utils_misc import print_white_blue 
from utils.utils_misc import colored
from utils import utils_coco
from utils.utils_coco import bin2midpointpitch
from maskrcnn_benchmark.utils.comm import get_world_size
import glob
import os
import numpy as np
import torch.distributed as dist
import shutil
import sys

import torch

torch_version = torch.__version__


pwdpath = os.getcwd()
checkpoints_path = pwdpath + "/checkpoint"
summary_path = pwdpath + "/summary"
summary_vis_path = pwdpath + "/summary_vis"


y_person = 1.75


class Logger:
    def __init__(self, filename="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def save_checkpoint(
    state,
    is_best,
    task_name,
    filename="checkpoint.pth.tar",
    save_folder=checkpoints_path,
    logger=None,
):
    save_path = os.path.join(os.path.join(save_folder, task_name), filename)
    best_path = os.path.join(os.path.join(save_folder, task_name), "best_" + filename)
    latest_path = os.path.join(os.path.join(save_folder, task_name), "latest.pth.tar")

    if not os.path.isdir(os.path.join(save_folder, task_name)):
        os.mkdir(os.path.join(save_folder, task_name))

    save_path = os.path.join(os.path.join(save_folder, task_name), filename)
    torch.save(state, save_path)
    logger.info(colored("Saved to " + save_path, "white", "on_magenta"))

    if is_best:
        print("best", state["eval_loss"])
        shutil.copyfile(save_path, best_path)
    else:
        print("NOT best", state["eval_loss"])

    shutil.copyfile(save_path, latest_path)


def clean_up_checkpoints(
    checkpoint_folder, leave_N, start_with="checkpoint_", logger=None
):
    list_checkpoints = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/%s*.*" % start_with))
    )
    list_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    last_checkpoint_file = os.path.join(checkpoint_folder, "last_checkpoint")
    try:
        with open(last_checkpoint_file) as f:
            last_saved = f.read()
            last_saved = last_saved.strip()
    except OSError:
        last_saved = None
        pass

    if logger is None:
        logger = logging.getLogger("clean_up_checkpoints")

    if len(list_checkpoints) > leave_N:
        for checkpoint_path in list_checkpoints[leave_N:]:
            if last_saved is not None and ntpath.basename(
                last_saved
            ) == ntpath.basename(checkpoint_path):
                logger.info(magenta("Skipping latest at " + last_saved))
                continue
            os.system("rm %s" % checkpoint_path)
            logger.info(white_blue("removed checkpoint at " + checkpoint_path))


def printensor(msg):
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}")
            print(f"min/max/mean: {tensor.max()}, {tensor.min()}, {tensor.mean()}")

    return printer


def summary_thres(array, threses, writer, name, tid):
    assert len(array.shape) == 1, "summary_thres only applies to array of shape [N,]!"
    thres_ratio_dict = {}
    for thres in threses:
        # print(array.shape)
        ratio = np.sum(array < thres) / array.shape[0]
        writer.add_scalar("%s_thres%.2f" % (name, thres), ratio, tid)
        # print(np.sum(array<thres) / array.shape[0])
        thres_ratio_dict.update({str(thres): ratio})
    return thres_ratio_dict


def load_densenet(model, ckpt_name, checkpoints_folder, opt, old=False):
    if ".pth.tar" in ckpt_name:
        checkpoint_path = os.path.join(opt.checkpoints_folder, ckpt_name)
    else:
        checkpoint_path = os.path.join(
            opt.checkpoints_folder, ckpt_name + "/latest.pth.tar"
        )
    checkpoint = torch.load(checkpoint_path)

    pretrained_state_dict = checkpoint["state_dict"]
    if torch_version >= "1.3.0":
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items()}
    else:
        pretrained_state_dict = {
            k.replace(".layers.denselayer", ".denselayer"): v
            for k, v in pretrained_state_dict.items()
        }

    if opt.feature_only:
        if torch_version == "1.3.0":
            pretrained_state_dict = {
                ("Densenet." + k).replace(".denselayer", ".layers.denselayer"): v
                for k, v in pretrained_state_dict.items()
                if "classifier" not in k
            }
        else:
            pretrained_state_dict = {
                ("Densenet." + k): v
                for k, v in pretrained_state_dict.items()
                if "classifier" not in k
            }

    best_loss = checkpoint["eval_loss"]
    state = model.state_dict()
    state.update(pretrained_state_dict)
    model.load_state_dict(state)

    best_loss = checkpoint["eval_loss"]
    del checkpoint

    return model, best_loss, checkpoint_path


def sum_bbox_ratios(writer, vt_loss_allBoxes_list, tid, prefix, title):
    vt_loss_allBoxes_np = torch.stack(vt_loss_allBoxes_list).cpu().data.numpy().copy()
    vt_loss_allBoxes_np.sort()
    thres_ratio_dict = summary_thres(
        vt_loss_allBoxes_np,
        [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        writer,
        "thres_%s/%s" % (prefix, title),
        tid,
    )
    return vt_loss_allBoxes_np, thres_ratio_dict


def reduce_loss_dict(loss_dict, mark="", if_print=False, logger=None):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()  # NUM of GPUs
    if world_size < 2:
        logger.debug("[train_utils] world_size==%d; not reduced!" % world_size)
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
            # print(k, loss_dict[k].shape, loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        if if_print:
            print(mark, "-0-all_losses", all_losses)
        dist.reduce(all_losses, dst=0)
        if if_print:
            print(mark, "-1-all_losses", all_losses)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
        if if_print:
            print(mark, "-2-reduced_losses", reduced_losses)
    return reduced_losses


def print_f_info(f_estim, f_pixels_yannick_batch, input_dict):
    print_white_blue("-----f_estim, f_pixels_yannick_batch-----")
    print(f_estim.cpu().detach().numpy())
    print(f_pixels_yannick_batch.cpu().detach().numpy())
    f_mm_list_yannick = []
    for f_single, H_single, W_single in zip(
        f_pixels_yannick_batch.cpu().detach().numpy(),
        input_dict["H_batch_array"],
        input_dict["W_batch_array"],
    ):
        f_single_mm = utils_coco.fpix_to_fmm(f_single, H_single, W_single)
        f_mm_list_yannick.append(f_single_mm)
    f_mm_list_est = []
    for f_single, H_single, W_single in zip(
        f_estim.cpu().detach().numpy(),
        input_dict["H_batch_array"],
        input_dict["W_batch_array"],
    ):
        f_single_mm = utils_coco.fpix_to_fmm(f_single, H_single, W_single)
        f_mm_list_est.append(f_single_mm)
    print_white_blue("+++++f_mm_yannick-----")
    f_mm_array_yannick = np.asarray(f_mm_list_yannick)
    print(f_mm_array_yannick)
    print_white_blue("+++++f_mm_est-----")
    f_mm_array_est = np.asarray(f_mm_list_est)
    print(f_mm_array_est)
    return f_mm_array_yannick, f_mm_array_est


def f_pixels_to_mm(f_estim, input_dict):
    f_mm_list_est = []
    for f_single, H_single, W_single in zip(
        f_estim.cpu().detach().numpy(),
        input_dict["H_batch_array"],
        input_dict["W_batch_array"],
    ):
        f_single_mm = utils_coco.fpix_to_fmm(f_single, H_single, W_single)
        f_mm_list_est.append(f_single_mm)
    f_mm_array_est = np.asarray(f_mm_list_est)
    return f_mm_array_est


def print_v0_info(v0_batch_est, v0_batch, output_pitch, H_batch):
    print_white_blue("+++++v0_batch_est-----")
    print(v0_batch_est.cpu().detach().numpy())
    print_white_blue("+++++v0_batch-----")
    print(v0_batch.cpu().detach().numpy())
    print_white_blue("+++++v0_batch_est_argmax-----")
    v0_argmax_image_single_list = []
    for idx, output_pitch_single in enumerate(output_pitch):
        v0_single_argmax_visible, v0_single_argmax = bin2midpointpitch(
            output_pitch_single.detach().squeeze().cpu().numpy()
        )  # [top 0 bottom 1]
        if v0_single_argmax_visible:
            v0_argmax_image_single = (
                H_batch[idx].cpu().numpy()
                - v0_single_argmax * H_batch[idx].cpu().numpy()
            )  # [top H bottom 0];
        else:
            v0_argmax_image_single = -1
        v0_argmax_image_single_list.append(v0_argmax_image_single)
    print(np.asarray(v0_argmax_image_single_list))


def print_v0_info_combine(v0_batch_est, v0_batch, output_pitch, H_batch):
    print_white_blue("+++++v0_batch_est-----")
    print(v0_batch_est.cpu().detach().numpy())
    print_white_blue("+++++v0_batch-----")
    print(v0_batch.cpu().detach().numpy())
    print_white_blue("+++++v0_batch_est_argmax-----")
    v0_argmax_image_single_list = []
    for idx, output_pitch_single in enumerate(output_pitch):
        v0_single_argmax_visible, v0_single_argmax = bin2midpointpitch(
            output_pitch_single.detach().squeeze().cpu().numpy()
        )  # [top 0 bottom 1]
        if v0_single_argmax_visible:
            v0_argmax_image_single = (
                H_batch[idx].cpu().numpy()
                - v0_single_argmax * H_batch[idx].cpu().numpy()
            )  # [top H bottom 0];
        else:
            v0_argmax_image_single = -1
        v0_argmax_image_single_list.append(v0_argmax_image_single)
    print(np.asarray(v0_argmax_image_single_list))


def copy_py_files(root_path, dest_path, exclude_paths=[]):
    from multiprocessing import Pool

    origin_path_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                origin_path = os.path.join(root, file)
                for exclude_path in exclude_paths:
                    if exclude_path != "" and exclude_path in origin_path:
                        break
                else:
                    origin_path_list.append([origin_path, dest_path])

    with Pool(processes=12, initializer=np.random.seed(123456)) as pool:
        for _ in list(
            tqdm(
                pool.imap_unordered(copy_file, origin_path_list),
                total=len(origin_path_list),
            )
        ):
            pass


def copy_file(origin_dest):
    os.system("cp %s %s/" % (origin_dest[0], origin_dest[1]))


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
