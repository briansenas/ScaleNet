import os

import torch
from eval_epoch_cvpr_RCNN import eval_epoch_cvpr_RCNN
from eval_epoch_v5_combine_RCNNOnly_pose_multiCat import eval_epoch_combine_RCNNOnly
from maskrcnn_benchmark.utils.comm import synchronize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils_misc import green

from .train_utils import clean_up_checkpoints


def check_eval_COCO(
    tid,
    epoch,
    rank,
    opt,
    model,
    eval_loader,
    writer,
    device,
    bins,
    logger,
    scheduler,
    epochs_evalued,
):
    is_better = False
    if not opt.not_val and epoch not in epochs_evalued:
        if rank == 0:
            logger.info(green("Evaluating on COCO..... epoch %d" % epoch))
        model.eval()
        with torch.no_grad():
            return_dict_epoch = eval_epoch_combine_RCNNOnly(
                model,
                eval_loader,
                device,
                opt,
                bins,
                tid,
                writer=writer,
                logger=logger,
                epoch=epoch,
            )
            synchronize()
            if isinstance(scheduler, ReduceLROnPlateau) and rank == 0:
                if "SUN360RCNN" not in opt.task_name:
                    step_metrics = return_dict_epoch["thres_ratio_dict"]["0.05"]
                    logger.info(
                        green(
                            (
                                "scheduler.step with thres_ratio_dict = %.2f at check_eval_coco epoch %d; "
                                + "lr: %.2E; num_bad_epochs: %d; patience: %d; best: %.2E"
                            )
                            % (
                                step_metrics,
                                epoch,
                                scheduler.optimizer.param_groups[0]["lr"],
                                scheduler.num_bad_epochs,
                                scheduler.patience,
                                scheduler.best,
                            ),
                        ),
                    )
                    scheduler.step(step_metrics)
                    is_better = scheduler.num_bad_epochs == 0
                    writer.add_scalar(
                        "training/scheduler-num_bad_epochs",
                        scheduler.num_bad_epochs,
                        tid,
                    )
                    writer.add_scalar("training/scheduler-best", scheduler.best, tid)
                    writer.add_scalar(
                        "training/scheduler-last_epoch",
                        scheduler.last_epoch,
                        tid,
                    )
                    writer.add_scalar("training/scheduler-epoch", epoch, tid)
        epochs_evalued.append(epoch)
        model.train()
    return is_better


def check_eval_SUN360(
    tid,
    epoch,
    rank,
    opt,
    model,
    eval_loader,
    writer,
    device,
    logger,
    scheduler,
    epochs_evaled,
):
    is_better = False
    # if epoch != 0 and epoch != epoch_start and not opt.not_val and epoch not in epochs_evaled:
    if not opt.not_val and epoch not in epochs_evaled:
        if rank == 0:
            logger.info(green("Evaluating on SUN360....."))
        model.eval()

        with torch.no_grad():
            return_dict_epoch = eval_epoch_cvpr_RCNN(
                model,
                eval_loader,
                epoch,
                tid,
                device,
                writer,
                logger,
                opt,
            )
            synchronize()
            if isinstance(scheduler, ReduceLROnPlateau) and rank == 0:
                print(
                    return_dict_epoch.keys(),
                    scheduler,
                    isinstance(scheduler, ReduceLROnPlateau),
                    "eval_loss_sum_SUN360" in return_dict_epoch,
                    scheduler.best,
                )
                if "SUN360RCNN" in opt.task_name:
                    scheduler.step(
                        return_dict_epoch["eval_loss_sum_SUN360"],
                    )
                    is_better = scheduler.num_bad_epochs == 0
                    logger.info(
                        green(
                            "scheduler.step with loss_mean = %.2f, best = %.2f, bad = %d, patience = %d"
                            % (
                                return_dict_epoch["eval_loss_sum_SUN360"],
                                scheduler.best,
                                scheduler.num_bad_epochs,
                                scheduler.patience,
                            ),
                        ),
                    )
                    writer.add_scalar(
                        "training/scheduler-num_bad_epochs",
                        scheduler.num_bad_epochs,
                        tid,
                    )
                    writer.add_scalar("training/scheduler-best", scheduler.best, tid)
                    writer.add_scalar(
                        "training/scheduler-last_epoch",
                        scheduler.last_epoch,
                        tid,
                    )
                    writer.add_scalar("training/scheduler-epoch", epoch, tid)
        epochs_evaled.append(epoch)
        model.train()

    return is_better


def check_vis_coco(
    tid,
    epoch,
    rank,
    opt,
    model,
    eval_loader,
    device,
    bins,
    logger,
    epochs_evaled,
    limit_sample=50,
    writer=None,
    prepostfix="",
    savepath_coco="",
    savepath_cocoPose="",
    if_eval_mode=True,
):
    if (
        not opt.not_val
        and (epoch < opt.save_every_epoch or epoch % opt.save_every_epoch == 0)
        and epoch not in epochs_evaled
    ):
        # if rank == 0 and not opt.not_val and epoch not in epochs_evaled:
        if rank == 0:
            logger.info(green("Visualizing COCO....." + prepostfix))
        if if_eval_mode:
            model.eval()
        with torch.no_grad():
            _ = eval_epoch_combine_RCNNOnly(
                model,
                eval_loader,
                device,
                opt,
                bins,
                tid,
                max_iter=limit_sample,
                writer=writer,
                logger=logger,
                savepath_coco=savepath_coco,
                savepath_cocoPose=savepath_cocoPose,
                if_debug=False,
                if_vis=True,
                prepostfix=prepostfix,
                if_loss=False,
            )
        model.train()
        epochs_evaled.append(epoch)


def check_vis_SUN360(
    tid,
    epoch,
    rank,
    opt,
    model,
    eval_loader,
    device,
    logger,
    epochs_evaled,
    limit_sample=50,
    writer=None,
    prepostfix="",
    savepath="",
    if_eval_mode=True,
):
    if (
        not opt.not_val
        and (epoch < opt.save_every_epoch or epoch % opt.save_every_epoch == 0)
        and epoch not in epochs_evaled
    ):
        # if rank == 0 and not opt.not_val and epoch not in epochs_evaled:
        if rank == 0:
            logger.info(green("Visualizing SUN360....." + prepostfix))
        if if_eval_mode:
            model.eval()
        with torch.no_grad():
            _ = eval_epoch_cvpr_RCNN(
                model,
                eval_loader,
                epoch,
                tid,
                device,
                writer,
                None,
                -1,
                logger,
                opt,
                max_iter=limit_sample,
                savepath=savepath,
                if_vis=True,
                prepostfix=prepostfix,
                if_loss=False,
            )
        model.train()
        epochs_evaled.append(epoch)
    # synchronize()


def check_save(
    rank,
    tid,
    epoch_save,
    epoch_total,
    opt,
    checkpointer,
    epochs_saved,
    checkpoints_folder,
    logger=None,
    is_better=False,
):
    arguments = {"iteration": tid, "epoch": epoch_total}
    if (
        rank == 0
        and (
            epoch_save < opt.save_every_epoch
            or epoch_save % opt.save_every_epoch == 0
            or (
                tid != 0
                and (opt.save_every_iter != 0 and tid % opt.save_every_iter == 0)
            )
        )
        and epoch_save not in epochs_saved
    ):

        saved_filename = checkpointer.save(
            "checkpointer_epoch%04d_iter%07d" % (epoch_total, tid),
            **arguments,
        )
        clean_up_checkpoints(
            os.path.join(checkpoints_folder, opt.task_name),
            leave_N=10,
            start_with="checkpointer_",
            logger=logger,
        )

        epochs_saved.append(epoch_save)

    if rank == 0 and is_better:
        ckpt_filepath = "best_checkpointer_epoch%04d_iter%07d" % (epoch_total, tid)
        saved_filename = checkpointer.save(ckpt_filepath, **arguments)
        logger.info(green("Saved BEST checkpoint to " + saved_filename))

    synchronize()
