import numpy as np
import torch
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_rui.data import samplers
from maskrcnn_rui.data.build import make_data_sampler
from utils.train_utils import cycle


class RandomSaturation:
    def __call__(self, sample):
        if np.random.rand() < 0.75:
            saturation_amt = np.random.triangular(-1, 0, 1)
            if np.random.rand() < 0.04:  # Grayscale
                saturation_amt = 1
            im = sample[0]
            im = torch.clamp(im + (torch.max(im, 0)[0] - im) * saturation_amt, 0, 1)
            sample[0] = im
        return sample


def make_batch_data_sampler(
    sampler,
    images_per_batch,
    num_iters=None,
    start_iter=0,
    drop_last=True,
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler,
        images_per_batch,
        drop_last=drop_last,
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler,
            num_iters,
            start_iter,
        )
    return batch_sampler


def make_data_loader(
    cfg,
    dataset,
    is_train=True,
    is_distributed=False,
    start_iter=0,
    logger=None,
    override_shuffle=None,
    collate_fn=None,
    batch_size_override=-1,
    is_sun360=False,
):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = (
            cfg.SOLVER.IMS_PER_BATCH
            if not is_sun360
            else cfg.SOLVER.IMS_PER_BATCH_SUN360
        )
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch,
            num_gpus,
        )
        images_per_gpu = (
            images_per_batch // num_gpus
            if batch_size_override == -1
            else batch_size_override
        )
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
        drop_last = True
    else:
        images_per_batch = (
            cfg.TEST.IMS_PER_BATCH if not is_sun360 else cfg.TEST.IMS_PER_BATCH_SUN360
        )
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch,
            num_gpus,
        )
        images_per_gpu = (
            images_per_batch // num_gpus
            if batch_size_override == -1
            else batch_size_override
        )
        # shuffle = False if not is_distributed else True
        shuffle = False
        num_iters = None
        start_iter = 0
        drop_last = False

    if override_shuffle is not None:
        shuffle = override_shuffle
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler,
        images_per_gpu,
        num_iters,
        start_iter,
        drop_last=drop_last,
    )
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )
    logger.info("DATALOADER WORKERS %d" % num_workers)
    logger.info(
        (
            "++++++[train_utils] len(dataset) %d, len(sampler) %d, len(batch_sampler) %d "
            + "len(data_loader) %d, is_train %s, is_distributed %s:"
        )
        % (
            len(dataset),
            len(sampler),
            len(batch_sampler),
            len(data_loader),
            is_train,
            is_distributed,
        ),
    )
    return data_loader


def iterator_coco_combine_alternate(iterator_A, iterator_B):
    flag = True
    # if len(iterator_A) > len(iterator_B):
    #     iterator_B = cycle(iterator_B)
    # else:
    #     iterator_A = cycle(iterator_A)
    iterator_A = cycle(iterator_A)
    iterator_B = cycle(iterator_B)
    iterator_A = iter(iterator_A)
    iterator_B = iter(iterator_B)

    # result = 0

    # while result is not None:
    while True:
        if flag:
            flag = not flag
            yield (next(iterator_A))
        else:
            flag = not flag
            yield (next(iterator_B))
