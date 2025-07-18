import os
import pickle
import random
import time
from glob import glob

import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from PIL import Image
from termcolor import colored


# class COCO2017Scale(torchvision.datasets.coco.CocoDetection):
class COCO2017Scale:
    def __init__(
        self,
        transforms_maskrcnn=None,
        split="",
        coco_subset="coco_scale",
        shuffle=True,
        logger=None,
        opt=None,
        dataset_name="",
    ):

        assert split in ["train", "val", "test"], (
            "COCO2017Scale: Wrong dataset split: %s!" % split
        )
        assert coco_subset in ["coco_full_bboxkps", "coco_scale", "coco_scale_eccv"], (
            "COCO2017Scale: Wrong coco subset: %s!" % coco_subset
        )

        COCO_SCALE_DATA_PATH = "data/COCO/coco_results"

        data_name_subset_dict = {
            "coco_full_bboxkps": {
                "train-val": "results_with_kps_20200407_GTnoFiltering",  # 64,115(61,933+2,182)
                "test": "",
            },
            "coco_scale": {
                "train-val": "results_with_kps_20200403_noNull_filtered_ratio2-8_moreThan2_total64115",
                "test": "results_with_kps_20200225_val2017_test_detOnly_filtered_2-8_moreThan2",
            },
            "coco_scale_eccv": {
                "train-val": "results_with_kps_20200208_morethan2_2-8",  # 13,875(10,547+2,648)
                "test": "results_test_20200302_Car_noSmall-ratio1-35-mergeWith-results_with_kps_20200225_train2017_detOnly_filtered_2-8_moreThan2",
            },  # MultiCat
        }
        self.coco_subset = coco_subset
        data_name = data_name_subset_dict[self.coco_subset]

        pickle_paths = {}
        list_paths = {}
        for data_split in ["train-val", "test"]:
            pickle_paths.update(
                {
                    data_split: os.path.join(
                        COCO_SCALE_DATA_PATH,
                        data_name[data_split],
                        "pickle",
                    ),
                },
            )
            list_paths.update(
                {data_split: os.path.join(COCO_SCALE_DATA_PATH, data_name[data_split])},
            )

        if split in ["train", "val"]:
            pickle_path = pickle_paths["train-val"]
            image_id_list_file = os.path.join(list_paths["train-val"], "%s.txt" % split)
            image_path = "data/COCO/train2017"
        else:
            pickle_path = pickle_paths["test"]
            image_path = "data/COCO/val2017"

        self.opt = opt
        self.cfg = self.opt.cfg
        self.GOOD_NUM = self.cfg.DATA.COCO.GOOD_NUM
        if split in ["train", "val"]:
            self.train_val = True
        else:
            self.train_val = False

        self.json_category_id_to_contiguous_id = {1: 1}

        self.transforms_maskrcnn = transforms_maskrcnn
        ts = time.time()

        if self.train_val:
            with open(image_id_list_file) as fp:
                self.img_filenames = [
                    img_filename06.strip() for img_filename06 in fp.readlines()
                ]
            self.pickle_filenames = self.img_filenames
            if (
                data_name["train-val"] == "results_with_kps_20200208_morethan2_2-8"
            ):  # backward compatibility for old data
                self.pickle_filenames = [
                    "000000" + pickle_filename
                    for pickle_filename in self.pickle_filenames
                ]
            self.pickle_files = [
                os.path.join(pickle_path, pickle_filename + ".data")
                for pickle_filename in self.pickle_filenames
            ]
        else:
            self.pickle_files = glob(
                os.path.join(pickle_path, "*.data"),
                recursive=True,
            )
            self.pickle_files.sort()
            self.img_filenames = [
                os.path.basename(pickle_file).split(".")[0]
                for pickle_file in self.pickle_files
            ]
        self.img_files = [
            os.path.join(image_path, "000000" + img_filename + ".jpg")
            for img_filename in self.img_filenames
        ]
        assert len(self.pickle_files) == len(self.img_files)
        if opt.rank == 0:
            logger.info(
                colored(
                    "[COCO-Scale | %s | %s] Loaded %d PICKLED files in %.4fs for %s set from %s"
                    % (
                        dataset_name,
                        coco_subset,
                        len(self.pickle_files),
                        time.time() - ts,
                        split,
                        pickle_path,
                    ),
                    "white",
                    "on_blue",
                ),
            )

        self.img_idxes = range(len(self.img_files))
        if shuffle:
            random.seed(314159265)
            list_zip = list(zip(self.img_files, self.pickle_files, self.img_idxes))
            random.shuffle(list_zip)
            self.img_files, self.pickle_files, self.img_idxes = zip(*list_zip)
            assert int(os.path.basename(self.img_files[0]).split(".")[0][-6:]) == int(
                os.path.basename(self.pickle_files[0]).split(".")[0][-6:],
            )

        if not self.train_val:
            print([os.path.basename(img_file) for img_file in self.img_files[:12]])

    def __getitem__(self, k):
        im_ori_RGB = Image.open(self.img_files[k]).convert(
            "RGB",
        )  # im_ori_RGB.size: (W, H
        with open(self.pickle_files[k], "rb") as filehandle:
            data = pickle.load(filehandle)
        bboxes = data["bboxes"].astype(np.float32)  # [xywh]
        assert len(bboxes.shape) == 2 and bboxes.shape[1] == 4
        num_bboxes_ori = bboxes.shape[0]

        if "label" in data:
            labels = data["label"]  # ['car', 'person', 'person']
        else:
            labels = ["person"] * num_bboxes_ori
        if (
            bboxes.shape[0] > self.cfg.DATA.COCO.GOOD_NUM
            and self.cfg.DATA.COCO.CLIP_N_IN_MASKRCNN
        ):
            bboxes = bboxes[: self.cfg.DATA.COCO.GOOD_NUM, :]
            labels = labels[: self.cfg.DATA.COCO.GOOD_NUM]

        target_boxes = torch.as_tensor(bboxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(target_boxes, im_ori_RGB.size, mode="xywh").convert("xyxy")
        num_boxes = target.bbox.shape[0]

        if self.opt.est_kps:
            if "kps" in data:
                kps_gt = data["kps"].astype(int)  # [N, 51]
                if (
                    num_bboxes_ori > self.cfg.DATA.COCO.GOOD_NUM
                    and self.cfg.DATA.COCO.CLIP_N_IN_MASKRCNN
                ):
                    kps_gt = kps_gt[: self.cfg.DATA.COCO.GOOD_NUM, :]
                kps_gt = kps_gt.tolist()  # [[51]]
            else:
                kps_gt = [[0] * 51 for i in range(num_boxes)]

            target_keypoints = PersonKeypoints(kps_gt, im_ori_RGB.size)
            target.add_field("keypoints", target_keypoints)

        if self.opt.est_bbox:
            classes = [1] * num_boxes  # !!!!! all person (1) for now...
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)
            scores = torch.tensor([1.0] * target.bbox.shape[0])
            target.add_field("scores", scores)
            target = target.clip_to_image(remove_empty=True)

        im_maskrcnnTransform, target_maskrcnnTransform = self.transforms_maskrcnn(
            im_ori_RGB,
            target,
        )  # [0., 1.] by default

        # Obsolete GT params from offline Yannick model
        W, H = im_ori_RGB.size[:2]
        f_pixels_yannick = -1
        v0 = -1
        yc_estCam = -1

        if self.train_val:
            if self.opt.est_kps:
                target_maskrcnnTransform.add_field("keypoints_ori", target_keypoints)
            if self.opt.est_bbox:
                target_maskrcnnTransform.add_field("boxlist_ori", target)
        target_maskrcnnTransform.add_field("img_files", [self.img_files[k]] * num_boxes)

        assert len(labels) == bboxes.shape[0]

        mis = [self.coco_subset, self.img_idxes[k]]
        return (
            im_maskrcnnTransform,
            W,
            H,
            float(yc_estCam),
            self.pad_bbox(bboxes, self.GOOD_NUM).astype(np.float32),
            bboxes.shape[0],
            float(v0),
            float(f_pixels_yannick),
            os.path.basename(self.img_files[k])[:12],
            self.img_files[k],
            target_maskrcnnTransform,
            labels,
            mis,
        )

    def __len__(self):
        return len(self.img_files)

    def pad_bbox(self, bboxes, max_length):
        bboxes_padded = np.zeros((max_length, bboxes.shape[1]))
        if bboxes.shape[0] > max_length:
            bboxes = bboxes[: self.cfg.DATA.COCO.GOOD_NUM, :]
        assert bboxes.shape[0] <= max_length, "bboxes length %d > max_length %d!" % (
            bboxes.shape[0],
            max_length,
        )
        bboxes_padded[: bboxes.shape[0], :] = bboxes
        return bboxes_padded


def my_collate(batch):
    # Refer to https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    (
        im_maskrcnnTransform_list,
        W_batch_list,
        H_batch_list,
        yc_batch_list,
        bboxes_batch_list,
        bboxes_length_batch_list,
        v0_batch_list,
        f_pixels_yannick_batch_list,
        im_filename_list,
        im_file_list,
        target_maskrcnnTransform_list,
        labels_list,
        mis_list,
    ) = zip(*batch)
    W_batch_array = np.stack(W_batch_list).copy()
    H_batch_array = np.stack(H_batch_list).copy()
    yc_batch = torch.tensor(yc_batch_list)
    bboxes_batch_array = np.stack(bboxes_batch_list).copy()
    bboxes_length_batch_array = np.stack(bboxes_length_batch_list).copy()
    v0_batch = torch.tensor(v0_batch_list)
    f_pixels_yannick_batch = torch.tensor(f_pixels_yannick_batch_list)
    return (
        im_maskrcnnTransform_list,
        W_batch_array,
        H_batch_array,
        yc_batch,
        bboxes_batch_array,
        bboxes_length_batch_array,
        v0_batch,
        f_pixels_yannick_batch,
        im_filename_list,
        im_file_list,
        target_maskrcnnTransform_list,
        labels_list,
        mis_list,
    )


def collate_fn_padd(batch):
    """
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """

    ims = [torch.Tensor(item[0]) for item in batch]
    bboxes = [torch.Tensor(item[1]) for item in batch]
    v0s = [torch.Tensor(np.asarray(item[2])) for item in batch]
    f_pixels_yannicks = [torch.Tensor(np.asarray(item[3])) for item in batch]
    img_filenames = [item[4] for item in batch]
    img_filepaths = [item[5] for item in batch]

    return [ims, bboxes, v0s, f_pixels_yannicks, img_filenames, img_filepaths]


if __name__ == "__main__":
    train = COCO2017Scale(train=True)
    print(len(train))
    for a in range(len(train)):
        _ = train[a]
