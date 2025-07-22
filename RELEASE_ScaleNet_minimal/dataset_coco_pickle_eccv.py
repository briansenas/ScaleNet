import os
import pickle
import random
import time
from glob import glob

import numpy as np
import torch
import torchvision
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from PIL import Image
from scipy.io import loadmat
from termcolor import colored

# NOTE: What is this directory
results_path_yannick = "data/COCO/coco_results/yannick_results_train2017_filtered"

pickle_paths = {
    "train-val": "data/COCO/coco_results/results_with_kps_20200208_morethan2_2-8/pickle",
    "test": "data/COCO/coco_results/results_with_kps_20200225_val2017_test_detOnly_filtered_2-8_moreThan2/pickle",
}

pickle_paths["test"] = (
    "/results_test_20200302_Car_noSmall-ratio1-35-mergeWith-results_with_kps_20200225_train2017_detOnly_filtered_2-8_moreThan2/pickle"  # MultiCat
)


bbox_paths = {key: pickle_paths[key].replace("/pickle", "/npy") for key in pickle_paths}


class COCO2017ECCV(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self,
        transforms_maskrcnn=None,
        transforms_yannick=None,
        split="",
        shuffle=True,
        logger=None,
        opt=None,
        write_split=False,
    ):

        assert split in ["train", "val", "test"], (
            "COCO2017Scale: Wrong dataset split: %s!" % split
        )
        if split in ["train", "val"]:
            ann_file = (
                "data/COCO/annotations/person_keypoints_train2017.json"  # !!!! tmp!
            )
            root = "data/COCO/train2017"
            pickle_path = pickle_paths["train-val"]
            bbox_path = bbox_paths["train-val"]
            image_path = "data/COCO/train2017"
            # self.train = True
        else:
            ann_file = (
                "data/COCO/annotations/person_keypoints_val2017.json"  # !!!! tmp!
            )
            root = "data/COCO/val2017"
            pickle_path = pickle_paths["test"]
            bbox_path = bbox_paths["test"]
            image_path = "data/COCO/val2017"
        super().__init__(root, ann_file)

        self.opt = opt
        self.cfg = self.opt.cfg
        self.GOOD_NUM = self.cfg.DATA.COCO.GOOD_NUM
        if split in ["train", "val"]:
            self.train = True
        else:
            self.train = False

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }

        self.transforms_maskrcnn = transforms_maskrcnn
        self.transforms_yannick = transforms_yannick
        ts = time.time()
        self.yannick_mat_files = glob(
            os.path.join(results_path_yannick, "*.mat"),
            recursive=True,
        )
        self.yannick_mat_files.sort()
        random.seed(123456)
        random.shuffle(self.yannick_mat_files)

        num_mat_files = len(self.yannick_mat_files)
        if split == "train":
            self.yannick_mat_files = self.yannick_mat_files[: int(num_mat_files * 0.8)]
        elif split == "val":
            self.yannick_mat_files = self.yannick_mat_files[-int(num_mat_files * 0.2) :]
        # I don't have the mat files, seems odd. It's a missing piece for the training dataset
        logger.info(self.yannick_mat_files[0])
        if self.train:
            self.img_filenames = [
                os.path.basename(yannick_mat_file).split(".")[0]
                for yannick_mat_file in self.yannick_mat_files
            ]
            self.img_files = [
                os.path.join(image_path, img_filename + ".jpg")
                for img_filename in self.img_filenames
            ]
            self.bbox_npy_files = [
                os.path.join(bbox_path, img_filename + ".npy")
                for img_filename in self.img_filenames
            ]
            self.pickle_files = [
                os.path.join(pickle_path, img_filename + ".data")
                for img_filename in self.img_filenames
            ]
            assert (
                len(self.bbox_npy_files)
                == len(self.pickle_files)
                == len(self.pickle_files)
                == len(self.img_files)
                == len(self.yannick_mat_files)
            )

            bbox_npy_files_filtered = []
            pickle_files_filtered = []
            img_files_filtered = []
            yannick_mat_files_filtered = []
            for bbox_npy_file, pickle_file, img_file, yannick_mat_file in zip(
                self.bbox_npy_files,
                self.pickle_files,
                self.img_files,
                self.yannick_mat_files,
            ):
                if os.path.isfile(pickle_file):
                    assert (
                        os.path.basename(bbox_npy_file)[:12]
                        == os.path.basename(pickle_file)[:12]
                        == os.path.basename(img_file)[:12]
                        == os.path.basename(yannick_mat_file)[:12]
                    )
                    bbox_npy_files_filtered.append(bbox_npy_file)
                    pickle_files_filtered.append(pickle_file)
                    img_files_filtered.append(img_file)
                    yannick_mat_files_filtered.append(yannick_mat_file)
            self.bbox_npy_files = bbox_npy_files_filtered
            self.pickle_files = pickle_files_filtered
            self.img_files = img_files_filtered
            self.yannick_mat_files = yannick_mat_files_filtered
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
                os.path.join(image_path, img_filename + ".jpg")
                for img_filename in self.img_filenames
            ]
            self.bbox_npy_files = [
                os.path.join(bbox_path, img_filename + ".npy")
                for img_filename in self.img_filenames
            ]
            self.yannick_mat_files = [""] * len(self.pickle_files)

        assert (
            len(self.bbox_npy_files)
            == len(self.pickle_files)
            == len(self.img_files)
            == len(self.yannick_mat_files)
        )

        if write_split and opt.rank == 0:
            list_file = pickle_path.replace("/pickle", "") + "/%s.txt" % split
            train_file = open(list_file, "a")

            for train_pickle in self.pickle_files:
                train_id_06 = os.path.basename(train_pickle)[6:12]
                train_file.write("%s\n" % (train_id_06))
            train_file.close()

        if opt.rank == 0:
            logger.info(
                colored(
                    "[COCO dataset] Loaded %d PICKLED files in %.4fs for %s set from %s."
                    % (len(self.pickle_files), time.time() - ts, split, pickle_path),
                    "white",
                    "on_blue",
                ),
            )

        if shuffle:
            random.seed(314159265)
            list_zip = list(
                zip(
                    self.img_files,
                    self.bbox_npy_files,
                    self.pickle_files,
                    self.yannick_mat_files,
                ),
            )
            random.shuffle(list_zip)
            (
                self.img_files,
                self.bbox_npy_files,
                self.pickle_files,
                self.yannick_mat_files,
            ) = zip(*list_zip)
            assert (
                os.path.basename(self.img_files[0])[:12]
                == os.path.basename(self.bbox_npy_files[0])[:12]
                == os.path.basename(self.pickle_files[0])[:12]
                == os.path.basename(self.yannick_mat_files[0])[:12]
            )

    def __getitem__(self, k):
        im_ori_RGB = Image.open(self.img_files[k]).convert(
            "RGB",
        )  # im_ori_RGB.size: (W, H)
        with open(self.pickle_files[k], "rb") as filehandle:
            data = pickle.load(filehandle)
        bboxes = data["bboxes"].astype(np.float32)  # [xywh]
        assert len(bboxes.shape) == 2 and bboxes.shape[1] == 4
        num_bboxes_ori = bboxes.shape[0]

        if "label" in data:
            labels = data["label"]  # ['car', 'person', 'person']
        else:
            labels = ["person"] * num_bboxes_ori
        if bboxes.shape[0] > self.cfg.DATA.COCO.GOOD_NUM:
            bboxes = bboxes[: self.cfg.DATA.COCO.GOOD_NUM, :]
            labels = labels[: self.cfg.DATA.COCO.GOOD_NUM]

        target_boxes = torch.as_tensor(bboxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(target_boxes, im_ori_RGB.size, mode="xywh").convert("xyxy")
        num_boxes = target.bbox.shape[0]

        if self.opt.est_kps:
            if "kps" in data:
                kps_gt = data["kps"].astype(int)  # [N, 51]
                if num_bboxes_ori > self.cfg.DATA.COCO.GOOD_NUM:
                    kps_gt = kps_gt[: self.cfg.DATA.COCO.GOOD_NUM, :]
                kps_gt = kps_gt.tolist()  # [[51]]
            else:
                kps_gt = [[0] * 51 for _ in range(num_boxes)]

            target_keypoints = PersonKeypoints(kps_gt, im_ori_RGB.size)
            target.add_field("keypoints", target_keypoints)
            target = target.clip_to_image(remove_empty=True)
            classes = [1] * num_boxes  # !!!!! all person (1) for now...
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)
            scores = torch.tensor([1.0] * target.bbox.shape[0])
            target.add_field("scores", scores)

        W, H = im_ori_RGB.size[:2]
        if self.train:
            yannick_results = loadmat(self.yannick_mat_files[k])
            horizon = yannick_results["pitch"][0][0].astype(np.float32)
            horizon_pixels_yannick = H * horizon
            v0 = H - horizon_pixels_yannick
            vfov = yannick_results["vfov"][0][0].astype(np.float32)
            f_pixels_yannick = H / 2.0 / (np.tan(vfov / 2.0))
        else:
            f_pixels_yannick = -1
            v0 = -1

        im_yannickTransform = self.transforms_yannick(im_ori_RGB)  # [0., 1.] by default
        im_maskrcnnTransform, target_maskrcnnTransform = self.transforms_maskrcnn(
            im_ori_RGB,
            target,
        )  # [0., 1.] by default
        if self.train and self.opt.est_kps:
            target_maskrcnnTransform.add_field("keypoints_ori", target_keypoints)
            target_maskrcnnTransform.add_field("boxlist_ori", target)
        target_maskrcnnTransform.add_field("img_files", [self.img_files[k]] * num_boxes)

        if self.train:
            y_person = 1.75
            bbox_good_list = bboxes
            vc = H / 2.0
            yc_list = []
            for bbox in bbox_good_list:
                vt = H - bbox[1]
                vb = H - (bbox[1] + bbox[3])
                yc_single = (
                    y_person
                    * (v0 - vb)
                    / (vt - vb)
                    / (1.0 + (vc - v0) * (vc - vt) / f_pixels_yannick**2)
                )
                yc_list.append(yc_single)
            yc_estCam = np.median(np.asarray(yc_list))
        else:
            yc_estCam = -1

        assert len(labels) == bboxes.shape[0]
        return (
            im_yannickTransform,
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
        )

    def __len__(self):
        return len(self.img_files)

    def pad_bbox(self, bboxes, max_length):
        bboxes_padded = np.zeros((max_length, bboxes.shape[1]))
        assert bboxes.shape[0] <= max_length, "bboxes length %d > max_length %d!" % (
            bboxes.shape[0],
            max_length,
        )
        bboxes_padded[: bboxes.shape[0], :] = bboxes
        return bboxes_padded


def my_collate(batch):
    # Refer to https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    (
        im_yannickTransform_list,
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
    ) = zip(*batch)
    W_batch_array = np.stack(W_batch_list).copy()
    H_batch_array = np.stack(H_batch_list).copy()
    yc_batch = torch.tensor(yc_batch_list)
    bboxes_batch_array = np.stack(bboxes_batch_list).copy()
    bboxes_length_batch_array = np.stack(bboxes_length_batch_list).copy()
    v0_batch = torch.tensor(v0_batch_list)
    f_pixels_yannick_batch = torch.tensor(f_pixels_yannick_batch_list)
    return (
        im_yannickTransform_list,
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
    train = COCO2017ECCV(train=True)
    print(len(train))
    for a in range(len(train)):
        _ = train[a]
