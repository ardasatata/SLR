import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

import os.path

sys.path.append("..")

CROP_X = 200
CROP_TOP = 200

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb"):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/csl/{mode}_info.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        print("Data Loader Init")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info'], None
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info'], None
        elif self.data_type == "video-keypoint":
            input_data, label, fi = self.read_video(idx)
            keypoint_data = self.load_keypoint(idx)
            input_data, keypoint_data, label = self.normalize(input_data, keypoint_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info'], keypoint_data
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info'], None

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]

        # print(fi['video_path'])
        # print(os.path.exists(fi['video_path']))
        # exit()
        try:
            cap = cv2.VideoCapture(fi['video_path'])
            video = []

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cropped = frame[0 + CROP_TOP:720, 0 + CROP_X:1280 - CROP_X]
                    resized_image = cv2.resize(cropped, (256, 256))

                    # append frame to be converted
                    video.append(np.asarray(resized_image))
                else:
                    break
            cap.release()
        except cv2.error as e:
            print(e)
            False

        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return video, label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def load_keypoint(self, index):
        # Keypoint config
        index_mirror = np.concatenate([
            [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
            [21, 22, 23, 18, 19, 20],
            np.arange(40, 23, -1), np.arange(50, 40, -1),
            np.arange(51, 55), np.arange(59, 54, -1),
            [69, 68, 67, 66, 71, 70], [63, 62, 61, 60, 65, 64],
            np.arange(78, 71, -1), np.arange(83, 78, -1),
            [88, 87, 86, 85, 84, 91, 90, 89],
            np.arange(113, 134), np.arange(92, 113)
        ]) - 1
        assert (index_mirror.shape[0] == 133)

        selected = np.concatenate(([0, 5, 6, 7, 8, 9, 10],
                                   [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                                   [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),  # 27

        # load file info
        fi = self.inputs_list[index]
        # data = np.load(f"{self.prefix}/features/keypoint-hrnet/{self.mode}/{fi['fileid']}.npy", allow_pickle=True)
        data = np.load(fi['keypoint_path'], allow_pickle=True)

        return torch.from_numpy(data[:, selected, :])

    # def normalize(self, video, label, file_id=None):
    #     video, label = self.data_aug(video, label, file_id)
    #     video = video.float() / 127.5 - 1
    #     return video, label

    def normalize(self, video, keypoint, label, file_id=None):
        video, keypoint, label = self.data_aug(video, keypoint, label, file_id)

        if isinstance(keypoint, torch.Tensor):
            keypoint = keypoint.cpu().detach().numpy()

        video = video.float() / 127.5 - 1
        keypoint = 2.*(keypoint - np.min(keypoint)) / np.ptp(keypoint)-1
        return video, torch.from_numpy(keypoint), label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info, keypoint = list(zip(*batch))

        # print(padded_keypoint)
        # print("hello")

        padded_keypoint = []

        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)

            max_len = len(keypoint[0])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_keypoint = [torch.cat(
                (
                    key[0][None].expand(left_pad, -1, -1, -1),
                    key,
                    key[-1][None].expand(max_len - len(key) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for key in keypoint]
            padded_keypoint = torch.stack(padded_keypoint)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])

        if max(label_length) == 0:
            return padded_video, video_length, [], [], info, padded_keypoint
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info, padded_keypoint

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
