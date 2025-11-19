# -*- coding: utf-8 -*-

import os
import glob
import time
import argparse
from typing import Union  # 用于正确的类型注解

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from sklearn.model_selection import train_test_split


class WM811kProcessor(object):
    def __init__(self, wm811k_file: str):

        start_time = time.time()
        self.data = pd.read_pickle(wm811k_file)
        print(f'Successively loaded WM811k data. {time.time() - start_time:.2f}s')

        # 修正原始数据列名拼写错误（trianTestLabel → trainTestLabel）
        self.data['labelString'] = self.data['failureType'].apply(self.getLabelString)
        self.data['trainTestLabel'] = self.data['trianTestLabel'].apply(self.getTrainTestLabel)  # 修正拼写

        self.data['waferMapDim'] = self.data['waferMap'].apply(lambda x: x.shape)
        self.data['waferMapSize'] = self.data['waferMapDim'].apply(lambda x: x[0] * x[1])
        self.data['lotName'] = self.data['lotName'].apply(lambda x: x.replace('lot', ''))
        self.data['waferIndex'] = self.data['waferIndex'].astype(int)

    @staticmethod
    def save_image(arr: np.ndarray, filepath: str = 'image.png', vmin: int = 0, vmax: int = 2):
        scaled_arr = (arr / vmax) * 255
        img = Image.fromarray(scaled_arr.astype(np.uint8))
        img.save(filepath, dpi=(500, 500))

    @staticmethod
    def load_image(filepath: str = 'image.png'):
        return Image.open(filepath)

    def write_images(self, root: str, indices: Union[list, tuple]):
        """Write wafer images to .png files."""
        os.makedirs(root, exist_ok=True)
        with tqdm.tqdm(total=len(indices), leave=True, colour='yellow') as pbar:
            for i, row in self.data.loc[indices].iterrows():
                pngfile = os.path.join(root, row['labelString'], f'{i:06}.png')
                os.makedirs(os.path.dirname(pngfile), exist_ok=True)
                self.save_image(row['waferMap'], pngfile)
                pbar.set_description_str(f" {root} - {i:06} ")
                pbar.update(1)
        print(f"✅ 所有图像已成功保存到 {root} 文件夹")

    def write_unlabeled_images(self,
                               root: str = './data/wm811k/unlabeled/',
                               train_size: float = 0.8,
                               valid_size: float = 0.1):
        """Write wafer images without labels."""
        test_size = 1 - train_size - valid_size

        unlabeled_indices = self.data.loc[self.data['trainTestLabel'] == -1].index
        train_indices, temp_indices = train_test_split(
            unlabeled_indices,
            train_size=train_size,
            shuffle=True,
            random_state=2015010720,
        )
        valid_indices, test_indices = train_test_split(
            temp_indices,
            train_size=valid_size / (valid_size + test_size),
            shuffle=True,
            random_state=2015010720,
        )

        self.write_images(os.path.join(root, 'train'), train_indices)
        self.write_images(os.path.join(root, 'valid'), valid_indices)
        self.write_images(os.path.join(root, 'test'), test_indices)

    def write_labeled_images(self,
                             root: str = './data/wm811k/labeled/',
                             train_size: float = 0.8,
                             valid_size: float = 0.1,
                             none_sample_ratio: float = 0.3):  # 新增：none标签采样比例
        """Write wafer images with labels. 对none标签样本只保留30%"""
        test_size = 1 - train_size - valid_size

        # 1. 先获取原始有标签数据（trainTestLabel != -1）
        labeled_indices = self.data.loc[self.data['trainTestLabel'] != -1].index
        # 获取这些样本的标签
        labeled_labels = self.data.loc[labeled_indices, 'labelString']

        # 2. 分离出'none'标签和非'none'标签的样本
        none_mask = (labeled_labels == 'none')  # 'none'标签的掩码
        none_indices = labeled_indices[none_mask]  # 所有'none'标签样本的索引
        non_none_indices = labeled_indices[~none_mask]  # 非'none'标签样本的索引

        print(f"原始有标签数据中：'none'标签样本数={len(none_indices)}，非'none'标签样本数={len(non_none_indices)}")

        # 3. 对'none'标签样本随机采样30%
        sampled_none_indices, _ = train_test_split(
            none_indices,
            train_size=none_sample_ratio,  # 只保留30%
            shuffle=True,
            random_state=2015010720  # 固定随机种子，保证可复现
        )
        print(f"对'none'标签采样后保留：{len(sampled_none_indices)}个样本（占原'none'样本的{none_sample_ratio*100}%）")

        # 4. 合并采样后的'none'和非'none'样本，作为新的有标签数据
        new_labeled_indices = np.concatenate([sampled_none_indices, non_none_indices])

        # 5. 基于新的有标签数据划分train/valid/test（保持分层抽样，保证类别比例）
        temp_indices, test_indices = train_test_split(
            new_labeled_indices,
            test_size=test_size,
            stratify=self.data.loc[new_labeled_indices, 'labelString'],  # 按新标签分层
            shuffle=True,
            random_state=2015010720,
        )
        train_indices, valid_indices = train_test_split(
            temp_indices,
            test_size=valid_size/(train_size + valid_size),
            stratify=self.data.loc[temp_indices, 'labelString'],  # 按临时标签分层
            random_state=2015010720,
        )

        # 6. 写入文件
        self.write_images(os.path.join(root, 'train'), train_indices)
        self.write_images(os.path.join(root, 'valid'), valid_indices)
        self.write_images(os.path.join(root, 'test'), test_indices)

    @staticmethod
    def nearest_interpolate(arr, s=(40, 40)):
        assert isinstance(arr, np.ndarray) and len(arr.shape) == 2
        ptt = torch.from_numpy(arr).view(1, 1, *arr.shape).float()
        return F.interpolate(ptt, size=s, mode='nearest').squeeze().long().numpy()

    @staticmethod
    def getLabelString(x):
        if len(x) == 1:
            ls = x[0][0].strip().lower()  # 标签为'none'时会被处理为'none'
        else:
            ls = '-'
        return ls

    @staticmethod
    def getTrainTestLabel(x):
        d = {
            'unlabeled': -1,  # 638,507
            'training': 0,    # 118,595
            'test': 1,        #  54,355
        }
        if len(x) == 1:
            lb = x[0][0].strip().lower()
        else:
            lb = 'unlabeled'
        return d[lb]


if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser("Process WM-811k data to individual image files.", add_help=True)
        parser.add_argument('--labeled_root', type=str, default='./data/wm811k/labeled')
        parser.add_argument('--unlabeled_root', type=str, default='./data/wm811k/unlabeled')
        parser.add_argument('--labeled_train_size', type=float, default=0.8)
        parser.add_argument('--labeled_valid_size', type=float, default=0.1)
        parser.add_argument('--unlabeled_train_size', type=float, default=0.8)
        parser.add_argument('--unlabeled_valid_size', type=float, default=0.1)
        parser.add_argument('--none_sample_ratio', type=float, default=0.07, 
                           help="对'none'标签样本的采样比例（默认30%）")  # 新增参数

        return parser.parse_args()

    def check_files_exist_in_directory(directory: str, file_ext: str = 'png', recursive: bool = True):
        """Check existence of files of specific types are under a directory"""
        files = glob.glob(os.path.join(directory, f"**/*.{file_ext}"), recursive=recursive)
        return len(files) > 0  # True if files exist, else False.

    args = parse_args()
    # 注意：确保pkl文件路径与实际存放位置一致
    processor = WM811kProcessor(wm811k_file='./data/wm811k/LSWMD.pkl')

    if not check_files_exist_in_directory(args.labeled_root):
        processor.write_labeled_images(
            root=args.labeled_root,
            train_size=args.labeled_train_size,
            valid_size=args.labeled_valid_size,
            none_sample_ratio=args.none_sample_ratio  # 传入采样比例
        )
    else:
        print(f"Labeled images exist in `{args.labeled_root}`. Skipping...")

    if not check_files_exist_in_directory(args.unlabeled_root):
        processor.write_unlabeled_images(
            root=args.unlabeled_root,
            train_size=args.unlabeled_train_size,
            valid_size=args.unlabeled_valid_size
        )
    else:
        print(f"Unlabeled images exist in `{args.unlabeled_root}`. Skipping...")