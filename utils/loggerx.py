# -*- coding: UTF-8 -*-
'''@Project : ProPos 
@File    : loggerx.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 8:47 PM 
'''
import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect
from six import string_types as string_classes
import collections.abc as container_abcs
import warnings
from utils.ops import load_network


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def reduce_tensor(rt):
    rt = rt.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    rt /= world_size
    return rt


class LoggerX(object):

    # 首先在LoggerX类的__init__方法中添加一个控制日志输出的参数
    def __init__(self, save_root, enable_wandb=False, show_logs=True, **kwargs):
        # 移除断言，改为条件判断
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        self.metrics_save_dir = osp.join(save_root, 'metrics')  # 创建metrics目录
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        os.makedirs(self.metrics_save_dir, exist_ok=True)  # 确保metrics目录存在
        self._modules = []
        self._module_names = []
        self.metrics_history = {}  # 存储指标历史的字典
        # 检查是否启用了分布式训练
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.local_rank = dist.get_rank()
        else:
            # 非分布式模式下的默认值
            self.world_size = 1
            self.local_rank = 0
        self.enable_wandb = enable_wandb
        self.show_logs = False  # 添加这个参数控制是否显示日志
        if enable_wandb and self.local_rank == 0:
            import wandb
            wandb.init(dir=save_root, settings=wandb.Settings(_disable_stats=True, _disable_meta=True), **kwargs)
    
    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def append(self, module, name=None):
        self._modules.append(module)
        if name is None:
            name = get_varname(module)
        self._module_names.append(name)

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))
        # 保存当前的指标历史
        self.save_metrics()

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))
        # 加载指标历史
        self.load_metrics()

    # 然后修改msg方法，添加对show_logs参数的检查
    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        output_dict = {}
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)
            output_dict[var_name] = var
    
        # 保存指标到历史记录
        if self.local_rank == 0:
            for key, value in output_dict.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append((step, value))
    
        if self.enable_wandb and self.local_rank == 0:
            import wandb
            wandb.log(output_dict, step)
    
        # 只有当show_logs为True且是主进程时才打印日志
        if self.show_logs and self.local_rank == 0:
            print(output_str)

    def msg_str(self, output_str):
        if self.local_rank == 0:
            print(str(output_str))

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.jpg'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)
    
    def save_metrics(self):
        """
        保存所有指标历史到文件，包括损失函数
        """
        if self.local_rank != 0:
            return
        
        # 保存所有指标到一个npz文件
        np.savez(osp.join(self.metrics_save_dir, 'metrics_history.npz'), **{k: np.array(v) for k, v in self.metrics_history.items()})
        
        # 为每个重要的评价指标单独保存为txt文件，方便查看
        for metric_name in ['train_acc', 'val_acc', 'train_nmi', 'val_nmi', 'ema_train_acc', 'ema_train_nmi', 
                            'loss', 'train_loss', 'val_loss', 'clustering_loss']:  # 添加损失函数相关指标
            if metric_name in self.metrics_history:
                with open(osp.join(self.metrics_save_dir, f'{metric_name}.txt'), 'w') as f:
                    f.write('# step\tvalue\n')
                    for step, value in self.metrics_history[metric_name]:
                        f.write(f'{step}\t{value}\n')
    
    def load_metrics(self):
        """
        加载指标历史
        """
        metrics_file = osp.join(self.metrics_save_dir, 'metrics_history.npz')
        if os.path.exists(metrics_file):
            data = np.load(metrics_file)
            self.metrics_history = {k: v.tolist() for k, v in data.items()}