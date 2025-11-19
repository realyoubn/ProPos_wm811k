# -*- coding: UTF-8 -*-
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从配置文件中读取参数
def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())
    return configs

# 检查图像原始尺寸
def check_original_image_channels(data_folder):
    print("检查原始图像通道尺寸...")
    # 遍历数据文件夹，找到几个图像文件
    image_paths = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= 5:  # 只检查5个样本
                    break
        if len(image_paths) >= 5:
            break
    
    if not image_paths:
        print("未找到图像文件!")
        return
    
    for img_path in image_paths:
        img = Image.open(img_path)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            print(f"图像 {os.path.basename(img_path)}: 灰度图，形状 {img_array.shape}，通道数: 1")
        else:
            print(f"图像 {os.path.basename(img_path)}: RGB图，形状 {img_array.shape}，通道数: {img_array.shape[2]}")

# 检查经过PyTorch变换后的尺寸
def check_transformed_image_channels(data_folder, img_size):
    print("\n检查经过PyTorch变换后的图像通道尺寸...")
    
    # 创建与项目中相同的变换
    train_transform = [
        transforms.RandomResizedCrop(size=img_size, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose(train_transform)
    
    # 测试一些图像
    image_paths = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= 5:
                    break
        if len(image_paths) >= 5:
            break
    
    if not image_paths:
        print("未找到图像文件!")
        return
    
    for img_path in image_paths:
        img = Image.open(img_path)
        # 检查原始PIL图像模式
        print(f"\n图像 {os.path.basename(img_path)}:")
        print(f"  PIL模式: {img.mode}")
        
        # 应用变换
        transformed_img = train_transform(img)
        print(f"  变换后形状: {transformed_img.shape}")
        print(f"  变换后通道数: {transformed_img.shape[0]}")
        
        # 检查是否为灰度图
        if transformed_img.shape[0] == 1:
            print("  注意: 这是单通道图像，可能需要扩展到3通道")
        elif transformed_img.shape[0] == 3:
            # 检查是否是通过复制灰度通道得到的3通道图像
            if torch.all(transformed_img[0] == transformed_img[1]) and torch.all(transformed_img[1] == transformed_img[2]):
                print("  这是通过复制灰度通道得到的3通道图像")
            else:
                print("  这是RGB彩色图像")

# 测试灰度图像到3通道的转换
def test_gray_to_rgb_conversion(data_folder, img_size):
    print("\n测试灰度图像到3通道的转换...")
    
    # 创建基本变换
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # 创建灰度转3通道的变换
    class GrayscaleTo3Channels(torch.nn.Module):
        def forward(self, img):
            if img.shape[0] == 1:  # 如果是单通道灰度图
                img = img.repeat(3, 1, 1)  # 扩展为3通道
            return img
    
    # 完整变换
    full_transform = transforms.Compose([
        base_transform,
        GrayscaleTo3Channels()
    ])
    
    # 测试一些图像
    image_paths = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= 3:
                    break
        if len(image_paths) >= 3:
            break
    
    if not image_paths:
        print("未找到图像文件!")
        return
    
    for img_path in image_paths:
        img = Image.open(img_path)
        # 转换为灰度图进行测试
        if img.mode != 'L':
            img = img.convert('L')
        
        # 应用完整变换
        transformed_img = full_transform(img)
        print(f"\n灰度图像 {os.path.basename(img_path)} 转换后:")
        print(f"  形状: {transformed_img.shape}")
        print(f"  通道数: {transformed_img.shape[0]}")
        print(f"  验证三个通道是否相同: {torch.all(transformed_img[0] == transformed_img[1]) and torch.all(transformed_img[1] == transformed_img[2])}")

if __name__ == '__main__':
    # 加载配置文件
    config_path = os.path.join('config', 'wm811k_r18_propos.yml')
    configs = load_config(config_path)
    
    # 获取数据路径和图像尺寸
    data_folder = configs.get('data_folder', 'D:/2025/python/ProPos/datasets/Python/data/wm811k/labeled')
    img_size = configs.get('img_size', 32)
    
    print(f"使用配置:\n  数据路径: {data_folder}\n  图像尺寸: {img_size}x{img_size}")
    
    # 检查图像
    check_original_image_channels(data_folder)
    check_transformed_image_channels(data_folder, img_size)
    test_gray_to_rgb_conversion(data_folder, img_size)