import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np  # 确保图像数据为数组格式

# 1. 读取数据
df = pd.read_pickle("D:/2025/python/ProPos/MIR-WM811K/MIR-WM811K/Python/WM811K.pkl")

# 2. 预处理：过滤无效数据（确保故障类型和图像非空）
df = df.dropna(subset=['failureType', 'waferMap'])

# 3. 获取所有唯一故障类型，创建对应文件夹
failure_types = df['failureType'].unique()
root_dir = "pure_wafer_images_by_failureType"  # 根目录（纯图像）
os.makedirs(root_dir, exist_ok=True)

# 创建子文件夹（处理特殊字符）
for ft in failure_types:
    clean_ft = str(ft).replace('/', '_').replace('\\', '_').replace(':', '_')  # 避免文件夹名称错误
    ft_dir = os.path.join(root_dir, clean_ft)
    os.makedirs(ft_dir, exist_ok=True)

# 4. 遍历数据，保存纯waferMap图像（无任何额外信息）
for idx, row in df.iterrows():
    try:
        # 提取故障类型和waferMap数据
        ft = row['failureType']
        wafer_map = row['waferMap']
        
        # 确保waferMap是二维numpy数组（处理可能的格式问题）
        if not isinstance(wafer_map, np.ndarray):
            wafer_map = np.array(wafer_map)
        if wafer_map.ndim != 2:
            print(f"行索引 {idx} 的waferMap不是二维数组，跳过")
            continue
        
        # 确定保存路径
        clean_ft = str(ft).replace('/', '_').replace('\\', '_').replace(':', '_')
        ft_dir = os.path.join(root_dir, clean_ft)
        # 生成唯一文件名（用waferIndex+行索引，避免重复）
        wafer_index = row['waferIndex']
        filename = f"wafer_{wafer_index}_row_{idx}.png"
        save_path = os.path.join(ft_dir, filename)
        
        # 绘制纯图像（无坐标轴、无边框、无边距）
        # 创建一个无边距的图像对象
        fig, ax = plt.subplots(figsize=(6, 6))  # 尺寸可根据需要调整
        # 关闭所有坐标轴和边框
        ax.axis('off')
        # 去除所有边距（left/right/top/bottom设为0，wspace/hspace设为0）
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # 显示waferMap（cmap可选，根据数据特点调整，如'viridis'或'gray'）
        ax.imshow(wafer_map, cmap='gray')  # 若数据是灰度图，用'gray'更合适
        
        # 保存图像（无额外边距，高分辨率）
        plt.savefig(
            save_path,
            bbox_inches='none',  # 去除所有额外边框
            pad_inches=0,        # 无内边距
            dpi=100              # 分辨率，可根据需要提高（如200）
        )
        # 强制关闭图像，释放内存（处理大量数据时关键）
        plt.close(fig)
        
        # 进度提示（每1000张）
        if idx % 1000 == 0:
            print(f"已处理 {idx} 张，当前保存：{save_path}")
            
    except Exception as e:
        print(f"处理行 {idx} 失败：{str(e)}")

print("所有纯waferMap图像保存完成！")