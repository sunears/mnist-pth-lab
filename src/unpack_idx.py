from utils import get_logger
import os
import argparse
import struct
import gzip
import numpy as np
from PIL import Image
import csv

def open_idx_images(path):
    """
    读取 MNIST 格式的图像文件 (IDX3-ubyte)。
    
    Args:
        path (str): 文件路径 (支持 .gz 压缩文件)。
        
    Returns:
        arr (numpy.ndarray): 图像数据，形状为 (num_images, rows, cols)。
    """
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        # 读取文件头 16 字节
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("无效的图像 IDX 文件 (文件头太短)")
        
        # 解析 Header: Magic Number, 图片数量, 行数, 列数
        magic, num, rows, cols = struct.unpack('>IIII', header)
        if magic != 2051:
            raise ValueError(f'无效的图像魔数 (Magic Number): {magic}')
        
        # 读取像素数据
        data = f.read()
        expected = num * rows * cols
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size != expected:
            raise ValueError(f'图像数据大小不匹配: 获取到 {arr.size}, 预期 {expected}')
            
        # 重塑数组形状
        arr = arr.reshape(num, rows, cols)
    return arr

def open_idx_labels(path):
    """
    读取 MNIST 格式的标签文件 (IDX1-ubyte)。
    
    Args:
        path (str): 文件路径 (支持 .gz 压缩文件)。
        
    Returns:
        arr (numpy.ndarray): 标签数据，形状为 (num_images,)。
    """
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        # 读取文件头 8 字节
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("无效的标签 IDX 文件 (文件头太短)")
            
        # 解析 Header: Magic Number, 标签数量
        magic, num = struct.unpack('>II', header)
        if magic != 2049:
            raise ValueError(f'无效的标签魔数 (Magic Number): {magic}')
        
        # 读取标签数据
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size != num:
            raise ValueError(f'标签数据大小不匹配: 获取到 {arr.size}, 预期 {num}')
    return arr

def ensure_dir(p):
    """确保目录存在，如果不存在则创建。"""
    os.makedirs(p, exist_ok=True)

def save_images_and_labels(images, labels, out_dir, fmt='png', write_txt=False):
    """
    将图像和标签保存为文件。
    
    Args:
        images (numpy.ndarray): 图像数组.
        labels (numpy.ndarray): 标签数组.
        out_dir (str): 输出目录.
        fmt (str): 图片保存格式 ('png').
        write_txt (bool): 是否同时保存对应的 .txt 标签文件.
    """
    ensure_dir(out_dir)
    img_dir = os.path.join(out_dir, 'images')
    ensure_dir(img_dir)
    if write_txt:
        txt_dir = os.path.join(out_dir, 'labels_txt')
        ensure_dir(txt_dir)
        
    csv_path = os.path.join(out_dir, 'labels.csv')
    
    # 打开 CSV 文件准备写入
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['filename', 'label']) # 写入表头
        
        # 遍历所有图片并保存
        for i in range(images.shape[0]):
            fn = f"{i:06d}.{fmt}"
            img_path = os.path.join(img_dir, fn)
            
            # 使用 PIL 保存图片
            im = Image.fromarray(images[i])
            im.save(img_path)
            
            # 写入 CSV 记录
            writer.writerow([fn, int(labels[i])])
            
            # 可选: 写入单独的 TXT 标签文件
            if write_txt:
                with open(os.path.join(txt_dir, f"{i:06d}.txt"), 'w', encoding='utf-8') as tf:
                    tf.write(str(int(labels[i])))
                    
    print(f"成功将 {images.shape[0]} 张图片写入 {img_dir}，标签已保存至 {csv_path}")
    if write_txt:
        print(f"同时也已写入单张标签文件至 {txt_dir}")

def main():
    """
    主入口函数。
    解析命令行参数，读取 IDX 文件并将其解压导出为图片格式。
    """
    parser = argparse.ArgumentParser(description='将 MNIST IDX 格式图像和标签解包为 PNG 和 CSV')
    parser.add_argument('--images-idx', required=True, help='图像 IDX 文件路径 (支持 .gz)')
    parser.add_argument('--labels-idx', required=True, help='标签 IDX 文件路径 (支持 .gz)')
    parser.add_argument('--out-dir', default='./unpacked', help='输出目录')
    parser.add_argument('--format', default='png', choices=['png','png8'], help='保存的图像格式')
    parser.add_argument('--write-txt', action='store_true', help='是否同时保存每个图片的 .txt 标签文件')
    args = parser.parse_args()

    print("正在读取图像文件...")
    images = open_idx_images(args.images_idx)
    print("正在读取标签文件...")
    labels = open_idx_labels(args.labels_idx)
    
    if images.shape[0] != labels.shape[0]:
        raise ValueError("错误: 图像数量与标签数量不匹配")

    # 执行保存操作
    save_images_and_labels(images, labels, args.out_dir, fmt=args.format, write_txt=args.write_txt)

if __name__ == "__main__":
    main()
