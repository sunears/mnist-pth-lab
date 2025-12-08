#!/usr/bin/env python3
"""
将图片和标签（CSV、单张 .txt 文件或按分类文件夹结构）打包成 MNIST IDX 格式文件。
生成的 images idx 魔数 (Magic Number) 为 2051，labels idx 魔数为 2049。
支持可选的 Gzip 压缩。

示例:
  # 使用 labels.csv
  python scripts/pack_idx.py --images-dir ./unpacked/images --labels-csv ./unpacked/labels.csv \
    --out-images-idx new-train-images-idx3-ubyte.gz --out-labels-idx new-train-labels-idx1-ubyte.gz --gzip

  # 使用位于 ./unpacked/labels_txt/ 下的单张 txt 标签文件
  python scripts/pack_idx.py --images-dir ./unpacked/images --labels-txt-dir ./unpacked/labels_txt \
    --out-images-idx out-images.idx --out-labels-idx out-labels.idx
"""
import os
import argparse
import struct
import gzip
import numpy as np
from PIL import Image
import csv

def write_idx_images(path, images: np.ndarray, gzip_out=False):
    """
    将图像数组写入 IDX3 格式文件。
    
    Args:
        path (str): 输出文件路径。
        images (np.ndarray): 图像数据数组，形状为 (N, H, W)，类型 uint8。
        gzip_out (bool): 是否使用 Gzip 压缩。
    """
    N, H, W = images.shape
    # 打包文件头: Magic(2051), 数量(N), 高(H), 宽(W)
    header = struct.pack('>IIII', 2051, N, H, W)
    
    opener = gzip.open if gzip_out else open
    with opener(path, 'wb') as f:
        f.write(header)
        f.write(images.tobytes())

def write_idx_labels(path, labels: np.ndarray, gzip_out=False):
    """
    将标签数组写入 IDX1 格式文件。
    
    Args:
        path (str): 输出文件路径。
        labels (np.ndarray): 标签数据数组，形状为 (N,)，类型 uint8。
        gzip_out (bool): 是否使用 Gzip 压缩。
    """
    N = labels.shape[0]
    # 打包文件头: Magic(2049), 数量(N)
    header = struct.pack('>II', 2049, N)
    
    opener = gzip.open if gzip_out else open
    with opener(path, 'wb') as f:
        f.write(header)
        f.write(labels.tobytes())

def read_labels_csv(path):
    """
    从 CSV 文件读取标签映射。
    CSV 格式应包含 filename 和 label 列。
    """
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 尝试获取文件名 (兼容 filename, file)
            fname = r.get('filename') or r.get('file') or list(r.values())[0]
            # 尝试获取标签 (兼容 label, label_id)
            lbl_val = r.get('label') or r.get('label_id') or list(r.values())[-1]
            lbl = int(lbl_val)
            mapping[fname] = lbl
    return mapping

def read_labels_from_txt_dir(txt_dir):
    """
    从包含 .txt 文件的目录读取标签。
    文件名应与图片对应（例如 00001.txt 对应 00001.png）。
    """
    mapping = {}
    for fn in os.listdir(txt_dir):
        if not fn.lower().endswith('.txt'):
            continue
        key = os.path.splitext(fn)[0]
        with open(os.path.join(txt_dir, fn), 'r', encoding='utf-8') as f:
            mapping[key] = int(f.read().strip())
    return mapping

def infer_from_folder_by_class(images_dir):
    """
    根据父文件夹名称推断标签。
    适用于 images_dir/0/xxx.png, images_dir/1/xxx.png 这种结构。
    """
    mapping = {}
    for root, dirs, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(('.png','.jpg','.jpeg')):
                parent = os.path.basename(root)
                try:
                    lbl = int(parent)
                except:
                    raise ValueError(f"按文件夹分类推断标签需要文件夹名称为整数 (0-9)，发现: {parent}")
                mapping[fn] = lbl
    return mapping

def load_and_prepare_image(path, size=(28,28), invert_if_white_bg=True):
    """
    读取并预处理图像：转灰度 -> 调整大小 -> (可选)反色。
    """
    im = Image.open(path).convert('L')  # 转为灰度图
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    
    # 如果背景看起来是白色的（平均像素值 > 128），则进行反色处理
    # MNIST 数据集通常是黑底白字（背景0，笔画255）
    # 而普通图片通常是白底黑字，所以需要反转
    if invert_if_white_bg and arr.mean() > 128:
        arr = 255 - arr
    return arr

def main():
    parser = argparse.ArgumentParser(description='将图片和标签打包成 MNIST IDX 格式文件')
    parser.add_argument('--images-dir', required=True, help='包含图片文件的目录 (例如 unpacked/images)')
    
    # 互斥参数组：指定标签来源
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--labels-csv', help='包含 filename, label 列的 CSV 文件')
    group.add_argument('--labels-txt-dir', help='包含单张 txt 标签文件的目录 (文件名与图片对应)')
    group.add_argument('--folder-by-class', action='store_true', help='从父文件夹名称 (0..9) 推断标签')
    
    parser.add_argument('--out-images-idx', required=True, help='输出图像 IDX 文件路径 (如果以 .gz 结尾则自动压缩)')
    parser.add_argument('--out-labels-idx', required=True, help='输出标签 IDX 文件路径 (如果以 .gz 结尾则自动压缩)')
    parser.add_argument('--gzip', action='store_true', help='强制使用 Gzip 压缩输出')
    parser.add_argument('--size', type=int, default=28, help='目标图像大小 (默认 28x28)')
    parser.add_argument('--no-invert', dest='invert', action='store_false', help='不自动反转白底图片')
    args = parser.parse_args()

    # 1. 收集标签映射
    if args.labels_csv:
        mapping = read_labels_csv(args.labels_csv)
    elif args.labels_txt_dir:
        mapping = read_labels_from_txt_dir(args.labels_txt_dir)
    elif args.folder_by_class:
        mapping = infer_from_folder_by_class(args.images_dir)
    else:
        mapping = {}

    # 2. 收集图片文件
    files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not files:
        raise ValueError("在指定目录中未找到图像文件")

    # 3. 确定顺序
    # 如果映射键是类似于 000000.png 的文件名，则尽量保持文件名排序
    files_sorted = sorted(files)
    images_list = []
    labels_list = []
    
    print(f"找到 {len(files_sorted)} 张图片，开始处理...")
    
    for fn in files_sorted:
        key = fn
        key_noext = os.path.splitext(fn)[0]
        
        # 尝试查找标签
        if args.labels_csv or args.folder_by_class:
            lbl = mapping.get(fn)
            if lbl is None:
                lbl = mapping.get(key_noext)
        else:
            lbl = mapping.get(key_noext)
            
        if lbl is None:
            raise ValueError(f"未找到文件 {fn} 的标签。请提供 CSV、txt 目录或使用按文件夹分类结构。")
            
        path = os.path.join(args.images_dir, fn)
        # 加载并预处理图片
        arr = load_and_prepare_image(path, size=(args.size,args.size), invert_if_white_bg=args.invert)
        
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
            
        # 如果是多通道图像（其实前面已经转灰度了，这里是双重保险），取单通道
        if arr.ndim == 3:
            arr = arr[...,0]
            
        images_list.append(arr)
        labels_list.append(int(lbl))
        
    # 堆叠成 numpy 数组
    images_np = np.stack(images_list).astype(np.uint8)
    labels_np = np.array(labels_list, dtype=np.uint8)
    
    # 尺寸检查
    N = images_np.shape[0]
    H = images_np.shape[1]
    W = images_np.shape[2]
    print(f"正在打包 {N} 张图片，尺寸 {H}x{W}")

    # 4. 写入文件
    # 根据文件名后缀或 --gzip 参数决定是否压缩
    use_gzip = args.gzip or args.out_images_idx.endswith('.gz')
    
    write_idx_images(args.out_images_idx, images_np, gzip_out=use_gzip)
    write_idx_labels(args.out_labels_idx, labels_np, gzip_out=use_gzip)
    
    print("已写入图像 IDX 文件:", args.out_images_idx)
    print("已写入标签 IDX 文件:", args.out_labels_idx)

if __name__ == '__main__':
    main()