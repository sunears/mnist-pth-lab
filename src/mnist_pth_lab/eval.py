import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from mnist_pth_lab.model import build_model
from mnist_pth_lab.dataset import get_dataloaders
from mnist_pth_lab.utils import load_model, get_logger, set_seed


def evaluate(args):
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger = get_logger("Eval")
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    model = build_model()
    try:
        model = load_model(model, args.model, device)
    except FileNotFoundError:
        logger.error(f"无法找到模型文件: {args.model}")
        return

    model.eval()

    all_preds = []
    all_labels = []
    images_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            images_list.extend(images.cpu())

    acc = accuracy_score(all_labels, all_preds)
    logger.info(f"整体准确率 (Overall Accuracy): {acc:.4f}")

    report = classification_report(all_labels, all_preds, digits=4)
    print("\n分类报告 (Classification Report):\n")
    print(report)

    with open(os.path.join(args.output_dir, 'eval_report.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签 (Predicted)')
    plt.ylabel('真实标签 (True)')
    plt.title('混淆矩阵 (Confusion Matrix)')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()

    mistakes = []
    for i, (true, pred) in enumerate(zip(all_labels, all_preds)):
        if true != pred:
            mistakes.append((i, true, pred))

    for idx, (i, true, pred) in enumerate(mistakes[:5]):
        img_tensor = images_list[i]
        img_np = img_tensor.squeeze().numpy()
        img_np = (img_np * 0.3081) + 0.1307
        img_np = np.clip(img_np, 0, 1)

        plt.figure()
        plt.imshow(img_np, cmap='gray')
        plt.title(f"真值(True): {true}, 预测(Pred): {pred}")
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, f'mistake_{idx}_T{true}_P{pred}.png'))
        plt.close()

    logger.info(f"评估完成。结果已保存至 {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate CNN on MNIST")
    parser.add_argument('--model', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小/Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='设备/Device (cpu/cuda)')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader 工作进程数')
    parser.add_argument('--output-dir', type=str, default='experiments/output', help='输出目录')

    args = parser.parse_args()
    evaluate(args)
