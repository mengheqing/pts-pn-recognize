import os
import re
import csv
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import torchvision.transforms as transforms

import timm
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(description='批量绘制 ROC 和 PR 曲线')
    parser.add_argument('--model_dir', type=str, default='outputs',
                        help='模型权重所在目录')
    parser.add_argument('--test_dir', type=str, default='dataset/test',
                        help='测试集目录')
    parser.add_argument('--output_dir', type=str, default='curve_plots',
                        help='曲线图保存目录')
    parser.add_argument('--ensemble_mode', type=str, default='soft',
                        choices=['soft', 'hard'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


def parse_model_path(model_path):
    """
    从文件名中提取信息
    例: best_model_densenet121_fold3_seed123_20260626_121205.pth
    返回: {'model_name': 'densenet121', 'fold': 3, 'seed': 123, 'timestamp': '20260626_121205'}
    """
    filename = os.path.basename(model_path)

    # 匹配模式: best_model_{name}_fold{N}_seed{S}_{timestamp}.pth
    pattern = r'best_(.+)_fold(\d+)_seed(\d+)_(\d+_\d+)\.pth'
    match = re.search(pattern, filename)

    if not match:
        return None

    return {
        'model_name': match.group(1),
        'fold': int(match.group(2)),
        'seed': int(match.group(3)),
        'timestamp': match.group(4),
        'path': model_path
    }


def group_models(model_dir):
    """
    扫描目录，按 model_name + seed 分组
    返回: {(model_name, seed): [fold1_path, fold2_path, ...], ...}
    """
    import glob

    # 递归搜索所有 .pth 文件
    all_pth = glob.glob(os.path.join(model_dir, '**', '*.pth'), recursive=True)

    print(f"找到 {len(all_pth)} 个 .pth 文件")

    groups = defaultdict(list)

    for pth_path in all_pth:
        info = parse_model_path(pth_path)
        if info is None:
            print(f"⚠️  无法解析文件名: {os.path.basename(pth_path)}")
            continue

        key = (info['model_name'], info['seed'])
        groups[key].append(info)

    # 每组按 fold 排序
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x['fold'])

    # 打印分组信息（调试用）
    print(f"\n找到 {len(groups)} 个模型组:")
    for key, fold_list in groups.items():
        fold_ids = [f['fold'] for f in fold_list]
        print(f"  • {key[0]} | seed={key[1]} | {len(fold_list)} folds: {fold_ids}")

    # 只保留有完整 5-fold 的组
    complete_groups = {}
    for key, fold_list in groups.items():
        if len(fold_list) == 5:
            complete_groups[key] = [f['path'] for f in fold_list]
        else:
            print(f"\n⚠️  跳过 {key[0]} seed={key[1]} (只有 {len(fold_list)} 个 fold)")

    return complete_groups


def normalize_model_name(model_name):
    """将文件名中的模型名映射为 timm 标准名"""
    # 移除前缀 'model_'
    if model_name.startswith('model_'):
        model_name = model_name[6:]

    # 常见映射
    name_map = {
        'mobilenetv3_large': 'mobilenetv3_large_100',
        'mobilenetv3_small': 'mobilenetv3_small_100',
        'densenet121': 'densenet121',
        'resnet50': 'resnet50',
        'efficientnet_b0': 'efficientnet_b0',
        'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224',
    }

    if model_name in name_map:
        return name_map[model_name]
    return model_name


def get_input_size(model_name):
    model_name = normalize_model_name(model_name)  # ← 添加这行
    model_lower = model_name.lower()
    if 'swin' in model_lower:
        if '224' in model_name:
            return 224
        elif '256' in model_name:
            return 256
        elif '384' in model_name:
            return 384
    if 'vit' in model_lower:
        if '224' in model_name:
            return 224
        elif '384' in model_name:
            return 384
        elif '512' in model_name:
            return 512
    if 'beit' in model_lower or 'deit' in model_lower:
        if '224' in model_name:
            return 224
        elif '384' in model_name:
            return 384
    return 256


def create_model(model_name, device, img_size, state_dict=None):
    model_name = normalize_model_name(model_name)  # ← 添加这行

    model_kwargs = {
        'pretrained': False,
        'num_classes': 2,
        'drop_rate': 0.0,
    }
    if any(x in model_name.lower() for x in ['swin', 'vit', 'beit', 'deit']):
        model_kwargs['img_size'] = img_size

    model = timm.create_model(model_name, **model_kwargs)

    if state_dict is not None:
        has_wrapped_fc = any('fc.1.weight' in k for k in state_dict.keys())
        has_wrapped_classifier = any('classifier.1.weight' in k for k in state_dict.keys())
        has_wrapped_head = any('head.1.weight' in k for k in state_dict.keys())

        if has_wrapped_fc and hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 2))
        elif has_wrapped_classifier and hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 2))
        elif has_wrapped_head and hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 2))

    return model.to(device)


def build_inference_transform(img_size):
    return transforms.Compose([
        transforms.Resize((600, 200)),
        transforms.Pad((28, 0, 28, 0), fill=0),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TestDatasetWithPath(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['negative', 'positive']
        self.samples = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"测试集目录不存在: {root_dir}")

        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 目录不存在 {class_dir}")
                continue
            file_names = sorted([
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
            for image_name in file_names:
                self.samples.append((os.path.join(class_dir, image_name), class_index))

        if len(self.samples) == 0:
            raise ValueError("测试集中没有找到任何图片!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"错误加载 {image_path}: {e}")
            image = Image.new('RGB', (200, 600), (0, 0, 0))
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image_path


def ensemble_predict(model_paths, test_loader, device, ensemble_mode='soft'):
    """5-fold 集成推理"""
    models_list = []

    model_name = parse_model_path(model_paths[0])['model_name']
    img_size = get_input_size(model_name)

    use_cuda = device.type == "cuda"
    use_amp = use_cuda
    amp_device_type = "cuda" if use_cuda else "cpu"

    for model_path in model_paths:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model = create_model(model_name, device, img_size, state_dict)
        model.load_state_dict(state_dict)
        model.eval()
        models_list.append(model)

    all_true_labels = []
    all_pred_labels = []
    all_pred_probs_positive = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda).long()

            batch_fold_probs = []
            batch_fold_preds = []

            for model in models_list:
                with autocast(amp_device_type, enabled=use_amp):
                    outputs = model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                batch_fold_probs.append(probabilities.unsqueeze(0))
                batch_fold_preds.append(predictions.unsqueeze(0))

            batch_fold_probs = torch.cat(batch_fold_probs, dim=0)
            batch_fold_preds = torch.cat(batch_fold_preds, dim=0)

            if ensemble_mode == "soft":
                mean_probabilities = torch.mean(batch_fold_probs, dim=0)
                final_predictions = torch.argmax(mean_probabilities, dim=1)
                final_prob_positive = mean_probabilities[:, 1]
            elif ensemble_mode == "hard":
                final_predictions = []
                final_prob_positive = []
                mean_probabilities = torch.mean(batch_fold_probs, dim=0)
                for sample_idx in range(batch_fold_preds.shape[1]):
                    votes = batch_fold_preds[:, sample_idx].cpu().numpy()
                    vote_count = np.bincount(votes, minlength=2)
                    final_pred = int(np.argmax(vote_count))
                    final_predictions.append(final_pred)
                    final_prob_positive.append(float(mean_probabilities[sample_idx, 1].item()))
                final_predictions = torch.tensor(final_predictions, device=device)
                final_prob_positive = torch.tensor(final_prob_positive, device=device)

            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(final_predictions.cpu().numpy().tolist())
            all_pred_probs_positive.extend(final_prob_positive.cpu().numpy().tolist())

    return all_true_labels, all_pred_labels, all_pred_probs_positive


def plot_roc_curve(y_true, y_probs, output_path, model_name, seed, auc_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5000)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve - {model_name} (seed={seed})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
             label=f'Optimal Threshold = {optimal_threshold:.3f}')
    plt.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_threshold, fpr[optimal_idx], tpr[optimal_idx]


def plot_pr_curve(y_true, y_probs, output_path, model_name, seed, ap_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    baseline = sum(y_true) / len(y_true)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {ap_score:.4f})')
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--',
                label=f'Random Classifier (AP = {baseline:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name} (seed={seed})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    plt.plot(optimal_recall, optimal_precision, 'ro', markersize=10,
             label=f'Best F1 = {optimal_f1:.3f} @ threshold={optimal_threshold:.3f}')
    plt.legend(loc="lower left", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_threshold, optimal_precision, optimal_recall, optimal_f1


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def process_one_group(model_name, seed, model_paths, test_loader, device,
                      output_dir, ensemble_mode):
    """处理一个 model+seed 组合"""
    print(f"\n{'=' * 70}")
    print(f"处理: {model_name} | seed={seed}")
    print(f"{'=' * 70}")

    # 集成推理
    y_true, y_pred, y_probs = ensemble_predict(
        model_paths, test_loader, device, ensemble_mode
    )

    # 计算指标
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    metrics = compute_metrics(y_true, y_pred)

    # 创建输出目录
    save_dir = os.path.join(output_dir, model_name, f'seed{seed}')
    os.makedirs(save_dir, exist_ok=True)

    # 绘制 ROC
    roc_path = os.path.join(save_dir, 'roc_curve.png')
    roc_info = plot_roc_curve(y_true, y_probs, roc_path, model_name, seed, auc)

    # 绘制 PR
    pr_path = os.path.join(save_dir, 'pr_curve.png')
    pr_info = plot_pr_curve(y_true, y_probs, pr_path, model_name, seed, ap)

    # 保存汇总
    summary_path = os.path.join(save_dir, 'metrics_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Model', model_name])
        writer.writerow(['Seed', seed])
        writer.writerow(['Ensemble Mode', ensemble_mode])
        writer.writerow(['Num Samples', len(y_true)])
        writer.writerow([''])
        writer.writerow(['Accuracy', f"{metrics['accuracy']:.4f}"])
        writer.writerow(['Precision', f"{metrics['precision']:.4f}"])
        writer.writerow(['Recall', f"{metrics['recall']:.4f}"])
        writer.writerow(['F1-Score', f"{metrics['f1']:.4f}"])
        writer.writerow(['AUC-ROC', f"{auc:.4f}"])
        writer.writerow(['AUC-PR (AP)', f"{ap:.4f}"])
        writer.writerow([''])
        writer.writerow(['ROC Optimal Threshold', f"{roc_info[0]:.4f}"])
        writer.writerow(['ROC Optimal FPR', f"{roc_info[1]:.4f}"])
        writer.writerow(['ROC Optimal TPR', f"{roc_info[2]:.4f}"])
        writer.writerow([''])
        writer.writerow(['PR Optimal Threshold', f"{pr_info[0]:.4f}"])
        writer.writerow(['PR Optimal Precision', f"{pr_info[1]:.4f}"])
        writer.writerow(['PR Optimal Recall', f"{pr_info[2]:.4f}"])
        writer.writerow(['PR Optimal F1', f"{pr_info[3]:.4f}"])
        writer.writerow([''])
        writer.writerow(['Confusion Matrix', ''])
        writer.writerow(['TN', metrics['tn']])
        writer.writerow(['FP', metrics['fp']])
        writer.writerow(['FN', metrics['fn']])
        writer.writerow(['TP', metrics['tp']])

    print(f"✅ 完成: {model_name} seed={seed}")
    print(f"   AUC-ROC: {auc:.4f} | AUC-PR: {ap:.4f} | Acc: {metrics['accuracy']:.4f}")
    print(f"   保存目录: {save_dir}")

    return {
        'model_name': model_name,
        'seed': seed,
        'auc': auc,
        'ap': ap,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 扫描模型文件
    print(f"\n扫描模型目录: {args.model_dir}")
    model_groups = group_models(args.model_dir)

    if len(model_groups) == 0:
        print("❌ 没有找到完整的 5-fold 模型组,请检查模型目录")
        return

    print(f"\n找到 {len(model_groups)} 个完整的模型组:")
    for (model_name, seed), paths in model_groups.items():
        print(f"  • {model_name} | seed={seed} | {len(paths)} folds")

    # 加载测试集
    print(f"\n加载测试集: {args.test_dir}")
    # 使用第一个模型的输入尺寸（假设所有模型用同一尺寸,如果不同需要单独处理）
    first_model_name = list(model_groups.keys())[0][0]
    img_size = get_input_size(first_model_name)

    transform = build_inference_transform(img_size)
    test_dataset = TestDatasetWithPath(args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"测试集样本数: {len(test_dataset)}")

    # 批量处理
    all_results = []

    for idx, ((model_name, seed), model_paths) in enumerate(model_groups.items(), 1):
        print(f"\n[{idx}/{len(model_groups)}]", end=" ")

        result = process_one_group(
            model_name, seed, model_paths, test_loader, device,
            args.output_dir, args.ensemble_mode
        )
        all_results.append(result)

    # 保存总汇总
    overall_summary_path = os.path.join(args.output_dir, 'overall_summary.csv')
    with open(overall_summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_name', 'seed', 'auc', 'ap', 'accuracy', 'precision', 'recall', 'f1'
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'=' * 70}")
    print(f"🎉 全部完成!")
    print(f"{'=' * 70}")
    print(f"总共处理: {len(model_groups)} 个模型组")
    print(f"输出目录: {args.output_dir}")
    print(f"总汇总文件: {overall_summary_path}")


if __name__ == '__main__':
    main()