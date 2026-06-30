import os
import re
import json
import glob
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.amp import autocast
import torchvision.transforms as transforms
import timm


CLASS_NAMES = ['negative', 'positive']


def parse_model_path(model_path):
    filename = os.path.basename(model_path)
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


def normalize_model_name(model_name):
    if model_name.startswith('model_'):
        model_name = model_name[6:]
    name_map = {
        'mobilenetv3_large': 'mobilenetv3_large_100',
        'mobilenetv3_small': 'mobilenetv3_small_100',
        'densenet121': 'densenet121',
        'resnet50': 'resnet50',
        'efficientnet_b0': 'efficientnet_b0',
        'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224',
    }
    return name_map.get(model_name, model_name)


def get_input_size(model_name):
    model_name = normalize_model_name(model_name)
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
    model_name = normalize_model_name(model_name)
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


def _scan_model_dir(model_dir):
    all_pth = glob.glob(os.path.join(model_dir, '**', '*.pth'), recursive=True)
    groups = defaultdict(list)
    for pth_path in all_pth:
        info = parse_model_path(pth_path)
        if info is None:
            continue
        key = (info['model_name'], info['seed'])
        groups[key].append(info['path'])

    for key in groups:
        groups[key] = sorted(groups[key])

    complete_groups = {}
    for key, fold_list in groups.items():
        if len(fold_list) == 5:
            complete_groups[key] = fold_list

    return complete_groups


def recognize(image_path, model_dir='model', device=None, ensemble_mode='soft'):
    """
    对单张图片进行识别，使用 model_dir 中的 5-fold 集成模型。

    Args:
        image_path: 图片文件路径
        model_dir: 模型权重目录，默认 'model'
        device: 推理设备，默认自动选择
        ensemble_mode: 集成方式，'soft' 或 'hard'

    Returns:
        JSON 格式的识别结果字符串
    """
    if not os.path.isfile(image_path):
        return json.dumps({
            'success': False,
            'error': f'图片文件不存在: {image_path}'
        }, ensure_ascii=False, indent=2)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_groups = _scan_model_dir(model_dir)
    if not model_groups:
        return json.dumps({
            'success': False,
            'error': f'未在 {model_dir} 中找到完整的 5-fold 模型'
        }, ensure_ascii=False, indent=2)

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'图片加载失败: {str(e)}'
        }, ensure_ascii=False, indent=2)

    all_results = []

    for (model_name, seed), model_paths in model_groups.items():
        img_size = get_input_size(model_name)
        transform = build_inference_transform(img_size)
        input_tensor = transform(image).unsqueeze(0).to(device)

        use_cuda = device.type == 'cuda'
        amp_device_type = 'cuda' if use_cuda else 'cpu'

        models_list = []
        for pth_path in model_paths:
            state_dict = torch.load(pth_path, map_location=device, weights_only=True)
            model = create_model(model_name, device, img_size, state_dict)
            model.load_state_dict(state_dict)
            model.eval()
            models_list.append(model)

        fold_probs = []
        fold_preds = []

        with torch.no_grad():
            for model in models_list:
                with autocast(amp_device_type, enabled=use_cuda):
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1)
                fold_probs.append(probabilities)
                fold_preds.append(prediction.item())

        stacked_probs = torch.cat(fold_probs, dim=0)
        mean_probs = torch.mean(stacked_probs, dim=0)

        if ensemble_mode == 'soft':
            final_pred = int(torch.argmax(mean_probs).item())
        else:
            vote_count = np.bincount(fold_preds, minlength=2)
            final_pred = int(np.argmax(vote_count))

        prob_negative = round(float(mean_probs[0].item()), 6)
        prob_positive = round(float(mean_probs[1].item()), 6)

        all_results.append({
            'model_name': model_name,
            'seed': seed,
            'predicted_class': CLASS_NAMES[final_pred],
            'predicted_label': final_pred,
            'confidence': round(max(prob_negative, prob_positive), 6),
            'probabilities': {
                'negative': prob_negative,
                'positive': prob_positive
            },
            'fold_predictions': [CLASS_NAMES[p] for p in fold_preds],
            'ensemble_mode': ensemble_mode
        })

    primary = all_results[0]
    if len(all_results) > 1:
        final_pred_label = max(set(r['predicted_label'] for r in all_results),
                              key=lambda l: sum(1 for r in all_results if r['predicted_label'] == l))
        final_pred_class = CLASS_NAMES[final_pred_label]
        avg_prob_neg = round(float(np.mean([r['probabilities']['negative'] for r in all_results])), 6)
        avg_prob_pos = round(float(np.mean([r['probabilities']['positive'] for r in all_results])), 6)
    else:
        final_pred_label = primary['predicted_label']
        final_pred_class = primary['predicted_class']
        avg_prob_neg = primary['probabilities']['negative']
        avg_prob_pos = primary['probabilities']['positive']

    result = {
        'success': True,
        'image_path': image_path,
        'predicted_class': final_pred_class,
        'predicted_label': final_pred_label,
        'confidence': round(max(avg_prob_neg, avg_prob_pos), 6),
        'probabilities': {
            'negative': avg_prob_neg,
            'positive': avg_prob_pos
        },
        'models': all_results
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # import sys
    #
    # if len(sys.argv) < 2:
    #     print('用法: python recognize.py <图片路径> [模型目录]')
    #     sys.exit(1)
    #
    # img_path = sys.argv[1]
    # m_dir = sys.argv[2] if len(sys.argv) > 2 else 'model'
    img_path = "/Users/mengheqing/PycharmProjects/pts-pn-recognize/dataset/test/negative/0007.png"
    m_dir = "./model/"

    output = recognize(img_path, model_dir=m_dir)
    print(output)
