"""
检测FER-Autism训练集内部的重复/增强图片组
==========================================
目标：找出训练集中有多少"独立原图"，有多少是同一原图的增强版本

Usage:
  python check_train_duplicates.py
"""

import os
import numpy as np
from PIL import Image
from collections import defaultdict

FER_ROOT = r"D:\BaiduNetdiskDownload\Autism emotion recogition dataset\Autism emotion recogition dataset"


def perceptual_hash(path, size=16):
    """计算感知哈希"""
    try:
        img = Image.open(path).convert('L').resize((size, size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        mean = arr.mean()
        return ''.join(['1' if p > mean else '0' for p in arr.flatten()])
    except:
        return None


def hamming_distance(h1, h2):
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def main():
    print("=" * 60)
    print("  FER-Autism 训练集内部重复检测")
    print("=" * 60)

    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"  分析: {split}")
        print(f"{'='*60}")

        # 收集所有图片
        all_files = {}
        for cls in sorted(os.listdir(os.path.join(FER_ROOT, split))):
            cls_dir = os.path.join(FER_ROOT, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    key = f"{cls}/{fname}"
                    all_files[key] = os.path.join(cls_dir, fname)

        print(f"  总图片数: {len(all_files)}")

        # 按类别分组计算
        class_files = defaultdict(dict)
        for key, path in all_files.items():
            cls = key.split("/")[0]
            class_files[cls][key] = path

        total_images = 0
        total_groups = 0
        total_unique = 0

        print(f"\n  {'类别':<12} {'图片数':>6} {'原图组数':>8} {'独立原图':>8} {'增强率':>8}")
        print(f"  {'-'*50}")

        for cls in sorted(class_files.keys()):
            files = class_files[cls]
            n_images = len(files)

            # 计算所有pHash
            hashes = {}
            for key, path in files.items():
                h = perceptual_hash(path)
                if h:
                    hashes[key] = h

            # 聚类：pHash距离<15的归为一组（同一原图的增强版本）
            assigned = set()
            groups = []

            keys = list(hashes.keys())
            for i in range(len(keys)):
                if keys[i] in assigned:
                    continue
                group = [keys[i]]
                assigned.add(keys[i])

                for j in range(i + 1, len(keys)):
                    if keys[j] in assigned:
                        continue
                    dist = hamming_distance(hashes[keys[i]], hashes[keys[j]])
                    if dist < 15:
                        group.append(keys[j])
                        assigned.add(keys[j])

                groups.append(group)

            # 没有匹配到pHash的图片，每张算独立一组
            unmatched = n_images - len(assigned)
            n_groups = len(groups) + unmatched
            aug_rate = n_images / n_groups if n_groups > 0 else 1

            print(f"  {cls:<12} {n_images:>6} {n_groups:>8} {n_groups:>8} {aug_rate:>7.1f}x")

            total_images += n_images
            total_groups += n_groups

            # 打印最大的几个组（展示增强程度）
            groups.sort(key=len, reverse=True)
            if groups and len(groups[0]) > 1:
                top3 = groups[:3]
                for g_idx, group in enumerate(top3):
                    if len(group) > 1:
                        print(f"           组{g_idx+1} ({len(group)}张): {group[0]}, {group[1]}...")

        avg_aug = total_images / total_groups if total_groups > 0 else 1
        print(f"  {'-'*50}")
        print(f"  {'总计':<12} {total_images:>6} {total_groups:>8} {total_groups:>8} {avg_aug:>7.1f}x")

        print(f"\n  结论: {split}集 {total_images} 张图片来自约 {total_groups} 张独立原图")
        print(f"  平均每张原图生成了 {avg_aug:.1f} 个增强版本")

    # 跨集检测：同一原图是否同时出现在train和test
    print(f"\n{'='*60}")
    print(f"  跨集检测: 同一原图是否同时在train和test中")
    print(f"{'='*60}")

    train_hashes = {}
    for cls in sorted(os.listdir(os.path.join(FER_ROOT, "train"))):
        cls_dir = os.path.join(FER_ROOT, "train", cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(cls_dir, fname)
                h = perceptual_hash(path)
                if h:
                    train_hashes[f"{cls}/{fname}"] = h

    test_hashes = {}
    for cls in sorted(os.listdir(os.path.join(FER_ROOT, "test"))):
        cls_dir = os.path.join(FER_ROOT, "test", cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(cls_dir, fname)
                h = perceptual_hash(path)
                if h:
                    test_hashes[f"{cls}/{fname}"] = h

    # 对每张测试图，找训练集中最相似的
    cross_leak = []
    for test_key, test_h in test_hashes.items():
        best_dist = 999
        best_train = None
        for train_key, train_h in train_hashes.items():
            dist = hamming_distance(test_h, train_h)
            if dist < best_dist:
                best_dist = dist
                best_train = train_key
        cross_leak.append((test_key, best_train, best_dist))

    cross_leak.sort(key=lambda x: x[2])

    # 统计不同阈值下的泄漏
    for threshold in [5, 10, 15, 20, 25]:
        count = sum(1 for _, _, d in cross_leak if d < threshold)
        print(f"  pHash距离<{threshold}: {count}/{len(cross_leak)} 张测试图有近似训练图 ({count/len(cross_leak)*100:.1f}%)")

    # 按类别显示
    print(f"\n  pHash距离<15 的详细泄漏（按类别）:")
    cls_leak = defaultdict(lambda: {"total": 0, "leaked": 0})
    for test_key, train_key, dist in cross_leak:
        cls = test_key.split("/")[0]
        cls_leak[cls]["total"] += 1
        if dist < 15:
            cls_leak[cls]["leaked"] += 1

    for cls in sorted(cls_leak.keys()):
        s = cls_leak[cls]
        rate = s['leaked'] / s['total'] * 100
        print(f"    {cls:<12}: {s['leaked']}/{s['total']} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
