"""
FER-Autism 数据集去重脚本
==========================
1. 检测test集中与train集近似重复的图片（pHash距离<15）
2. 检测train集内部的近似重复图片，每组只保留一张
3. 检测test集内部的近似重复图片，每组只保留一张
4. 将干净的数据集复制到新文件夹

Usage:
  python clean_dataset.py
"""

import os
import shutil
import json
import numpy as np
from PIL import Image
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
FER_ROOT = r"D:\BaiduNetdiskDownload\Autism emotion recogition dataset\Autism emotion recogition dataset"
CLEAN_ROOT = r"D:\BaiduNetdiskDownload\FER_Autism_Clean"
PHASH_THRESHOLD = 15  # pHash距离阈值，<15视为同一原图


# ============================================================
# 工具函数
# ============================================================
def perceptual_hash(path, size=16):
    try:
        img = Image.open(path).convert('L').resize((size, size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        mean = arr.mean()
        return ''.join(['1' if p > mean else '0' for p in arr.flatten()])
    except:
        return None


def hamming_distance(h1, h2):
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def scan_files(root, split):
    """扫描目录，返回 [(key, path)] 列表"""
    files = []
    split_dir = os.path.join(root, split)
    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                key = f"{cls}/{fname}"
                path = os.path.join(cls_dir, fname)
                files.append((key, path))
    return files


def find_internal_duplicates(files_with_hash):
    """找出内部重复组，每组只保留第一张，返回要删除的key集合"""
    remove = set()
    assigned = set()
    keys = list(files_with_hash.keys())
    hashes = list(files_with_hash.values())

    for i in range(len(keys)):
        if keys[i] in assigned:
            continue
        assigned.add(keys[i])
        for j in range(i + 1, len(keys)):
            if keys[j] in assigned:
                continue
            if hamming_distance(hashes[i], hashes[j]) < PHASH_THRESHOLD:
                assigned.add(keys[j])
                remove.add(keys[j])  # 保留第一张，删除后面的

    return remove


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  FER-Autism 数据集去重")
    print(f"  pHash阈值: {PHASH_THRESHOLD}")
    print(f"  输出目录: {CLEAN_ROOT}")
    print("=" * 60)

    # ========== Step 1: 扫描并计算哈希 ==========
    print("\n  Step 1: 扫描文件并计算pHash...")
    train_files = scan_files(FER_ROOT, "train")
    test_files = scan_files(FER_ROOT, "test")
    print(f"  原始训练集: {len(train_files)} 张")
    print(f"  原始测试集: {len(test_files)} 张")

    train_hashes = {}
    for key, path in train_files:
        h = perceptual_hash(path)
        if h:
            train_hashes[key] = h

    test_hashes = {}
    for key, path in test_files:
        h = perceptual_hash(path)
        if h:
            test_hashes[key] = h

    print(f"  成功计算pHash: train={len(train_hashes)}, test={len(test_hashes)}")

    # ========== Step 2: 去除跨集泄漏（从test中删除） ==========
    print(f"\n  Step 2: 检测跨集泄漏...")
    cross_leak_test = set()
    for test_key, test_h in test_hashes.items():
        for train_key, train_h in train_hashes.items():
            if hamming_distance(test_h, train_h) < PHASH_THRESHOLD:
                cross_leak_test.add(test_key)
                break  # 找到一个就够了

    print(f"  测试集中与训练集近似重复: {len(cross_leak_test)} 张")
    for key in sorted(cross_leak_test):
        print(f"    删除: {key}")

    # ========== Step 3: 去除训练集内部重复 ==========
    print(f"\n  Step 3: 检测训练集内部重复...")
    # 按类别处理，避免跨类误删
    train_internal_remove = set()
    for cls in sorted(set(k.split("/")[0] for k in train_hashes)):
        cls_hashes = {k: v for k, v in train_hashes.items() if k.startswith(cls + "/")}
        cls_remove = find_internal_duplicates(cls_hashes)
        train_internal_remove.update(cls_remove)
        if cls_remove:
            print(f"    {cls}: 删除 {len(cls_remove)} 张内部重复")

    print(f"  训练集内部重复总计: {len(train_internal_remove)} 张")

    # ========== Step 4: 去除测试集内部重复 ==========
    print(f"\n  Step 4: 检测测试集内部重复...")
    # 先排除已标记的跨集泄漏
    clean_test_hashes = {k: v for k, v in test_hashes.items() if k not in cross_leak_test}
    test_internal_remove = set()
    for cls in sorted(set(k.split("/")[0] for k in clean_test_hashes)):
        cls_hashes = {k: v for k, v in clean_test_hashes.items() if k.startswith(cls + "/")}
        cls_remove = find_internal_duplicates(cls_hashes)
        test_internal_remove.update(cls_remove)
        if cls_remove:
            print(f"    {cls}: 删除 {len(cls_remove)} 张内部重复")

    print(f"  测试集内部重复总计: {len(test_internal_remove)} 张")

    # ========== Step 5: 复制干净数据集 ==========
    print(f"\n  Step 5: 创建干净数据集...")

    # 合并所有要删除的
    all_test_remove = cross_leak_test | test_internal_remove

    # 统计
    clean_train_count = len(train_files) - len(train_internal_remove)
    clean_test_count = len(test_files) - len(all_test_remove)

    print(f"\n  清理结果:")
    print(f"    训练集: {len(train_files)} -> {clean_train_count} (删除 {len(train_internal_remove)})")
    print(f"    测试集: {len(test_files)} -> {clean_test_count} (删除 {len(all_test_remove)})")

    # 按类别统计
    print(f"\n  {'类别':<12} {'Train原':>6} {'Train新':>6} {'Test原':>6} {'Test新':>6}")
    print(f"  {'-'*45}")

    for cls in sorted(set(k.split("/")[0] for k, _ in train_files)):
        train_orig = sum(1 for k, _ in train_files if k.startswith(cls + "/"))
        train_removed = sum(1 for k in train_internal_remove if k.startswith(cls + "/"))
        test_orig = sum(1 for k, _ in test_files if k.startswith(cls + "/"))
        test_removed = sum(1 for k in all_test_remove if k.startswith(cls + "/"))
        print(f"  {cls:<12} {train_orig:>6} {train_orig - train_removed:>6} "
              f"{test_orig:>6} {test_orig - test_removed:>6}")

    # 创建输出目录并复制
    if os.path.exists(CLEAN_ROOT):
        shutil.rmtree(CLEAN_ROOT)

    copied_train, copied_test = 0, 0

    for key, path in train_files:
        if key in train_internal_remove:
            continue
        cls, fname = key.split("/")
        dest_dir = os.path.join(CLEAN_ROOT, "train", cls)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dest_dir, fname))
        copied_train += 1

    for key, path in test_files:
        if key in all_test_remove:
            continue
        cls, fname = key.split("/")
        dest_dir = os.path.join(CLEAN_ROOT, "test", cls)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(dest_dir, fname))
        copied_test += 1

    print(f"\n  复制完成:")
    print(f"    训练集: {copied_train} 张 -> {os.path.join(CLEAN_ROOT, 'train')}")
    print(f"    测试集: {copied_test} 张 -> {os.path.join(CLEAN_ROOT, 'test')}")

    # 保存清理报告
    report = {
        "original": {"train": len(train_files), "test": len(test_files)},
        "removed": {
            "train_internal_duplicates": len(train_internal_remove),
            "test_cross_leak": len(cross_leak_test),
            "test_internal_duplicates": len(test_internal_remove),
        },
        "clean": {"train": copied_train, "test": copied_test},
        "removed_files": {
            "train": sorted(train_internal_remove),
            "test_leak": sorted(cross_leak_test),
            "test_internal": sorted(test_internal_remove),
        },
        "phash_threshold": PHASH_THRESHOLD,
    }
    report_path = os.path.join(CLEAN_ROOT, "cleaning_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  清理报告: {report_path}")
    print(f"\n  后续使用干净数据集路径:")
    print(f"    {CLEAN_ROOT}")


if __name__ == "__main__":
    main()
