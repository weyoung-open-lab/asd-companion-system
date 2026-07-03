"""
检测FER-Autism训练集和测试集之间的图片重复/相似度
============================================
方法：
  1. 文件hash完全相同 -> 完全重复
  2. 感知哈希(pHash) -> 视觉相似（旋转、缩放、轻微变换后仍能检测）
  3. 像素级相似度 -> 量化相似程度

Usage:
  python check_leakage.py
"""

import os
import hashlib
from PIL import Image
import numpy as np
from collections import defaultdict

FER_ROOT = r"D:\BaiduNetdiskDownload\Autism emotion recogition dataset\Autism emotion recogition dataset"


def file_hash(path):
    """计算文件MD5哈希"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def perceptual_hash(path, size=16):
    """计算感知哈希（pHash简化版）"""
    try:
        img = Image.open(path).convert('L').resize((size, size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        mean = arr.mean()
        return ''.join(['1' if p > mean else '0' for p in arr.flatten()])
    except:
        return None


def hamming_distance(h1, h2):
    """两个哈希之间的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def pixel_similarity(path1, path2, size=64):
    """计算两张图片的像素相似度（0-1，1=完全相同）"""
    try:
        img1 = np.array(Image.open(path1).convert('RGB').resize((size, size)), dtype=np.float32)
        img2 = np.array(Image.open(path2).convert('RGB').resize((size, size)), dtype=np.float32)
        mse = np.mean((img1 - img2) ** 2)
        similarity = 1.0 - mse / (255.0 ** 2)
        return similarity
    except:
        return 0.0


def scan_directory(root, split):
    """扫描目录，返回 {class/filename: full_path} 和统计信息"""
    files = {}
    for cls in sorted(os.listdir(os.path.join(root, split))):
        cls_dir = os.path.join(root, split, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                key = f"{cls}/{fname}"
                files[key] = os.path.join(cls_dir, fname)
    return files


def main():
    print("=" * 60)
    print("  FER-Autism 训练集-测试集 泄漏检测")
    print("=" * 60)

    # 扫描文件
    train_files = scan_directory(FER_ROOT, "train")
    test_files = scan_directory(FER_ROOT, "test")
    print(f"\n  训练集: {len(train_files)} 张")
    print(f"  测试集: {len(test_files)} 张")

    # ========== 检测1：文件名完全相同 ==========
    print(f"\n{'='*60}")
    print("  检测1: 文件名重复")
    print(f"{'='*60}")
    train_names = {os.path.basename(p) for p in train_files.values()}
    test_names = {os.path.basename(p) for p in test_files.values()}
    common_names = train_names & test_names
    print(f"  相同文件名: {len(common_names)}")
    if common_names and len(common_names) <= 20:
        for n in sorted(common_names)[:20]:
            print(f"    {n}")

    # ========== 检测2：MD5哈希完全相同 ==========
    print(f"\n{'='*60}")
    print("  检测2: MD5完全相同（像素级完全重复）")
    print(f"{'='*60}")
    print("  计算训练集哈希...")
    train_hashes = {}
    for key, path in train_files.items():
        h = file_hash(path)
        if h not in train_hashes:
            train_hashes[h] = []
        train_hashes[h].append(key)

    print("  计算测试集哈希...")
    exact_duplicates = []
    for key, path in test_files.items():
        h = file_hash(path)
        if h in train_hashes:
            exact_duplicates.append((key, train_hashes[h]))

    print(f"  完全重复: {len(exact_duplicates)} 张测试图片在训练集中有完全相同的副本")
    for test_key, train_keys in exact_duplicates[:10]:
        print(f"    TEST: {test_key}  <->  TRAIN: {train_keys[0]}")

    # ========== 检测3：感知哈希相似 ==========
    print(f"\n{'='*60}")
    print("  检测3: 感知哈希（pHash）相似度检测")
    print(f"{'='*60}")
    print("  计算训练集pHash...")
    train_phash = {}
    for key, path in train_files.items():
        h = perceptual_hash(path)
        if h:
            train_phash[key] = h

    print("  计算测试集pHash并比较...")
    similar_pairs = []  # (test_key, train_key, hamming_dist)

    for test_key, test_path in test_files.items():
        test_h = perceptual_hash(test_path)
        if not test_h:
            continue

        for train_key, train_h in train_phash.items():
            dist = hamming_distance(test_h, train_h)
            # pHash size=16, total bits=256
            # dist < 20 means very similar (~92% similar)
            # dist < 10 means nearly identical (~96% similar)
            if dist < 20:
                similar_pairs.append((test_key, train_key, dist))

    # Remove exact duplicates from similar pairs
    exact_test_keys = {t for t, _ in exact_duplicates}
    near_duplicates = [(t, tr, d) for t, tr, d in similar_pairs
                       if t not in exact_test_keys and d > 0]

    print(f"  近似重复 (pHash距离<20): {len(near_duplicates)} 对")
    print(f"  其中极相似 (pHash距离<10): {sum(1 for _,_,d in near_duplicates if d < 10)} 对")

    # Show some examples
    near_duplicates.sort(key=lambda x: x[2])
    for test_key, train_key, dist in near_duplicates[:15]:
        print(f"    dist={dist:>3d}: TEST {test_key}  <->  TRAIN {train_key}")

    # ========== 检测4：按类别统计 ==========
    print(f"\n{'='*60}")
    print("  检测4: 按类别统计泄漏")
    print(f"{'='*60}")

    all_leaked_test = set()
    # From exact duplicates
    for test_key, _ in exact_duplicates:
        all_leaked_test.add(test_key)
    # From near duplicates (dist < 15)
    for test_key, _, dist in near_duplicates:
        if dist < 15:
            all_leaked_test.add(test_key)

    class_stats = defaultdict(lambda: {"total": 0, "leaked": 0})
    for key in test_files:
        cls = key.split("/")[0]
        class_stats[cls]["total"] += 1
        if key in all_leaked_test:
            class_stats[cls]["leaked"] += 1

    print(f"\n  {'类别':<15} {'测试总数':>8} {'泄漏数':>8} {'泄漏率':>8}")
    print(f"  {'-'*42}")
    total_leaked = 0
    total_test = 0
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        rate = s['leaked'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {cls:<15} {s['total']:>8} {s['leaked']:>8} {rate:>7.1f}%")
        total_leaked += s['leaked']
        total_test += s['total']

    overall_rate = total_leaked / total_test * 100 if total_test > 0 else 0
    print(f"  {'-'*42}")
    print(f"  {'总计':<15} {total_test:>8} {total_leaked:>8} {overall_rate:>7.1f}%")

    # ========== 总结 ==========
    print(f"\n{'='*60}")
    print("  总结")
    print(f"{'='*60}")
    print(f"  完全重复: {len(exact_duplicates)} 张")
    print(f"  近似重复 (增强变换): {len([x for x in near_duplicates if x[2] < 15])} 张")
    print(f"  总泄漏: {total_leaked}/{total_test} ({overall_rate:.1f}%)")

    if overall_rate > 20:
        print(f"\n  ⚠️ 泄漏严重（>{overall_rate:.0f}%），建议重新划分数据集")
    elif overall_rate > 5:
        print(f"\n  ⚠️ 存在一定泄漏，建议去重后重新划分")
    else:
        print(f"\n  ✓ 泄漏程度较轻，可接受")

    # Save report
    report = {
        "train_count": len(train_files),
        "test_count": len(test_files),
        "exact_duplicates": len(exact_duplicates),
        "near_duplicates_lt15": len([x for x in near_duplicates if x[2] < 15]),
        "near_duplicates_lt10": len([x for x in near_duplicates if x[2] < 10]),
        "total_leaked": total_leaked,
        "leak_rate_pct": round(overall_rate, 1),
        "per_class": {cls: dict(s) for cls, s in class_stats.items()},
    }

    import json
    report_path = os.path.join(os.path.dirname(FER_ROOT), "leakage_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
