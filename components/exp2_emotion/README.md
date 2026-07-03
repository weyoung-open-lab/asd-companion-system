# Exp2 — 置信度感知情感识别（FERAC 4类）

EfficientNet-B0，4 类（Natural/Anger/Fear/Joy），干净 FERAC 770（train615 / test155，合并成池做 5 折 CV）。
预训练 backbone：`models/effb0_affectnet_8cls.pth`（AffectNet 8 类微调、无分类头，strict=False 加载 + 新接 `Linear(1280,4)`）。

> 所有路径走 `common/paths.py`，不要硬编码。环境：`asd_env`（torch 2.11.0+cu128，GPU）。
> 运行前先确认数据干净（见下 §数据）。

## 数据
- `data/ferac/`：干净 FERAC 770（来自 `archive (5).zip`）。每类 Natural 184 / anger 74 / fear 49 / joy 463。
- ⚠️ Fear 全集仅 49 张（每折 val 约 9–10 张）——指标方差大，见 honest_eval 的 bootstrap CI。
- 不要混入增强版「Autism emotion recogition dataset」(1425，脏，泄漏7.1%) 或任何其他数据集。

## 四步流程（严格 0→1→2→3，上一步达标再下一步）

### 第0步：纯复现
```
python repro_cv.py                 # 完整 5 折，种子 42（忠实复现 confidence_analysis.py）
python repro_cv.py --smoke         # 1 折 × 2 epoch 自检
```
- 协议（一字不改）：Focal(γ2, α=class_weights, LS0.2) + Mixup(α0.2,p0.5) + WeightedRandomSampler
  + AdamW(1e-4,wd1e-3) + warmup5+cosine + 80ep + patience15 + 单种子42。
- 目标：Macro-F1 ≈ 0.707 ± 0.051（达标区间 0.65–0.76）。
- 输出：`artifacts/exp2/repro_cv_results.json`、`artifacts/exp2/oof/oof_logits.npz`（+`oof_paths.json`）。

### 第1步：诚实小样本评估（叠加，不重训）✅
```
python honest_eval.py --n_boot 2000 --seed 42
```
- 用第0步 OOF 预测：4×4 混淆矩阵(计数+行归一)、每类 P/R/F1、bootstrap 95% CI。
- 结果：Macro-F1 0.659；joy F1 0.919[0.900,0.937]；**fear F1 0.390[0.274,0.496]（n=49，CI宽0.22）**。
- 输出：`artifacts/exp2/honest_eval/`（confusion_matrix_*.png、per_class_metrics.json、bootstrap_ci.json）。

### 第2步：置信度门控（核心）✅
```
python confidence_gating.py --target_acc 0.90 --seed 42
```
- 温度缩放(OOF上拟合)、ECE前后+可靠性图；3 种不确定度 risk-coverage+AURC；阈值表 pre/post-T；OOF上选工作点。
- 结果：**T=0.383，ECE 0.291→0.034**；AURC margin0.093≈maxsoft0.093≈entropy0.094；工作点阈值0.73→覆盖66%/精度90%。
- 选择只在 OOF(验证面)做；真正测试集报告在第3步。
- 输出：`artifacts/exp2/confidence_gating/`（reliability_diagram.png、risk_coverage.png、gating_operating_points.json、ece_comparison.json、risk_coverage.json）。

### 第3步：部署 checkpoint ✅
```
python deploy.py --seed 42 --target_acc 0.90
```
- train615 内分层 carve val_deploy(早停+拟合T+选阈值)，test155 只报一次；保存含 T+阈值 的权重。
- 结果(test155，成品具体表现，≠CV0.707)：**Macro-F1 0.633 / Acc 0.781**；T=0.377、门控阈值0.86 → 覆盖40.6%/保留子集精度92.1%。
- 输出：`models/ferac_efficientnetb0_deploy.pth`、`artifacts/exp2/deploy_report.json`、`deploy_confusion_test155.png`。
- 纪律：部署分数≠CV 的 0.707（前者是成品在155测试集的具体表现，后者是方法的无偏估计）。

## 部署推理（system/inference 用）
```python
import torch, torch.nn as nn; from torchvision import models
ck = torch.load("models/ferac_efficientnetb0_deploy.pth", map_location="cpu")
m = models.efficientnet_b0(weights=None); m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280,4))
m.load_state_dict(ck["state_dict"]); m.eval()
# 预处理: Resize224 + ImageNet norm(ck["preprocess"])；前向后用温度缩放：
# probs = softmax(logits / ck["temperature"]); conf = probs.max()
# 门控: conf >= ck["gate_threshold"](=0.86) 才采信，否则标 uncertain → 下游走保守模式
```

## 复用的原始脚本（勿重写训练逻辑）
- `confidence_analysis.py` — 0.707 的来源（Focal+Mixup+8cls pretrain+阈值扫描）。
- `transfer_experiment.py` — no-pretrain/8cls/4cls 迁移对比（纯CE，非0.707来源）。
- `sota_benchmark_cv.py` — 9 模型 5 折对比。
- `tools/check_leakage.py`、`check_train_duplicates.py`、`clean_dataset.py` — 数据干净度工具。
