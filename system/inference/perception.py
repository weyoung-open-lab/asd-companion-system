# -*- coding: utf-8 -*-
"""
感知模块参考实现（Perception）
==================================
把 Exp2 部署 checkpoint（models/ferac_efficientnetb0_deploy.pth，含 backbone + 温度T + 阈值）
封装成有明确契约的感知模块。abstain 是一等输出，任何内部错误都降级为 abstain，绝不抛异常。

用法：
    from system.inference.perception import Perception
    P = Perception()                       # 默认加载部署 checkpoint
    r = P.predict("face.jpg")              # 接受 路径 / numpy(HxWx3 RGB uint8) / PIL.Image
    if r.accepted:
        print(r.predicted_label, r.confidence)
    else:
        print("abstain:", r.abstain_reason)   # 走保守策略

替换说明：换任何识别器只要实现 predict(image)->PerceptionResult 即可，下游零改动。
"""
import os, sys, math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.paths import MODELS_DIR
from system.inference.contract import (
    PerceptionConfig, PerceptionResult, Decision, AbstainReason)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = MODELS_DIR / "ferac_efficientnetb0_deploy.pth"
# 契约默认（仅当 checkpoint 缺失时兜底；正常以 checkpoint 内的 T/阈值为准）
FALLBACK_THRESHOLD = 0.73
FALLBACK_TEMPERATURE = 0.383


class Perception:
    def __init__(self, checkpoint_path: Union[str, Path, None] = None,
                 config: Optional[PerceptionConfig] = None):
        self.ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CKPT
        ck = torch.load(self.ckpt_path, map_location="cpu")
        self.classes = ck.get("classes", ["Natural", "anger", "fear", "joy"])
        self.classes_out = [c.lower() for c in self.classes]   # 契约输出用小写
        self.temperature = float(ck.get("temperature", FALLBACK_TEMPERATURE))
        self.ckpt_threshold = float(ck.get("gate_threshold", FALLBACK_THRESHOLD))
        pre = ck.get("preprocess", {"resize": 224,
                                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

        # 模型
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, len(self.classes)))
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(DEVICE).eval()

        # 预处理（与训练一致）
        self.tf = transforms.Compose([
            transforms.Resize((pre["resize"], pre["resize"])),
            transforms.ToTensor(),
            transforms.Normalize(pre["mean"], pre["std"]),
        ])

        # 配置：threshold/temperature 优先用 checkpoint 值
        base = config or PerceptionConfig()
        if base.threshold is None:
            base.threshold = self.ckpt_threshold
        if base.temperature is None:
            base.temperature = self.temperature
        self.config = base

        # 人脸检测器（OpenCV Haar，可插拔；换更强检测器只改 _detect_faces）
        self._cascade = None
        try:
            import cv2
            self._cv2 = cv2
            hc = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(hc)
            if cascade.empty():
                # 中文/unicode 路径下 cv2 FileStorage 无法直接打开 -> 用内存加载绕过
                xml = Path(hc).read_text(encoding="utf-8")
                fs = cv2.FileStorage(xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
                cascade = cv2.CascadeClassifier()
                cascade.read(fs.getFirstTopLevelNode())
            self._cascade = cascade if not cascade.empty() else None
        except Exception:
            self._cv2 = None
            self._cascade = None

    # ---------- 输入处理 ----------
    def _load_image(self, image) -> np.ndarray:
        """统一成 RGB uint8 numpy（HxWx3）。路径用 PIL（unicode 安全，避免 cv2.imread 中文路径问题）。"""
        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        raise TypeError(f"unsupported image type: {type(image)}")

    def _blur_var(self, rgb: np.ndarray) -> float:
        """Laplacian 方差：越低越糊。无 cv2 则返回 +inf（不触发 low_quality）。"""
        if self._cv2 is None:
            return float("inf")
        gray = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2GRAY)
        return float(self._cv2.Laplacian(gray, self._cv2.CV_64F).var())

    def _detect_faces(self, rgb: np.ndarray):
        """返回人脸框列表 [(x,y,w,h), ...]。无检测器则返回 None（表示"未知"，不强制 no_face）。"""
        if self._cascade is None:
            return None
        gray = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(48, 48))
        return [tuple(int(v) for v in f) for f in faces]

    def _crop_face(self, rgb: np.ndarray, box, margin: float) -> np.ndarray:
        x, y, w, h = box
        H, W = rgb.shape[:2]
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        crop = rgb[y0:y1, x0:x1]
        return crop if crop.size else rgb

    # ---------- 推理 ----------
    def _infer(self, rgb_crop: np.ndarray, temperature: float):
        x = self.tf(Image.fromarray(rgb_crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)[0]
            raw = F.softmax(logits, 0).cpu().numpy()
            cal = F.softmax(logits / temperature, 0).cpu().numpy()
        return cal.astype(float), raw.astype(float)

    @staticmethod
    def _entropy(p):
        return float(-(p * np.log(p + 1e-12)).sum())

    @staticmethod
    def _margin(p):
        s = np.sort(p)
        return float(s[-1] - s[-2])

    def _gate_score(self, cfg, conf, ent, margin):
        if cfg.uncertainty_metric == "margin":
            return margin
        if cfg.uncertainty_metric == "entropy":
            return 1.0 - ent / math.log(len(self.classes))   # 归一化置信度
        return conf  # "confidence"（默认，= 校准后 max-softmax）

    def _result(self, decision, reason, probs_cal, raw_conf, ent, margin, pred_idx,
                cfg, face_detected, n_faces, blur):
        emotion_probs = ({c: float(round(p, 4)) for c, p in zip(self.classes_out, probs_cal)}
                         if (probs_cal is not None and
                             (decision == Decision.ACCEPT or cfg.return_probs_on_abstain))
                         else None)
        return PerceptionResult(
            decision=decision,
            emotion_probs=emotion_probs,
            predicted_label=(self.classes_out[pred_idx] if decision == Decision.ACCEPT else None),
            confidence=(float(round(probs_cal.max(), 4)) if probs_cal is not None else None),
            raw_confidence=(float(round(raw_conf, 4)) if raw_conf is not None else None),
            uncertainty={"entropy": round(ent, 4), "margin": round(margin, 4)},
            abstain_reason=reason,
            meta={"threshold": round(cfg.threshold, 4), "temperature": round(cfg.temperature, 4),
                  "gate_metric": cfg.uncertainty_metric, "face_detected": face_detected,
                  "n_faces": n_faces, "blur_var": (round(blur, 1) if blur != float("inf") else None),
                  "checkpoint": self.ckpt_path.name},
        )

    # ---------- 契约入口 ----------
    def predict(self, image, config: Optional[PerceptionConfig] = None) -> PerceptionResult:
        cfg = self.config.merged(config)
        # 顶层兜底：任何异常 -> abstain(inference_error)
        try:
            try:
                rgb = self._load_image(image)
            except Exception:
                return self._result(Decision.ABSTAIN, AbstainReason.INFERENCE_ERROR,
                                    None, None, 0.0, 0.0, 0, cfg, False, 0, float("inf"))

            blur = self._blur_var(rgb)
            faces = self._detect_faces(rgb)
            n_faces = 0 if faces is None else len(faces)
            face_detected = n_faces >= 1

            # 选裁剪区域：检到脸用最大框，否则用整图（best-effort 仍出概率）
            if face_detected:
                box = max(faces, key=lambda b: b[2] * b[3])
                crop = self._crop_face(rgb, box, cfg.face_margin)
            else:
                crop = rgb

            probs_cal, probs_raw = self._infer(crop, cfg.temperature)
            conf = float(probs_cal.max()); raw_conf = float(probs_raw.max())
            ent = self._entropy(probs_cal); margin = self._margin(probs_cal)
            pred_idx = int(probs_cal.argmax())

            # abstain 原因判定（优先级：无脸 > 低质量 > 多脸 > 低置信）
            reason = None
            if cfg.face_required and faces is not None and n_faces == 0:
                reason = AbstainReason.NO_FACE
            elif blur < cfg.blur_threshold:
                reason = AbstainReason.LOW_QUALITY
            elif cfg.multiple_faces_abstain and n_faces > 1:
                reason = AbstainReason.MULTIPLE_FACES
            elif self._gate_score(cfg, conf, ent, margin) < cfg.threshold:
                reason = AbstainReason.LOW_CONFIDENCE

            decision = Decision.ACCEPT if reason is None else Decision.ABSTAIN
            return self._result(decision, reason, probs_cal, raw_conf, ent, margin,
                                pred_idx, cfg, face_detected, n_faces, blur)
        except Exception:
            return self._result(Decision.ABSTAIN, AbstainReason.INFERENCE_ERROR,
                                None, None, 0.0, 0.0, 0, cfg, False, 0, float("inf"))


if __name__ == "__main__":
    # 最小自检：纯噪声/无脸 -> abstain；FERAC 脸 -> 高置信 accept / 低置信 abstain
    import glob
    P = Perception()
    print(f"  loaded: {P.ckpt_path.name} | T={round(P.temperature,4)} | ckpt_threshold={P.ckpt_threshold}")
    print(f"  classes: {P.classes_out} | face detector: {'Haar OK' if P._cascade else 'NONE'}")

    # 1) 纯噪声 -> 期望 abstain（检测器在则 no_face_detected）
    noise = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    r1 = P.predict(noise)
    print("\n  [noise]        decision=%-7s reason=%-18s face=%s" % (
        r1.decision, r1.abstain_reason, r1.meta["face_detected"]))

    # 2) 全黑（low_quality）
    black = np.zeros((224, 224, 3), np.uint8)
    r0 = P.predict(black, PerceptionConfig(face_required=False))
    print("  [black]        decision=%-7s reason=%-18s blur=%s" % (
        r0.decision, r0.abstain_reason, r0.meta["blur_var"]))

    # 3) 遍历 FERAC 测试图（pre-cropped，face_required=False 当整脸用）：统计 accept/abstain
    cfg = PerceptionConfig(face_required=False)
    files = []
    for c in ["joy", "Natural", "anger", "fear"]:
        files += sorted(glob.glob(str(MODELS_DIR.parent / "data" / "ferac" / "test" / c / "*")))[:8]
    n_acc = n_abs = 0
    one_accept = one_abstain = None
    for f in files:
        r = P.predict(f, cfg)
        if r.accepted:
            n_acc += 1; one_accept = one_accept or r
        else:
            n_abs += 1; one_abstain = one_abstain or r
    print(f"\n  [FERAC {len(files)} imgs] accept={n_acc}  abstain={n_abs}  (阈值{P.config.threshold} 高=偏保守)")
    if one_accept:
        print("\n  --- 一个 ACCEPT 示例 ---"); print(one_accept.to_json(indent=2))
    if one_abstain:
        print("\n  --- 一个 ABSTAIN 示例 ---"); print(one_abstain.to_json(indent=2))
