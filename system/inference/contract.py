# -*- coding: utf-8 -*-
"""
感知接口契约 —— 显式数据结构
================================
让"契约"在代码里也是一等公民：任何第三方感知模块只要 predict(image)->PerceptionResult，
即可替换本模块（平台"可替换接口"主张的落点）。

核心：abstain（诚实弃权）是正常输出，不是异常分支。
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


class Decision:
    ACCEPT = "accept"
    ABSTAIN = "abstain"


class AbstainReason:
    LOW_CONFIDENCE = "low_confidence"       # 校准后置信度 < 阈值
    NO_FACE = "no_face_detected"            # 检测不到人脸
    LOW_QUALITY = "low_quality"             # 过糊/过暗（Laplacian 模糊检测）
    MULTIPLE_FACES = "multiple_faces"       # 多张脸，无法确定目标
    INFERENCE_ERROR = "inference_error"     # 内部异常，已捕获降级
    ALL = {LOW_CONFIDENCE, NO_FACE, LOW_QUALITY, MULTIPLE_FACES, INFERENCE_ERROR}


@dataclass
class PerceptionConfig:
    """配置契约（可调，不写死）。threshold/temperature 默认 None => 从 checkpoint 读取。"""
    threshold: Optional[float] = None        # 校准后置信度门控阈值（Exp2 验证工作点）
    temperature: Optional[float] = None      # 温度缩放（一般从 checkpoint 读，不改）
    uncertainty_metric: str = "confidence"   # "confidence" | "entropy" | "margin"
    blur_threshold: float = 25.0             # Laplacian 方差阈值，低于则 low_quality
    return_probs_on_abstain: bool = True     # abstain 时是否仍返回概率（供日志/可解释）
    face_required: bool = True               # True：检测不到脸 => abstain(no_face)
    face_margin: float = 0.20                # 人脸框外扩比例后再裁剪
    multiple_faces_abstain: bool = True      # 多脸 => abstain(multiple_faces)

    def merged(self, override: "Optional[PerceptionConfig]") -> "PerceptionConfig":
        if override is None:
            return self
        out = PerceptionConfig(**asdict(self))
        for k, v in asdict(override).items():
            # 只覆盖显式非默认项（None 视为"不覆盖 threshold/temperature"以外的字段保持）
            if v is not None:
                setattr(out, k, v)
        return out


@dataclass
class PerceptionResult:
    """输出契约：永远有 decision 字段（accept / abstain）。"""
    decision: str                                   # "accept" | "abstain"
    emotion_probs: Optional[Dict[str, float]]       # {natural,anger,fear,joy}；abstain 也给（标不可信）
    predicted_label: Optional[str]                  # accept 才给硬标签；abstain 置 null
    confidence: Optional[float]                     # 校准后 max-softmax
    raw_confidence: Optional[float]                 # 未校准 max-softmax（仅透明/调试）
    uncertainty: Dict[str, float]                   # {entropy, margin}
    abstain_reason: Optional[str]                   # AbstainReason 之一；accept 时 None
    meta: Dict[str, Any] = field(default_factory=dict)  # threshold/temperature/face_detected/...

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    @property
    def accepted(self) -> bool:
        return self.decision == Decision.ACCEPT
