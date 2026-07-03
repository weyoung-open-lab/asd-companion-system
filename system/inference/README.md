# 感知接口（Perception）— 可替换槽位的参考实现

把 Exp2 部署 checkpoint（`models/ferac_efficientnetb0_deploy.pth`，含 backbone + 温度T + 阈值）
封装成**有明确契约**的感知模块。系统"四个可替换接口"中的**感知槽位参考实现**。

> 核心定位：**abstain（诚实弃权）是一等输出，不是异常分支。** 低置信 / 无脸 / 图糊 / 多脸 / 内部错误
> 一律返回结构化 abstain，绝不抛异常让调用方崩溃。诚实优先于覆盖。

## 文件
- `contract.py` —— `PerceptionResult` / `PerceptionConfig` / `Decision` / `AbstainReason`（契约在代码里显式化）。
- `perception.py` —— `Perception` 类（`__init__` 加载 checkpoint，`predict()` 实现契约）。
- `README.md` —— 本文件。

## 用法
```python
from system.inference.perception import Perception
from system.inference.contract import PerceptionConfig

P = Perception()                      # 默认加载 models/ferac_efficientnetb0_deploy.pth
r = P.predict("face.jpg")             # 接受 路径(str/Path) / numpy(HxWx3 RGB uint8) / PIL.Image

if r.accepted:                        # r.decision == "accept"
    use(r.predicted_label, r.confidence, r.emotion_probs)
else:                                 # r.decision == "abstain" -> 下游走保守策略
    log(r.abstain_reason)             # low_confidence / no_face_detected / low_quality / multiple_faces / inference_error
```
自检：`python system/inference/perception.py`（噪声→no_face、全黑→low_quality、FERAC脸→accept/abstain各一例）。

## 输出契约（永远有 `decision`）
| 字段 | accept | abstain |
|---|---|---|
| `decision` | `"accept"` | `"abstain"` |
| `predicted_label` | 硬标签 | **null**（防误用） |
| `emotion_probs` | {natural,anger,fear,joy} | 仍返回（供日志/可解释，标不可信） |
| `confidence` | 校准后 max-softmax | 同（< 阈值） |
| `raw_confidence` | 未校准 max-softmax（仅调试） | 同 |
| `uncertainty` | {entropy, margin} | 同 |
| `abstain_reason` | null | 见枚举 |
| `meta` | threshold/temperature/face_detected/n_faces/blur_var/checkpoint | 同 |

**置信度是校准过的**：`confidence = softmax(logits / T).max()`，T 从 checkpoint 读（本部署模型 T=0.377）。
原始 softmax 因 Focal+LabelSmoothing 系统性欠自信（原始 ECE 0.29），**不可直接门控**，故必须用校准值。

## 配置（`PerceptionConfig`，可覆盖，不写死）
| 字段 | 默认 | 说明 |
|---|---|---|
| `threshold` | 从 checkpoint（=**0.86**） | 校准后置信度门控阈值 |
| `temperature` | 从 checkpoint（=**0.377**） | 温度缩放，一般不改 |
| `uncertainty_metric` | `"confidence"` | `confidence`(max-softmax) / `entropy` / `margin` |
| `blur_threshold` | 25.0 | Laplacian 方差，低于则 low_quality |
| `return_probs_on_abstain` | True | abstain 时是否仍给概率 |
| `face_required` | True | 检测不到脸 → abstain(no_face) |
| `face_margin` | 0.20 | 人脸框外扩比例后裁剪 |
| `multiple_faces_abstain` | True | 多脸 → abstain(multiple_faces) |

> **阈值/温度纪律**：来自 Exp2 **验证集**选定的工作点（部署 checkpoint 在 val_deploy 上 fit T、选 0.86）。
> **不要用 FERAC 155 测试集重选阈值。** 部署方要调松紧需说明依据。
> 注：契约文档示例里的 0.73/0.383 是 Exp2 **CV/OOF 阶段**的工作点；本部署 checkpoint 是另一次训练（train615），
> 自带 T=0.377、阈值=0.86，以 checkpoint 为准。

## 关于人脸检测器（重要）
当前用 **OpenCV Haar 级联**（asd_env 无 dlib/MTCNN，mediapipe 此 build 无 solutions）。Haar 是**基线检测器**：
- 在紧裁脸（如 FERAC）上可能误检出第2张脸 → 触发 `multiple_faces`，或漏检 → `no_face`。
- **生产建议换更强检测器**（MTCNN / RetinaFace / mediapipe Tasks），只需改 `Perception._detect_faces`，契约不变。
- **输入已是裁好的脸**时：传 `PerceptionConfig(face_required=False)`（必要时 `multiple_faces_abstain=False`），跳过检测当整脸用。

## 如何替换成你自己的感知模块（兑现"可替换槽位"）
任何第三方感知器（别的识别器、骨架/眼动感知…）**只要实现**：

```python
def predict(self, image, config=None) -> PerceptionResult: ...
```

返回符合 `contract.py` 的 `PerceptionResult`（`decision` 必有、abstain 时 `predicted_label=None`、错误降级为 abstain），
即可**零改动**替换本模块。下游（RL 策略 / 安全反思层）只依赖契约，不依赖内部模型。

## 与下游对接（让 abstain 真正起作用）
- **RL 策略**：观测里的情绪4维+置信度维直接取 `emotion_probs` 和 `confidence`；`decision==abstain` 时，
  置信度信号告知策略走保守动作（对应 Exp3 奖励里的"置信度安全项"）。
- **安全反思层**：`decision==abstain` 作为触发保守/安全模式的信号之一。
- 本接口只"感知 + 诚实弃权"，**不替下游做决策**——只如实报告"看到什么、多大把握、要不要信我"。
