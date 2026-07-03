"""
集中式路径常量（杜绝硬编码绝对路径）。
============================================
所有组件脚本的数据/模型/产物路径都应改为从这里导入，
而不是写死 D:\\project1\\... 之类的绝对路径。

用法：
    from common.paths import FERAC_ROOT, MODELS_DIR, EXP3_OUTPUT
    # 或在 components/ 下的脚本里：
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from common.paths import FERAC_ROOT

注意：本文件目前仅【定义常量】，尚未接入任何组件脚本。
脚本的路径替换是下一步单独执行（见对照清单 PATH_REWRITE）。
"""
from pathlib import Path

# 项目根 = 本文件(common/paths.py) 的上两级目录 = asd-companion-system/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- 顶层目录 ----
COMMON_DIR      = PROJECT_ROOT / "common"
SYSTEM_DIR      = PROJECT_ROOT / "system"
COMPONENTS_DIR  = PROJECT_ROOT / "components"
MODELS_DIR      = PROJECT_ROOT / "models"
DATA_DIR        = PROJECT_ROOT / "data"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
DOCS_DIR        = PROJECT_ROOT / "docs"

# ---- 数据集 ----
ENGAGNITION_ROOT = DATA_DIR / "engagnition"          # Exp1 校准数据
FERAC_ROOT       = DATA_DIR / "ferac"                # Exp2 图片数据 = 论文用的干净 FERAC 770（4类 Natural/anger/fear/joy，train615/test155，来自 archive(5).zip）
# AffectNet 原始数据不在本仓库（旧仓库指向 D:\BaiduNetdiskDownload\AffectNet_Processed）。
# 如需复现 Exp2 预训练，请单独提供并在此处指定：
AFFECTNET_ROOT   = DATA_DIR / "affectnet_processed"  # 占位：当前不存在

# ---- 模型权重 ----
EXP2_EFFB0_CKPT      = MODELS_DIR / "B2_EfficientNetB0_best.pth"      # Exp2 部署权重
DLIB_LANDMARK_MODEL  = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
SURROGATE_CKPT = {                                                    # Exp3 per-Persona surrogate
    "P1": MODELS_DIR / "surrogate_P1.pt",
    "P2": MODELS_DIR / "surrogate_P2.pt",
    "P3": MODELS_DIR / "surrogate_P3.pt",
}

# ---- 各组件输出目录（产物落盘位置；按需 mkdir）----
EXP1_OUTPUT = ARTIFACTS_DIR / "exp1"
EXP2_OUTPUT = ARTIFACTS_DIR / "exp2"
EXP3_OUTPUT = ARTIFACTS_DIR / "exp3"

# Exp1 校准文件
CALIBRATION_FILE = ARTIFACTS_DIR / "exp1" / "calibration_final.json"
# Exp3 离线数据
OFFLINE_DATA = ARTIFACTS_DIR / "exp3" / "offline_data.json"


def ensure_dirs():
    """需要时创建输出目录。"""
    for d in (EXP1_OUTPUT, EXP2_OUTPUT, EXP3_OUTPUT):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("PROJECT_ROOT =", PROJECT_ROOT)
    for name in ("ENGAGNITION_ROOT", "FERAC_ROOT", "MODELS_DIR", "EXP3_OUTPUT"):
        v = globals()[name]
        print(f"  {name:18} = {v}   exists={v.exists()}")
