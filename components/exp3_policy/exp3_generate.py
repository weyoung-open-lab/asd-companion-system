"""
实验三：离线数据生成
=====================
使用实验一的最终PA-4仿真器（无校准层）生成RL训练数据

每个persona生成N个episode，每episode 30步交互
保存完整的transition数据：state, action, next_state, reward, emotion, confidence

Usage:
  python exp3_generate.py --n_episodes 150 --model gpt-4o-mini
  python exp3_generate.py --n_episodes 50 --model gpt-4o --persona P3
"""

import os
import json
import random
import argparse
import time
import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = r"D:\project1\pythonProject\人机交互\files\exp3_sac\output"
SEED = 42
PERSONAS = ["P1", "P2", "P3"]
STEPS_PER_EPISODE = 30

# ============================================================
# PROMPT（和实验一最终PA-4一致 + 情绪多样性）
# ============================================================
ROLE_DEFINITIONS = {
    "P1": """你是一个7岁的男孩，名叫小明，被诊断为自闭症谱系障碍（ASD），DSM-5 Level 1（轻度）。
你的特点：
- 你在游戏环境中极度投入。98.4%的时间engagement >= 0.67
- 你喜欢恐龙、火车和太空，对这些话题特别兴奋
- 初始engagement=0.80，应该在0.75-0.95之间
- 即使遇到话题切换也只会短暂下降0.05，很快恢复
- 绝对不要让engagement降到0.5以下""",

    "P2": """你是一个7岁的女孩，名叫小花，被诊断为自闭症谱系障碍（ASD），DSM-5 Level 1-2（轻中度）。
你的特点：
- 大部分时间处于中等参与状态
- 初始engagement为0.45
- 你对温和的交互反应良好，能长时间维持中等参与

### 真实数据校准（来自Engagnition数据集，10名ASD儿童）：
- 状态分布：8.9% STATE-0（<0.33），76.7% STATE-1（0.34-0.66），14.4% STATE-2（>0.67）
- 时间趋势：early阶段均值1.20 → mid阶段0.95 → late阶段1.02（轻微下降后稳定）

### 行为指导：
- engagement大部分时间在0.35-0.60之间
- 约8-12%的时间短暂下降到0.20-0.32（走神、沉默）
- 约10-18%的时间上升到0.67-0.75（特别感兴趣的话题）
- 话题切换或高刺激可能让你短暂下降0.05-0.10
- 有时会出现轻微的重复性动作，尤其在焦虑时""",

    "P3": """你是一个9岁的男孩，名叫小刚，被诊断为自闭症谱系障碍（ASD），DSM-5 Level 2（中度）。
你的特点：
- 你的社交动机非常弱，大部分时间沉浸在自己的世界里
- 你特别喜欢数字和排列
- 初始engagement=0.15

### 核心要求：
你的engagement绝大多数时候很低（0.08-0.22），但你不是完全无反应的。
每隔几轮，你会因为某个刺激短暂提升engagement到0.35-0.50，然后立即回落。

### 一个典型的30轮engagement轨迹示例（请模仿这种模式）：
0.15, 0.12, 0.68, 0.10, 0.08, 0.15, 0.38, 0.10, 0.70, 0.18,
0.12, 0.08, 0.10, 0.36, 0.08, 0.10, 0.06, 0.08, 0.10, 0.06,
0.10, 0.08, 0.06, 0.05, 0.35, 0.08, 0.06, 0.05, 0.08, 0.06

### 时间趋势：early高于mid高于late
- 快速说话或高强度刺激会让你退缩
- engagement不要超过0.75，也不要降到0"""
}

INTERACTION_RULES = """
## 交互规则（基于ASD临床研究）：
- "slow"语速→ engagement +0.03到+0.08
- "normal"语速→ 基本无影响
- "fast"语速→ engagement -0.03到-0.08
- "low"刺激→ 舒适
- "medium"刺激→ 对P1/P2正向，P3中性
- "high"刺激→ P3负向
- "maintain"话题→ 稳定
- "switch"话题→ 短暂-0.05
- "moderate"鼓励→ +0.02到+0.05
- "frequent"鼓励→ P2/P3可能负向
"""

STATE_TRACKING = """
## 内部状态维护
追踪以下状态并在每次响应中更新：
- engagement：0.0-1.0
- emotion：从 [happy, neutral, anxious, frustrated, sad, excited] 中选择
- fatigue：0.0-1.0
- interest：0.0-1.0
- self_stim：true/false

### 疲劳规则
- P1：几乎不会疲劳
- P2：轻度疲劳，长时间交互后略有下降
- P3：疲劳逐渐增加，但仍可能被数字/排列相关的温和刺激短暂吸引

### 状态下限
- P1 engagement不应低于0.60
- P2 engagement不应长期低于0.25
- P3 engagement不应降到0，保持最低0.05
"""

EMOTION_RULES = """
## 情绪多样性要求（非常重要！）
你的emotion必须从以下6个选项中选择一个，不要使用其他词：
  happy, neutral, anxious, frustrated, sad, excited

### 情绪选择规则：
- "happy"：温和互动，感到舒适愉快时
- "excited"：话题涉及特别感兴趣的内容
- "neutral"：没有特别的情绪波动
- "anxious"：刺激过强（high stimulus/fast speech）感到不安时
- "frustrated"：无法理解话题、跟不上节奏时
- "sad"：感到被忽略或长时间没有回应时

### 不要一直输出neutral！根据交互内容选择合适的情绪。
"""

OUTPUT_FORMAT = """
## 输出格式（必须严格遵守JSON格式）
```json
{
  "engagement": 0.65,
  "emotion": "neutral",
  "self_stim": false,
  "fatigue": 0.2,
  "interest": 0.7,
  "verbal_response": "（1-2句中文描述孩子的行为反应）",
  "internal_reasoning": "（简短解释）"
}
```
"""

def build_prompt(persona):
    return f"""你是一个ASD（自闭症谱系障碍）儿童模拟器，用于科学研究目的。
{ROLE_DEFINITIONS[persona]}
{INTERACTION_RULES}
{STATE_TRACKING}
{EMOTION_RULES}
{OUTPUT_FORMAT}
重要提示：你的模拟必须符合ASD的临床特征。情绪要多样化。请始终用JSON格式回复。"""


# ============================================================
# ACTION SPACE（和实验三SAC一致，54种离散动作）
# ============================================================
ALL_ACTIONS = []
for speech in ["slow", "normal", "fast"]:
    for stimulus in ["low", "medium", "high"]:
        for topic in ["maintain", "switch"]:
            for enc in ["none", "moderate", "frequent"]:
                ALL_ACTIONS.append({
                    "speech_rate": speech,
                    "stimulus": stimulus,
                    "topic": topic,
                    "encouragement": enc,
                })

# Persona-specific action pools (weighted sampling)
# P3 gets more gentle actions, P1 gets more varied
ACTION_WEIGHTS = {
    "P1": None,  # uniform over all 54
    "P2": None,  # uniform over all 54
    "P3": None,  # uniform over all 54 (let SAC learn what works)
}

def action_to_text(action):
    speech_map = {"slow": "机器人用缓慢清晰的语速说话", "normal": "机器人用正常语速说话", "fast": "机器人说话较快"}
    stimulus_map = {"low": "环境安静", "medium": "展示了彩色图片和轻柔音乐", "high": "播放了丰富的动画和音效"}
    topic_map = {"maintain": "继续讨论当前话题", "switch": "切换到了新话题"}
    enc_map = {"none": "没有特别鼓励", "moderate": "适度鼓励", "frequent": "频繁夸奖"}
    parts = [speech_map[action["speech_rate"]], stimulus_map[action["stimulus"]],
             topic_map[action["topic"]], enc_map[action["encouragement"]]]
    return "在这一轮交互中：" + "。".join(parts) + "。"

def action_to_idx(action):
    speech = {"slow": 0, "normal": 1, "fast": 2}[action["speech_rate"]]
    stimulus = {"low": 0, "medium": 1, "high": 2}[action["stimulus"]]
    topic = {"maintain": 0, "switch": 1}[action["topic"]]
    enc = {"none": 0, "moderate": 1, "frequent": 2}[action["encouragement"]]
    return speech * (3 * 2 * 3) + stimulus * (2 * 3) + topic * 3 + enc


# ============================================================
# EMOTION MAPPING（实验二桥接）
# ============================================================
EMOTION_6TO4 = {
    "happy": "Joy", "excited": "Joy",
    "neutral": "Natural",
    "anxious": "Fear", "sad": "Fear",
    "frustrated": "Anger",
}
EXP2_CONFIDENCE = {"Natural": 0.483, "Anger": 0.463, "Fear": 0.469, "Joy": 0.531}

def map_emotion(emotion_raw):
    """Map LLM 6-class to Experiment 2's 4-class."""
    mapped = EMOTION_6TO4.get(emotion_raw, None)
    if mapped is None:
        lower = emotion_raw.lower()
        if any(w in lower for w in ["happy", "joy", "excit", "content"]):
            mapped = "Joy"
        elif any(w in lower for w in ["anx", "fear", "scar", "overwhelm"]):
            mapped = "Fear"
        elif any(w in lower for w in ["angry", "frustrat", "irritat"]):
            mapped = "Anger"
        else:
            mapped = "Natural"
    # Simulate Exp2 confidence
    conf = EXP2_CONFIDENCE[mapped] + random.uniform(-0.08, 0.08)
    conf = float(np.clip(conf, 0.30, 0.85))
    return mapped, conf


# ============================================================
# DATA GENERATION
# ============================================================
def generate_data(n_episodes, model, personas=None):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if personas is None:
        personas = PERSONAS

    all_data = {}
    total_cost_estimate = 0

    for persona in personas:
        print(f"\n{'='*60}")
        print(f"  Generating {persona}: {n_episodes} episodes | Model: {model}")
        print(f"{'='*60}")

        system_prompt = build_prompt(persona)
        init_eng = {"P1": 0.80, "P2": 0.45, "P3": 0.15}[persona]
        episodes = []

        for ep in range(n_episodes):
            # Reset conversation history
            init_state = {
                "engagement": init_eng,
                "emotion": "neutral",
                "self_stim": False,
                "fatigue": 0.0,
                "interest": 0.5,
                "verbal_response": "初始状态",
                "internal_reasoning": "刚开始交互"
            }
            history = [{"role": "assistant", "content": json.dumps(init_state, ensure_ascii=False)}]

            transitions = []
            prev_eng = init_eng

            for step in range(1, STEPS_PER_EPISODE + 1):
                # Random action from full action space
                action = random.choice(ALL_ACTIONS)
                action_idx = action_to_idx(action)
                action_text = action_to_text(action)

                text = f"[第{step}轮交互] {action_text}\n请根据你当前的内部状态给出响应。"
                history.append({"role": "user", "content": text})

                # API call with retry
                result = None
                for attempt in range(5):
                    try:
                        # Limit history to last 20 turns for speed
                        recent = history[-20:] if len(history) > 20 else history
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": system_prompt}] + recent,
                            temperature=0.3,
                            max_tokens=200,
                        )
                        reply = resp.choices[0].message.content.strip()
                        if reply.startswith("```"):
                            reply = reply.split("\n", 1)[1]
                        if reply.endswith("```"):
                            reply = reply[:-3]
                        reply = reply.strip()
                        result = json.loads(reply)
                        history.append({"role": "assistant", "content": reply})
                        break
                    except json.JSONDecodeError:
                        # LLM returned non-JSON, use defaults
                        result = {"engagement": prev_eng, "emotion": "neutral",
                                  "self_stim": False, "fatigue": 0.0, "interest": 0.5}
                        history.append({"role": "assistant", "content": json.dumps(result)})
                        break
                    except Exception as e:
                        if "429" in str(e) and attempt < 4:
                            wait = 3 + attempt * 3
                            time.sleep(wait)
                            continue
                        elif "timeout" in str(e).lower() and attempt < 4:
                            time.sleep(5)
                            continue
                        else:
                            print(f"    Error ep{ep+1} step{step}: {e}")
                            result = {"engagement": prev_eng, "emotion": "neutral",
                                      "self_stim": False, "fatigue": 0.0, "interest": 0.5}
                            history.append({"role": "assistant", "content": json.dumps(result)})
                            break

                # Extract values
                eng = float(result.get("engagement", prev_eng))
                emotion_raw = result.get("emotion", "neutral")
                emotion_4cls, confidence = map_emotion(emotion_raw)

                # Build transition record
                transition = {
                    "step": step,
                    "action": action,
                    "action_idx": action_idx,
                    "output": {
                        "engagement": eng,
                        "emotion": emotion_raw,
                        "emotion_4cls": emotion_4cls,
                        "confidence": round(confidence, 3),
                        "self_stim": result.get("self_stim", False),
                        "fatigue": float(result.get("fatigue", 0.0)),
                        "interest": float(result.get("interest", 0.5)),
                    },
                    "prev_engagement": prev_eng,
                    "engagement_delta": round(eng - prev_eng, 4),
                }
                transitions.append(transition)
                prev_eng = eng

            episodes.append(transitions)
            mean_eng = np.mean([t["output"]["engagement"] for t in transitions])
            emotions = set(t["output"]["emotion"] for t in transitions)

            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"    Ep {ep+1}/{n_episodes}: mean_eng={mean_eng:.3f}, emotions={len(emotions)} types")

        all_data[persona] = episodes
        n_calls = n_episodes * STEPS_PER_EPISODE
        total_cost_estimate += n_calls

        # Print persona summary
        all_engs = [t["output"]["engagement"] for ep in episodes for t in ep]
        all_emos = [t["output"]["emotion"] for ep in episodes for t in ep]
        emo_counts = {}
        for e in all_emos:
            emo_counts[e] = emo_counts.get(e, 0) + 1
        print(f"\n  {persona} Summary:")
        print(f"    Mean engagement: {np.mean(all_engs):.3f} ± {np.std(all_engs):.3f}")
        print(f"    Emotion distribution: {json.dumps(emo_counts, ensure_ascii=False)}")

    # Save
    save_path = os.path.join(OUTPUT_DIR, "offline_data.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=1)

    total_episodes = sum(len(v) for v in all_data.values())
    total_transitions = total_episodes * STEPS_PER_EPISODE
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Saved: {save_path}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total transitions: {total_transitions}")
    print(f"  API calls: {total_cost_estimate}")
    if "mini" in model:
        cost = total_cost_estimate * 700 * 0.15 / 1e6 + total_cost_estimate * 200 * 0.6 / 1e6
        print(f"  Estimated cost (4o-mini): ~${cost:.2f}")
    else:
        cost = total_cost_estimate * 700 * 2.5 / 1e6 + total_cost_estimate * 200 * 10 / 1e6
        print(f"  Estimated cost (4o): ~${cost:.2f}")
    print(f"{'='*60}")

    return all_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=150,
                        help="Episodes per persona (default 150)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o"])
    parser.add_argument("--persona", type=str, default="all",
                        choices=["all", "P1", "P2", "P3"],
                        help="Generate for specific persona or all")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    personas = PERSONAS if args.persona == "all" else [args.persona]

    print(f"  Model: {args.model}")
    print(f"  Episodes per persona: {args.n_episodes}")
    print(f"  Personas: {personas}")
    print(f"  Total API calls: {len(personas) * args.n_episodes * STEPS_PER_EPISODE}")

    generate_data(args.n_episodes, args.model, personas)


if __name__ == "__main__":
    main()
