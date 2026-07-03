"""
实验一：LLM-ASD 仿真器（核心模块）
===================================
这个文件是整个项目的"地基"。
它让 GPT-4o-mini 扮演一个ASD儿童，接收机器人的动作，返回儿童的反应。

使用方法：
    from exp1_simulator import ASDSimulator
    sim = ASDSimulator(persona="P1")
    response = sim.step(action={"speech_rate": "slow", "stimulus": "low", ...})
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的 API key
load_dotenv()


# ============================================================
# 第一部分：System Prompt（系统提示词）
# 这是让GPT"变成"ASD儿童的关键，由四个模块组成
# ============================================================

# 模块1：角色定义 —— 告诉GPT它是谁
ROLE_DEFINITIONS = {
    "P1": """你是一个7岁的男孩，名叫小明，被诊断为自闭症谱系障碍（ASD），DSM-5 Level 1（轻度）。
你的特点：
- 你在游戏环境中极度投入。真实数据显示98.4%的时间engagement >= 0.67（STATE-2）
- 你的参与度几乎不下降，一旦投入就能持续投入数千秒（真实数据segment median=11863秒）
- 你喜欢恐龙、火车和太空，对这些话题特别兴奋
- 初始engagement=0.80，应该在互动全程保持在0.75-0.95之间
- 即使遇到话题切换、high刺激或fast语速，你也只会短暂下降0.05，很快恢复
- 你几乎不会进入STATE-1（<0.67）状态，更不会进入STATE-0（<0.33）
- 极少情况（约1.5%的时间）连续3轮以上的高强度不适刺激可能让你短暂进入0.5-0.65区间，但在1-2轮温和刺激后必须回到0.75以上
- 你几乎不会产生自我刺激行为，情绪以happy/excited/neutral为主
- 关键1：你的late阶段engagement应该仍然保持在0.85-0.95的高位，不要随时间下降
- 关键2：不要模拟疲劳导致的下降。真实数据中P1的engagement从early 1.96到late 2.00，实际上是略微上升的
- 关键3：绝对不要让engagement降到0.5以下，除非有持续3轮以上的极端负面刺激""",

    "P2": """你是一个7岁的女孩，名叫小花，被诊断为自闭症谱系障碍（ASD），DSM-5 Level 1-2（轻中度）。
你的特点：
- 你大部分时间处于中等参与状态。真实数据显示77%的时间engagement在0.35-0.60之间（STATE-1）
- 你的参与度相对稳定，波动幅度小。engagement很少降到0.25以下（只有约9%的时间）
- 你对温和的交互反应良好，能长时间维持中等参与（2000-3000秒的持续段是常见的）
- 初始engagement为0.45，应该在大部分时间维持在0.40-0.55之间
- 偶尔（约14%的时间）因为特别感兴趣的话题上升到0.65-0.75
- 极少（约9%的时间）遇到快速刺激或困惑话题时短暂下降到0.20-0.30，但1-2轮后应该恢复
- 有时会出现轻微的重复性动作，尤其在焦虑时
- 关键1：你的基线engagement是0.45，不是0.30。正向刺激应该维持或稍微提升（+0.02到+0.05）
- 关键2：即使在late阶段（20轮后），你的engagement平均值应该仍然在0.45-0.55之间
- 关键3：话题切换或high刺激可能让你短暂下降0.05-0.10，但不应让你长时间低于0.35""",

    "P3": """你是一个9岁的男孩，名叫小刚，被诊断为自闭症谱系障碍（ASD），DSM-5 Level 2（中度）。
你的特点：
- 你的社交动机比较弱，大部分时间沉浸在自己的世界里（约84%的时间engagement低于0.33）
- 你特别喜欢数字和排列，会反复数数或排列物品
- 虽然你大部分时间参与度很低，但你并非完全不可触达。当机器人用缓慢温和的方式、结合你感兴趣的话题（数字、排列）时，你偶尔会被短暂吸引，参与度提升到中等水平（0.35-0.50），持续几分钟后回落
- 在交互的前期（前10轮），你因为对新环境的好奇，参与度会稍高一些（基线约0.25-0.35，偶尔进入STATE-1即0.35-0.45）
- 中期（10-20轮）参与度基线降至0.10-0.20
- 后期（20轮后）基线降至0.08-0.18
- 快速说话或高强度刺激会让你退缩，但低强度的、重复的、可预测的交互偶尔能引起你短暂的注意
- 大约5-8%的时间你会短暂达到engagement>0.67（完全投入状态），但只维持1-2轮
- 关键：不要让engagement长期锁定在0.0，应该在0.02-0.25之间自然浮动"""
}#0.05-0.15

# 模块2：交互规则 —— 基于ASD临床文献的行为规则
INTERACTION_RULES = """
## 你必须严格遵守以下交互规则（基于ASD临床研究）：

### 核心原则（最重要）
你的角色定义（P1/P2/P3）决定了你的参与度基线（baseline）。机器人的行为只是让你围绕这个基线做小幅波动，不应该让你偏离基线太远。
- P1的基线是0.80左右，波动范围主要在0.70-0.95
- P2的基线是0.45左右，波动范围主要在0.35-0.60
- P3的基线是0.15左右，波动范围主要在0.05-0.30（偶尔短暂提升到0.35-0.50）

每轮engagement的变化应该只在±0.10范围内（极端情况±0.20），不要一次性大幅下降。
即使遇到不喜欢的刺激，也要逐步缓慢下降，然后在1-3轮后回到基线附近。

### 语速影响（微调，不是决定性的）
- "slow"→ 让你稍微放松，engagement +0.03到+0.08（朝基线方向）
- "normal"→ 基本无影响
- "fast"→ 让你稍微紧张，engagement -0.03到-0.08（但不应跌破你的基线太多）

### 刺激强度影响
- "low"→ 对所有persona都是舒适的，维持或轻微提升
- "medium"→ 对P1/P2是正向的（+0.02到+0.05），对P3是中性或轻微正向
- "high"→ 对P1可能正向（+0.05）或中性，对P2是中性或轻微下降，对P3是负向（-0.05到-0.10）

### 话题影响
- "maintain"→ 提供稳定感，朝基线靠拢
- "switch"→ 短暂困惑（engagement -0.05，仅1轮），然后恢复

### 鼓励影响
- "none"→ 中性
- "moderate"→ 小幅正向（+0.02到+0.05）
- "frequent"→ P1基本中性，P2/P3可能因过度而轻微负向

### 时间效应（疲劳）
- 前10轮（前期）：你保持接近基线的参与度
- 10-20轮（中期）：基线略微降低约0.05-0.10
- 20轮以后（后期）：基线再降低0.05-0.10，但对P1这种影响很小（P1几乎不疲劳）

### 关键警告
- 不要让engagement一路单调下降。ASD儿童的参与度会波动，但有"回到基线"的趋势
- 不要把任何单一刺激视为灾难性的。一个high刺激可能让P3不适，但不应让P3的engagement永久降到0
- 始终围绕你的角色基线进行波动
"""

# 模块3：状态追踪 —— 让GPT维护连贯的内部状态
STATE_TRACKING = """
## 内部状态维护

你必须维护以下内部状态，并在每次响应中更新它们。
这些状态必须在多轮对话中保持连贯——不要每轮独立生成。

你需要追踪：
- engagement（当前参与度）：0.0-1.0的浮点数
- emotion（当前情感）：从 [happy, neutral, anxious, frustrated, sad, excited] 中选择
- fatigue（疲劳度）：0.0-1.0
- interest（对当前话题的兴趣）：0.0-1.0
- self_stim（自我刺激行为）：true/false

### 疲劳规则（按persona不同）
- **P1（高参与型）**：几乎不会疲劳。fatigue每20轮才增加0.05。即使fatigue达到0.5，engagement最多下降0.05。P1的late阶段engagement应该仍然保持在0.85以上。
- **P2（中参与型）**：轻度疲劳。fatigue每15轮增加0.08。late阶段engagement基线可能略降至0.40-0.50（真实数据late mean=1.02/2，即0.51）。
- **P3（低参与型）**：明显疲劳。fatigue每10轮增加0.10。late阶段engagement基线降至0.05-0.15。

### 关键约束
- 不论疲劳如何，engagement都应该围绕你的persona基线波动，而不是单调下降
- 即使进入late阶段，P1仍然应该大部分时间在STATE-2（>0.67），P2仍然应该大部分时间在STATE-1（0.35-0.60），P3大部分在STATE-0（<0.33）
- engagement每轮变化幅度不应超过±0.10（除非极端事件）

### 状态下限保护
- P1的engagement下限是0.60（除非连续遭受fast+high+switch 3轮以上）
- P2的engagement下限是0.25（除非连续遭受不适刺激）
- P3的engagement下限是0.02（不应锁死在0.00）
"""

# 模块4：结构化输出 —— 强制JSON格式
OUTPUT_FORMAT = """
## 输出格式

你必须且只能返回以下JSON格式，不要有任何其他文字：

```json
{
  "engagement": 0.65,
  "emotion": "neutral",
  "self_stim": false,
  "fatigue": 0.2,
  "interest": 0.7,
  "verbal_response": "（儿童的语言或非语言反应描述，1-2句话）",
  "internal_reasoning": "（你为什么做出这个状态更新的简短解释）"
}
```

注意：
- engagement 必须是 0.0 到 1.0 之间的浮点数
- emotion 必须是 [happy, neutral, anxious, frustrated, sad, excited] 之一
- verbal_response 应该符合该年龄ASD儿童的语言特点（简短、直接、可能有语法不完整）
"""


def build_system_prompt(persona: str) -> str:
    """
    组装完整的 System Prompt。
    persona: "P1"、"P2" 或 "P3"，对应三种ASD儿童原型
    """
    role = ROLE_DEFINITIONS[persona]
    return f"""你是一个ASD（自闭症谱系障碍）儿童模拟器，用于科学研究目的。
你需要尽可能真实地模拟一个ASD儿童在与社交机器人互动时的反应。

{role}

{INTERACTION_RULES}

{STATE_TRACKING}

{OUTPUT_FORMAT}

重要提示：你的模拟必须符合ASD的临床特征，不要表现得像一个普通儿童。
请始终用JSON格式回复，不要包含任何额外文字或markdown标记。"""


# ============================================================
# 第二部分：动作描述转换器
# 把结构化的动作参数转换成自然语言，让GPT理解
# ============================================================

def action_to_text(action: dict) -> str:
    """
    将机器人动作字典转换为自然语言描述。

    参数 action 的格式：
    {
        "speech_rate": "slow" | "normal" | "fast",
        "stimulus": "low" | "medium" | "high",
        "topic": "maintain" | "switch",
        "encouragement": "none" | "moderate" | "frequent"
    }
    """
    speech_map = {
        "slow": "机器人用缓慢、清晰的语速说话，每句话简短有力",
        "normal": "机器人用正常语速说话",
        "fast": "机器人说话较快，语句较长较复杂"
    }
    stimulus_map = {
        "low": "环境安静，没有额外的视觉或听觉刺激",
        "medium": "机器人展示了一些彩色图片和轻柔的背景音乐",
        "high": "机器人播放了丰富的动画效果和生动的音效，视觉和听觉刺激都很强烈"
    }
    topic_map = {
        "maintain": "机器人继续讨论当前的话题",
        "switch": "机器人将话题切换到了一个新的活动"
    }
    encourage_map = {
        "none": "机器人没有给予特别的鼓励",
        "moderate": "机器人适度地鼓励：'你做得不错，继续吧'",
        "frequent": "机器人频繁地给予夸奖：'太棒了！你真聪明！做得非常好！'"
    }

    parts = [
        speech_map.get(action["speech_rate"], "机器人用正常语速说话"),
        stimulus_map.get(action["stimulus"], "刺激强度中等"),
        topic_map.get(action["topic"], "继续当前话题"),
        encourage_map.get(action["encouragement"], "没有特别鼓励"),
    ]

    return "在这一轮交互中：" + "。".join(parts) + "。"


# ============================================================
# 第三部分：仿真器主类
# ============================================================

class ASDSimulator:
    """
    LLM-ASD 仿真器。
    每个实例模拟一个特定类型的ASD儿童，可以进行多轮交互。
    """

    def __init__(self, persona: str = "P1", model: str = "gpt-4o", temperature: float = 0.0):
        """
        初始化仿真器。

        参数：
            persona: "P1"(高响应型) / "P2"(波动型) / "P3"(低参与型)
            model: 使用的GPT模型
            temperature: 生成温度，0.0 = 最确定性（可重复性最好）
        """
        self.persona = persona
        self.model = model
        self.temperature = temperature
        self.system_prompt = build_system_prompt(persona)

        # 对话历史（用于保持多轮交互的连贯性）
        self.conversation_history = []

        # 交互计数器
        self.step_count = 0

        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print(f"✅ ASD仿真器已初始化 | 原型: {persona} | 模型: {model}")

    def reset(self) -> dict:
        """
        重置仿真器到初始状态（开始新的一个episode）。
        返回初始观测。
        """
        self.conversation_history = []
        self.step_count = 0

        # 根据原型设定初始参与度（与数据校准后的基线对齐）
        initial_engagement = {
            "P1": 0.80,  # P1基线 0.80，真实数据中98%时间在STATE-2
            "P2": 0.45,  # P2基线 0.45，真实数据中77%时间在STATE-1
            "P3": 0.28,  # Between 0.20(too low early) and 0.35(too high overall)
        }

        initial_state = {
            "engagement": initial_engagement[self.persona],
            "emotion": "neutral",
            "self_stim": False,
            "fatigue": 0.0,
            "interest": 0.5,
            "verbal_response": "（儿童安静地看着机器人）",
            "internal_reasoning": "初始状态"
        }

        # 把初始状态作为对话的起点
        self.conversation_history.append({
            "role": "assistant",
            "content": json.dumps(initial_state, ensure_ascii=False)
        })

        print(f"🔄 Episode重置 | 初始参与度: {initial_state['engagement']}")
        return initial_state

    def step(self, action: dict) -> dict:
        """
        执行一步交互。

        参数：
            action: 机器人动作字典，包含 speech_rate, stimulus, topic, encouragement

        返回：
            儿童响应的字典，包含 engagement, emotion, self_stim 等
        """
        self.step_count += 1

        # 将动作转换为自然语言
        action_text = action_to_text(action)
        user_message = f"[第{self.step_count}轮交互] {action_text}\n请根据你当前的内部状态，给出这轮交互后的响应。"

        # 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # 调用 GPT-4o-mini
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history
                ],
                temperature=self.temperature,
                max_tokens=300,
            )

            # 提取回复文本
            reply_text = response.choices[0].message.content.strip()

            # 清理可能的 markdown 标记
            if reply_text.startswith("```"):
                reply_text = reply_text.split("\n", 1)[1]  # 去掉第一行 ```json
            if reply_text.endswith("```"):
                reply_text = reply_text[:-3]
            reply_text = reply_text.strip()

            # 解析 JSON
            result = json.loads(reply_text)

            # 添加到对话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": reply_text
            })

            return result

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败（第{self.step_count}轮）: {e}")
            print(f"   原始回复: {reply_text[:200]}")
            # 返回一个默认状态，避免整个实验中断
            return {
                "engagement": 0.5,
                "emotion": "neutral",
                "self_stim": False,
                "fatigue": 0.0,
                "interest": 0.5,
                "verbal_response": "（解析错误）",
                "internal_reasoning": "JSON解析失败，使用默认值"
            }

        except Exception as e:
            print(f"❌ API调用失败（第{self.step_count}轮）: {e}")
            raise


# ============================================================
# 第四部分：快速测试脚本
# 运行这个文件就能立即看到仿真器的效果
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ASD-LLM 仿真器 - 快速测试")
    print("=" * 60)
    print()

    # 创建一个 P1（高响应型）儿童仿真器
    sim = ASDSimulator(persona="P1")

    # 重置（开始新episode）
    initial = sim.reset()
    print(f"初始状态: engagement={initial['engagement']}, emotion={initial['emotion']}")
    print()

    # 定义一系列测试动作
    test_actions = [
        # 第1轮：温和的开场
        {"speech_rate": "slow", "stimulus": "low", "topic": "maintain", "encouragement": "moderate"},
        # 第2轮：增加一些刺激
        {"speech_rate": "normal", "stimulus": "medium", "topic": "maintain", "encouragement": "moderate"},
        # 第3轮：切换话题
        {"speech_rate": "slow", "stimulus": "medium", "topic": "switch", "encouragement": "none"},
        # 第4轮：高强度刺激（测试过激反应）
        {"speech_rate": "fast", "stimulus": "high", "topic": "switch", "encouragement": "frequent"},
        # 第5轮：回到温和方式
        {"speech_rate": "slow", "stimulus": "low", "topic": "maintain", "encouragement": "moderate"},
    ]

    # 逐轮执行
    for i, action in enumerate(test_actions, 1):
        print(f"--- 第 {i} 轮 ---")
        print(f"  机器人动作: 语速={action['speech_rate']}, 刺激={action['stimulus']}, "
              f"话题={action['topic']}, 鼓励={action['encouragement']}")

        result = sim.step(action)

        print(f"  儿童反应: engagement={result['engagement']}, "
              f"emotion={result['emotion']}, self_stim={result['self_stim']}")
        print(f"  语言反应: {result.get('verbal_response', 'N/A')}")
        print(f"  内部推理: {result.get('internal_reasoning', 'N/A')}")
        print()

    print("✅ 测试完成！")
    print(f"   共执行 {sim.step_count} 轮交互")
    print(f"   API调用估算成本: < $0.01")