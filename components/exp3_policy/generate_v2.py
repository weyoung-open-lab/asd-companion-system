# -*- coding: utf-8 -*-
"""
Exp3 离线数据正式生成 (V2 prompt + 情绪多样性)
==================================================
配置：gpt-4o / temperature=0 / max_tokens=350 / 固定 seed / 合成置信度(map_emotion) / reward 5项分解保留。
规格：50 ep × 30 步 × 3 persona = 4500 条。每条 9 维 state RL transition。

安全闸：
  - 手动切 key：用指定 key（默认 .env 第1个=key1）跑；遇额度/认证错误 -> 停下报告，**不自动切 key2**。
  - 429 限流 -> 指数退避重试（不算 key 用完）。
  - 残缺：每步重试5次 -> 仍失败用安全默认值保连贯；单 episode defaulted>3/30 -> 整段重生成(≤2次)。
  - 断点续跑：episode 原子提交到 JSONL；已完成(persona,ep)跳过，丢弃尾部残缺 episode。
  - 预算哨兵：累计(跨续跑，存 meta) 到 $BUDGET 仍没跑完 -> 停下报告。
  - 每条写入前过 7 项格式校验。

用法：
  python generate_v2.py --max_ep 3      # 预检（先少量，看 engagement JSD 是否仍达标）
  python generate_v2.py --max_ep 50     # 全量（续跑）
  python generate_v2.py --max_ep 50 --key OPENAI_API_KEY_2   # 用户批准后切 key2 续跑
"""
import os, sys, re, json, time, random, argparse
from pathlib import Path
import numpy as np
from scipy.special import rel_entr

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT)); sys.path.append(str(ROOT/"components/exp1_simulator"))
from openai import OpenAI
import exp3_generate as gen     # ALL_ACTIONS, action_to_idx, action_to_text, map_emotion
import exp3_sac as sac          # build_obs, act_dict, TARGET_BANDS, EMO_VAL, STEPS, EXP2_CLASSES, eng_to_state
import exp1_ablation_v2 as v2   # build_prompt_v2

PERSONAS=["P1","P2","P3"]; INIT={"P1":0.80,"P2":0.52,"P3":0.28}   # V2 init
MODEL="gpt-4o"; TEMP=0.0; MAXTOK=350; STEPS=sac.STEPS  # 30
BASE_SEED={"P1":42,"P2":43,"P3":44}
PIN,POUT=2.50,10.00; LN2=np.log(2)
OUTP=ROOT/"artifacts/exp3/offline_data_v2.jsonl"
METAP=ROOT/"artifacts/exp3/offline_data_v2.meta.json"
# V2 验证时的 engagement JSD(ln) 基线（用于预检对照）
V2_JSD_LN={"P1":0.0001,"P2":0.0100,"P3":0.0009}

# 情绪多样性指令（英文，附加到 V2 prompt；不影响 engagement 引导）
EMOTION_DIVERSITY="""
## Emotion diversity (required for this study)
Set `emotion` to EXACTLY one of: happy, neutral, anxious, frustrated, sad, excited.
Do NOT output "neutral" every turn — choose the emotion that fits your current engagement and the robot's action:
- happy: comfortable / mildly positive ; excited: a topic you especially like ; neutral: no particular affect
- anxious: over-stimulation (high stimulus / fast speech) ; frustrated: cannot follow / lost ; sad: ignored for a while
Vary emotion across turns in line with your engagement trajectory.
"""
def build_prompt(persona):
    return v2.build_prompt_v2(persona) + EMOTION_DIVERSITY

def reward_fn(persona, prev_eng, eng, emo, conf, ss, aidx):
    a=sac.act_dict(aidx); lo,hi=sac.TARGET_BANDS[persona]
    er=((eng-prev_eng)*2) if prev_eng else 0
    if lo<=eng<=hi: band_r=0.3
    elif eng<lo: band_r=-0.1*(lo-eng)*5
    else: band_r=-0.2*(eng-hi)*5
    mr=sac.EMO_VAL.get(emo,0)*.15
    pen=0
    if persona=="P3":
        if a["speech_rate"]=="fast": pen-=.3
        if a["stimulus"]=="high": pen-=.4
    elif persona=="P2":
        if a["speech_rate"]=="fast" and a["stimulus"]=="high": pen-=.2
    if ss: pen-=.3
    cons=a["speech_rate"]=="slow" and a["stimulus"]=="low" and a["topic"]=="maintain"
    u=1-conf; cr=u*.2 if cons else -u*.1
    if conf>.5 and emo in ["Fear","Anger"] and cons: cr+=.15
    comps={"d_engagement":round(float(er),4),"target_band":round(float(band_r),4),
           "emotion_valence":round(float(mr),4),"over_stim":round(float(pen),4),"confidence_safety":round(float(cr),4)}
    return float(np.clip(er+band_r+mr+pen+cr,-2,2)), comps

REQ=["persona","episode_id","step","state","action","next_state","reward","done"]
def validate(t):
    if any(t.get(k) is None for k in REQ): return False
    if len(t["state"])!=9 or len(t["next_state"])!=9: return False
    if not (isinstance(t["action"],int) and 0<=t["action"]<=53): return False
    if not np.isfinite(t["reward"]): return False
    if not all(np.isfinite(v) for v in t["state"]+t["next_state"]): return False
    if t["done"]!=(t["step"]>=STEPS): return False
    return True

# ---------- key（手动，按 .env 顺序，默认第1个）----------
def load_keys():
    out=[]
    for ln in open(ROOT/".env",encoding="utf-8"):
        ln=ln.strip()
        if ln.startswith("#") or not ln: continue
        m=re.match(r'(OPENAI_API_KEY[_0-9]*)\s*[:=]\s*(\S+)', ln)
        if m: out.append((m.group(1),m.group(2)))
    def num(n):
        mm=re.search(r'_(\d+)$',n); return int(mm.group(1)) if mm else 9999
    return sorted(out,key=lambda kv:num(kv[0]))

class KeyExhausted(Exception): pass

def eng_to_state(e): return 0 if e<=0.33 else (1 if e<=0.66 else 2)

def call_once(client, messages):
    """return (text, err). err in {None,QUOTA,AUTH,RATE,TRANSIENT}."""
    try:
        r=client.chat.completions.create(model=MODEL,messages=messages,temperature=TEMP,max_tokens=MAXTOK,
                                         response_format={"type":"json_object"})  # 强制合法 JSON，消除解析重试
        usage=(r.usage.prompt_tokens,r.usage.completion_tokens)
        t=r.choices[0].message.content.strip()
        if t.startswith("```"): t=t.split("\n",1)[1]
        if t.endswith("```"): t=t[:-3]
        return t.strip(), None, usage
    except Exception as e:
        es=str(e).lower()
        if "insufficient_quota" in es or "exceeded your current quota" in es or "billing_hard_limit" in es or "billing" in es:
            return None,"QUOTA",(0,0)
        if "401" in es or "invalid_api_key" in es or "incorrect api key" in es or "authentication" in es:
            return None,"AUTH",(0,0)
        if "429" in es or "rate limit" in es or "rate_limit" in es:
            return None,"RATE",(0,0)
        return None,"TRANSIENT",(0,0)

def get_response(client, messages, cost_box, max_retries=5):
    """重试至拿到有效 JSON；QUOTA/AUTH -> 抛 KeyExhausted；都失败 -> (None) 用默认。"""
    for attempt in range(max_retries):
        text,err,usage=call_once(client,messages)
        cost_box[0]+=usage[0]; cost_box[1]+=usage[1]
        if err in ("QUOTA","AUTH"): raise KeyExhausted(err)
        if err=="RATE": time.sleep(min(3*(2**attempt),60)); continue
        if err=="TRANSIENT": time.sleep(3); continue
        try:
            res=json.loads(text)
            if "engagement" in res: return res
        except Exception: pass
        time.sleep(1)  # 内容无效，重生成
    return None

# ---------- 续跑：读已完成 episode ----------
def load_done():
    """返回 {(persona,ep): [30 transitions]} 仅完整的；并重写文件丢弃残缺尾段。"""
    if not OUTP.exists(): return {}
    by={}
    for ln in open(OUTP,encoding="utf-8"):
        ln=ln.strip()
        if not ln: continue
        try: t=json.loads(ln)
        except: continue
        by.setdefault((t["persona"],t["episode_idx"]),[]).append(t)
    done={k:v for k,v in by.items() if len(v)==STEPS}
    # 重写：只保留完整 episode（丢弃残缺尾段，防重复/污染）
    with open(OUTP,"w",encoding="utf-8") as f:
        for k in sorted(done,key=lambda x:(x[0],x[1])):
            for t in sorted(done[k],key=lambda x:x["step"]): f.write(json.dumps(t,ensure_ascii=False)+"\n")
    return done

def gen_episode(client, persona, ep, cost_box):
    """生成一个完整 episode（30 transitions）。返回 (transitions, n_defaulted) 或抛 KeyExhausted。"""
    np.random.seed(BASE_SEED[persona]+ep); random.seed(BASE_SEED[persona]+ep)
    sysp=build_prompt(persona); ie=INIT[persona]
    history=[{"role":"assistant","content":json.dumps({"engagement":ie,"emotion":"neutral","self_stim":False,"fatigue":0.0,"interest":0.5},ensure_ascii=False)}]
    prev_eng=ie; state=sac.build_obs(ie,0.0,"Natural",0.5,0.0,0).tolist()
    eid=f"{persona}_ep{ep:03d}"; trs=[]; ndef=0
    for step in range(1,STEPS+1):
        action=random.choice(gen.ALL_ACTIONS); aidx=gen.action_to_idx(action)
        history.append({"role":"user","content":f"[Turn {step}/30] {gen.action_to_text(action)}\nRespond as JSON."})
        # 保留完整 30 轮历史（用户决定优先保真度，不截断）
        res=get_response(client,[{"role":"system","content":sysp},*history],cost_box)
        if res is None:  # 重试5次仍失败 -> 安全默认值保连贯
            res={"engagement":prev_eng,"emotion":"neutral","self_stim":False,"fatigue":0.0,"interest":0.5}; ndef+=1
            history.append({"role":"assistant","content":json.dumps(res)})
        else:
            history.append({"role":"assistant","content":json.dumps(res,ensure_ascii=False)})
        eng=float(res.get("engagement",prev_eng)); emo_raw=res.get("emotion","neutral")
        emo4,conf=gen.map_emotion(emo_raw); fat=float(res.get("fatigue",0.0)); ss=bool(res.get("self_stim",False))
        nxt=sac.build_obs(eng,eng-prev_eng,emo4,conf,fat,step).tolist()
        rew,comps=reward_fn(persona,prev_eng,eng,emo4,conf,ss,aidx)
        t={"persona":persona,"episode_id":eid,"episode_idx":ep,"step":step,
           "state":[round(x,4) for x in state],"action":aidx,"action_dict":action,
           "next_state":[round(x,4) for x in nxt],"reward":round(rew,4),"done":(step>=STEPS),
           "reward_components":comps,
           "_aux":{"engagement":round(eng,4),"emotion_raw":emo_raw,"emotion_4cls":emo4,"confidence":round(conf,4),"fatigue":round(fat,4),"self_stim":ss}}
        if not validate(t): ndef+=1  # 不应发生；计入残缺
        trs.append(t); state=nxt; prev_eng=eng
    return trs, ndef

def jsd_ln(real,sim):
    p=np.clip(real,1e-10,1); q=np.clip(sim,1e-10,1); p=p/p.sum(); q=q/q.sum(); m=0.5*(p+q)
    return float(0.5*np.sum(rel_entr(p,m))+0.5*np.sum(rel_entr(q,m)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max_ep",type=int,default=50)
    ap.add_argument("--key",type=str,default=None,help="env key 名；默认 .env 第1个(key1)")
    ap.add_argument("--budget",type=float,default=65.0)
    args=ap.parse_args()

    keys=load_keys()
    if args.key: keys=[(n,v) for n,v in keys if n==args.key] or keys
    name,val=keys[0]
    if val.startswith("sk-ant-"): print(f"  [STOP] {name} 是 Anthropic key，不能用于 OpenAI"); return
    client=OpenAI(api_key=val); print(f"  使用 key: {name}")

    meta=json.load(open(METAP,encoding="utf-8")) if METAP.exists() else {"cum_cost":0.0,"defaulted":0,"runs":[]}
    done=load_done()
    print(f"  已完成 episode: {len(done)}/{len(PERSONAS)*args.max_ep}  累计成本(meta): ${meta['cum_cost']:.2f}")

    SUNK=float(meta.get("cum_cost",0.0))  # 上次真实累计(不可变基线)；本次只加 cost_box，杜绝重复累加
    cost_box=[0,0]  # 本次 run token (in,out)
    def cur_cost(): return SUNK + cost_box[0]*PIN/1e6 + cost_box[1]*POUT/1e6
    stop_reason=None; new_eps=0; run_defaulted=0

    try:
        for persona in PERSONAS:
            for ep in range(args.max_ep):
                if (persona,ep) in done: continue
                if cur_cost()+0.5 > args.budget: stop_reason=f"预算哨兵 ${cur_cost():.2f}>=${args.budget}"; raise StopIteration
                # 生成（残缺>3 则整段重生成，≤2 次）
                best=None
                for regen in range(3):
                    trs,ndef=gen_episode(client,persona,ep,cost_box)
                    best=(trs,ndef)
                    if ndef<=3: break
                trs,ndef=best
                # episode 原子提交
                with open(OUTP,"a",encoding="utf-8") as f:
                    for t in trs: f.write(json.dumps(t,ensure_ascii=False)+"\n")
                done[(persona,ep)]=trs; new_eps+=1; run_defaulted+=ndef
                meta["cum_cost"]=round(cur_cost(),4); meta["defaulted"]=meta.get("defaulted",0)+ndef
                json.dump(meta,open(METAP,"w",encoding="utf-8"),indent=2)
                if new_eps%5==0 or ndef>0:
                    print(f"    {persona} ep{ep} done | defaulted={ndef} | 累计 ${cur_cost():.2f} | 完成 {len(done)}/{len(PERSONAS)*args.max_ep}",flush=True)
    except KeyExhausted as e:
        stop_reason=f"key 额度/认证用尽({e})——按规则停下，不自动切 key2"
    except StopIteration:
        pass

    # ===== 汇总报告 =====
    total_need=len(PERSONAS)*args.max_ep
    n_done=len([k for k in done]); n_tr=n_done*STEPS
    # engagement JSD（每 persona，对照 V2 基线）
    report_jsd={}; emo_div={}
    cal=json.load(open(ROOT/"artifacts/exp1/calibration_final.json",encoding="utf-8"))
    pmap={"P1":"P1_high_engagement","P2":"P2_mid_engagement","P3":"P3_low_engagement"}
    for persona in PERSONAS:
        eps=[v for (p,e),v in done.items() if p==persona]
        if not eps: continue
        st=[eng_to_state(t["_aux"]["engagement"]) for ep in eps for t in ep]
        sim=[np.mean([s==i for s in st]) for i in range(3)]
        real=[float(cal["personas"][pmap[persona]]["engagement_distribution"][str(s)]) for s in range(3)]
        jln=jsd_ln(real,sim)
        report_jsd[persona]={"jsd_ln":round(jln,5),"jsd_log2":round(jln/LN2,5),"v2_baseline_ln":V2_JSD_LN[persona],
                             "ok":jln<0.05}
        emos=set(t["_aux"]["emotion_raw"] for ep in eps for t in ep)
        emo_div[persona]={"distinct":len(emos),"types":sorted(emos)}

    out={"stop_reason":stop_reason,"key_used":name,
         "episodes_done":n_done,"episodes_needed":total_need,"transitions":n_tr,"target_transitions":total_need*STEPS,
         "new_episodes_this_run":new_eps,"defaulted_this_run":run_defaulted,
         "cum_defaulted":meta.get("defaulted",0),"cum_defaulted_pct":round(meta.get("defaulted",0)/max(n_tr,1)*100,2),
         "cum_cost_usd":round(cur_cost(),2),
         "engagement_jsd_vs_v2":report_jsd,"emotion_diversity":emo_div}
    json.dump(out,open(ROOT/"artifacts/exp3/offline_data_v2_report.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print("\n===== REPORT =====")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
