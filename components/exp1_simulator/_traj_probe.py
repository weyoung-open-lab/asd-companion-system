# -*- coding: utf-8 -*-
"""复现 flash/pro P3 的轨迹(seed=44前N个ep)抓原始engagement, 分析late波动 vs 呆板."""
import os,sys,re,json,time,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT)); sys.path.append(str(ROOT/"components/exp1_simulator"))
from openai import OpenAI
from exp1_ablation import action_to_text, ACTION_POOLS, STEPS_PER_EPISODE, eng_to_state
import exp1_ablation_v2 as v2

def load_dk():
    for ln in open(ROOT/".env",encoding="utf-8"):
        m=re.match(r'(Deepseek_API_KEY|DEEPSEEK_API_KEY)\s*[:=]\s*(\S+)',ln.strip(),re.I)
        if m: return m.group(2)
    raise RuntimeError("no dk")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--neps",type=int,default=10)
    a=ap.parse_args()
    client=OpenAI(api_key=load_dk(),base_url="https://api.deepseek.com")
    def call(msgs):
        for k in range(8):
            try:
                r=client.chat.completions.create(model=a.model,messages=msgs,response_format={"type":"json_object"},
                    temperature=0,max_tokens=350,extra_body={"thinking":{"type":"disabled"}})
                txt=r.choices[0].message.content.strip()
                if txt.startswith("```"): txt=txt.split("\n",1)[1]
                if txt.endswith("```"): txt=txt[:-3]
                res=json.loads(txt.strip())
                if "engagement" in res: return res
            except Exception as e: time.sleep(min(3*2**k,60))
        return {"engagement":0.28}
    np.random.seed(44); sysp=v2.build_prompt_v2("P3"); ie=0.28; eps=[]
    for ep in range(a.neps):
        hist=[{"role":"assistant","content":json.dumps({"engagement":ie,"emotion":"neutral","self_stim":False,"fatigue":0.0,"interest":0.5})}]
        prev=ie; seq=[]
        for s in range(1,STEPS_PER_EPISODE+1):
            act=ACTION_POOLS["P3"][np.random.randint(len(ACTION_POOLS["P3"]))]
            hist.append({"role":"user","content":f"[Turn {s}/30] {action_to_text(act)}\nRespond as JSON."})
            res=call([{"role":"system","content":sysp},*hist])
            hist.append({"role":"assistant","content":json.dumps(res)})
            eng=float(res.get("engagement",prev)); seq.append(eng); prev=eng
        eps.append(seq); print(f"  ep{ep+1}/{a.neps} done",flush=True)
    out=ROOT/f"artifacts/exp1/traj_{a.model}_P3.json"
    json.dump({"model":a.model,"persona":"P3","eps":eps},open(out,"w"),indent=1)
    print("saved",out)
if __name__=="__main__": main()
