# -*- coding: utf-8 -*-
"""
实验1 跨模型对比：同 V2 prompt + 同配置 + N=30/persona，只换模型。
保真度(JSD log2 / cosine / W1 / D3) + 成本 + 速度 + 稳定性。
gpt-4o 复用 v2_official_N30.json（不重跑）；mini/5.5 用本脚本跑。

模型自适应配置：
  gpt-4o / gpt-4o-mini : temperature=0, max_tokens=350
  gpt-5* (如 gpt-5.5)  : max_completion_tokens=350, temperature=1（5.5 只支持默认1）
统一：V2 prompt(无情绪多样性，与实验1主结果一致)、30轮历史、response_format强制JSON、seed。

用法：
  python compare_models.py --model gpt-4o-mini --max_ep 30
  python compare_models.py --model gpt-5.5 --max_ep 2      # 先验
"""
import os, sys, re, json, time, argparse
from pathlib import Path
import numpy as np
from scipy.special import rel_entr
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT)); sys.path.append(str(ROOT/"components/exp1_simulator"))
from openai import OpenAI
from exp1_ablation import eng_to_state, action_to_text, ACTION_POOLS, compute_metrics, STEPS_PER_EPISODE
import exp1_ablation_v2 as v2

PERSONAS=["P1","P2","P3"]; INIT={"P1":0.80,"P2":0.52,"P3":0.28}
PMAP={"P1":"P1_high_engagement","P2":"P2_mid_engagement","P3":"P3_low_engagement"}
BASE_SEED={"P1":42,"P2":43,"P3":44}
LN2=np.log(2)
# list 价（$/1M）。real≈×0.6（prompt 缓存）。5.5 价未知，按 4o 占位、以实测 token 为准报相对。
PRICE={"gpt-4o":(2.50,10.0),"gpt-4o-mini":(0.15,0.60),"gpt-5.5":(2.50,10.0),
       "gpt-5.4":(2.50,10.0),"gpt-5.4-mini":(0.30,1.20),
       "deepseek-v4-flash":(0.30,0.50),"deepseek-v4-pro":(0.60,1.00)}  # deepseek 价占位(pro 估 2×flash)；以 DeepSeek 平台账单为准。5.4 经$1看板反推≈4o价
CALIB=json.load(open(ROOT/"artifacts/exp1/calibration_final.json",encoding="utf-8"))

def is_gpt5(m): return m.startswith("gpt-5")
def temp1_only(m): return m.startswith("gpt-5.5")   # 仅 5.5 强制 temp=1；5.4/5.4-mini 支持 temp=0
def is_deepseek(m): return m.startswith("deepseek")
def make_call(client, model, messages):
    kw=dict(model=model,messages=messages,response_format={"type":"json_object"})
    if is_deepseek(model):
        kw["temperature"]=0; kw["max_tokens"]=350
        kw["extra_body"]={"thinking":{"type":"disabled"}}   # 非思考模式（实测此开关有效）
    elif is_gpt5(model):
        kw["max_completion_tokens"]=350; kw["seed"]=42      # gpt-5 系列用 max_completion_tokens
        if not temp1_only(model): kw["temperature"]=0       # 5.4/5.4-mini 可 temp=0
    else:
        kw["temperature"]=0; kw["max_tokens"]=350; kw["seed"]=42
    return client.chat.completions.create(**kw)

def load_key(which=None, deepseek=False):
    if deepseek:
        for ln in open(ROOT/".env",encoding="utf-8"):
            ln=ln.strip()
            m=re.match(r'(Deepseek_API_KEY|DEEPSEEK_API_KEY)\s*[:=]\s*(\S+)', ln, re.I)
            if m: return (m.group(1), m.group(2))
        raise RuntimeError("no deepseek key in .env")
    keys=[]
    for ln in open(ROOT/".env",encoding="utf-8"):
        ln=ln.strip()
        if ln.startswith("#") or not ln: continue
        m=re.match(r'(OPENAI_API_KEY[_0-9]*)\s*[:=]\s*(\S+)',ln)
        if m: keys.append((m.group(1),m.group(2)))
    keys.sort(key=lambda kv:int(re.search(r'_(\d+)$',kv[0]).group(1)) if re.search(r'_(\d+)$',kv[0]) else 9999)
    if which: keys=[(n,v) for n,v in keys if n==which] or keys
    return keys[0]

class KeyExhausted(Exception): pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--max_ep",type=int,default=30)
    ap.add_argument("--key",default=None)
    ap.add_argument("--budget",type=float,default=30.0)
    ap.add_argument("--pace",type=float,default=0.0,help="每次调用后 sleep 秒数，压低 TPM 防限流(mini 用)")
    args=ap.parse_args()
    kname,kval=load_key(args.key, deepseek=is_deepseek(args.model))
    client=OpenAI(api_key=kval, base_url="https://api.deepseek.com") if is_deepseek(args.model) else OpenAI(api_key=kval)
    pin,pout=PRICE.get(args.model,(2.5,10.0))
    print(f"  model={args.model} | key={kname} | N={args.max_ep}/persona | temp={'1(forced)' if is_gpt5(args.model) else '0'}",flush=True)

    tok_in=tok_out=0; n_calls=0; n_json_fail=0; latencies=[]; t0=time.time()
    def cost_list(): return tok_in*pin/1e6+tok_out*pout/1e6
    def call_valid(messages, max_retry=8):
        nonlocal tok_in,tok_out,n_calls,n_json_fail
        for a in range(max_retry):
            try:
                ts=time.time(); r=make_call(client,args.model,messages); latencies.append(time.time()-ts)
                n_calls+=1; tok_in+=r.usage.prompt_tokens; tok_out+=r.usage.completion_tokens
                txt=r.choices[0].message.content.strip()
                if txt.startswith("```"): txt=txt.split("\n",1)[1]
                if txt.endswith("```"): txt=txt[:-3]
                res=json.loads(txt.strip())
                if "engagement" in res: return res
                n_json_fail+=1; time.sleep(1)   # 真·内容失败：合法JSON但缺 engagement（罕见）
            except Exception as e:
                es=str(e).lower()
                if "insufficient_quota" in es or "exceeded your current quota" in es or "billing" in es: raise KeyExhausted("QUOTA")
                if "401" in es or "invalid_api_key" in es or "authentication" in es: raise KeyExhausted("AUTH")
                time.sleep(min(3*2**a,90))      # 其余(429/限流/瞬时)统一指数退避(最多90s)，不计 json_fail
        return None

    results={}; defaulted=0; stop=None
    try:
        for p in PERSONAS:
            np.random.seed(BASE_SEED[p])
            sysp=v2.build_prompt_v2(p); ie=INIT[p]; seqs=[]
            for ep in range(args.max_ep):
                if cost_list()*0.6+0.3>args.budget: stop=f"预算(real~${cost_list()*0.6:.2f})"; raise KeyExhausted("BUDGET")
                hist=[{"role":"assistant","content":json.dumps({"engagement":ie,"emotion":"neutral","self_stim":False,"fatigue":0.0,"interest":0.5})}]
                prev=ie; seq=[]
                for s in range(1,STEPS_PER_EPISODE+1):
                    act=ACTION_POOLS[p][np.random.randint(len(ACTION_POOLS[p]))]
                    hist.append({"role":"user","content":f"[Turn {s}/30] {action_to_text(act)}\nRespond as JSON."})
                    res=call_valid([{"role":"system","content":sysp},*hist])
                    if args.pace>0: time.sleep(args.pace)   # 节流：压低 TPM 防限流
                    if res is None: res={"engagement":prev,"emotion":"neutral"}; defaulted+=1
                    hist.append({"role":"assistant","content":json.dumps(res)})
                    eng=float(res.get("engagement",prev)); seq.append(eng); prev=eng
                seqs.append(seq)
                # 逐 episode 进度/成本落盘，便于实时监控
                json.dump({"model":args.model,"persona":p,"ep_done":ep+1,"calls":n_calls,
                           "list_usd":round(cost_list(),3),"real_est_usd":round(cost_list()*0.6,3),
                           "in_tok":tok_in,"out_tok":tok_out,"json_fail":n_json_fail,"defaulted":defaulted},
                          open(ROOT/f"artifacts/exp1/model_compare_{args.model.replace('.','_')}.progress.json","w"))
                if (ep+1)%10==0: print(f"    {p} ep{ep+1}/{args.max_ep} | list ${cost_list():.2f} real~${cost_list()*0.6:.2f} | calls={n_calls}",flush=True)
            full=[s for s in seqs if len(s)==STEPS_PER_EPISODE]
            m=compute_metrics(full,CALIB,PMAP[p]); d3=m.get("D3_temporal",{})
            results[p]={"episodes":len(full),"JSD_ln":m["D1_JSD"],"JSD_log2":round(m["D1_JSD"]/LN2,5),
                        "cosine":m["D1_cosine"],"W1":m["D2_wasserstein"],
                        "D3":{k:{"real":v["real"],"sim":v["sim"],"err_pct":v["rel_error_pct"]} for k,v in d3.items()}}
            print(f"    [{p}] JSD_log2={results[p]['JSD_log2']} cos={m['D1_cosine']} W1={m['D2_wasserstein']} D3late={d3.get('late',{}).get('rel_error_pct')}%",flush=True)
    except KeyExhausted as e:
        stop=stop or f"{e}"

    out={"model":args.model,"key":kname,"max_ep":args.max_ep,"stop":stop,
         "config":{"temperature":1 if temp1_only(args.model) else 0,"token_param":"max_completion_tokens" if is_gpt5(args.model) else "max_tokens",
                   "response_format":"json_object","history":"full-30","prompt":"V2 (no emotion-diversity)"},
         "per_persona":results,
         "cost":{"list_usd":round(cost_list(),3),"real_est_usd":round(cost_list()*0.6,3),"per_ep_real":round(cost_list()*0.6/max(sum(r['episodes'] for r in results.values()),1),4),
                 "note":"list=token×单价上限; real≈×0.6(prompt缓存); 以OpenAI看板为准"},
         "speed":{"n_calls":n_calls,"avg_latency_s":round(np.mean(latencies),2) if latencies else None,"total_min":round((time.time()-t0)/60,1)},
         "stability":{"json_fail":n_json_fail,"json_fail_rate":round(n_json_fail/max(n_calls,1),4),"defaulted":defaulted}}
    outp=ROOT/f"artifacts/exp1/model_compare_{args.model.replace('.','_')}.json"
    json.dump(out,open(outp,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print("\n===== "+args.model+" =====")
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
