# -*- coding: utf-8 -*-
"""
实验1 跨模型对比 — 出版级图表（0 token，纯画图，数据全部实读自产物 JSON/轨迹）。
4 模型：gpt-4o / gpt-5.4 / deepseek-v4-flash / deepseek-v4-pro（mini 不画）。
图1 分布保真度分组柱状  图2 ★偏差-vs-多样性散点(诚实,不误导)  图3 轨迹塌缩可视化  图4 成本
输出 png(dpi=300)+pdf 到 artifacts/exp1/figures/
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.append(str(Path(__file__).resolve().parent))
from exp1_ablation import eng_to_state

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT/"artifacts/exp1/figures"; FIG.mkdir(parents=True, exist_ok=True)
LN2 = np.log(2)

# ---- 统一出版级样式 ----
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,"axes.labelsize":12,
    "axes.titleweight":"bold","xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":120,"savefig.bbox":"tight",
})
# 同一模型全图一致配色（flash 用红=退化解）
COL = {"gpt-4o":"#4C72B0","gpt-5.4":"#8172B3","deepseek-v4-pro":"#55A868","deepseek-v4-flash":"#C44E52"}
LAB = {"gpt-4o":"gpt-4o","gpt-5.4":"gpt-5.4","deepseek-v4-pro":"deepseek-v4-pro","deepseek-v4-flash":"deepseek-v4-flash"}
MODELS = ["gpt-4o","gpt-5.4","deepseek-v4-pro","deepseek-v4-flash"]

def save(fig,name):
    for ext in ("png","pdf"):
        fig.savefig(FIG/f"{name}.{ext}", dpi=300)
    print(f"  saved {name}.png/.pdf")

# ================= 实读数据 =================
def load_metrics():
    M={}
    # gpt-4o：JSON 存 ln-JSD，需 ÷ln2 → log2
    d=json.load(open(ROOT/"artifacts/exp1/v2_official_N30.json",encoding="utf-8"))
    M["gpt-4o"]={p:{"JSD":round(d[p]["JSD"]/LN2,4),"cos":d[p]["cosine"],"W1":d[p]["W1"],
                    "D3late":d[p]["D3"]["late"]["err_pct"]} for p in ["P1","P2","P3"]}
    M["gpt-4o"]["cost_ep"]=0.21
    for m,f in [("gpt-5.4","model_compare_gpt-5_4.json"),
                ("deepseek-v4-flash","model_compare_deepseek-v4-flash.json"),
                ("deepseek-v4-pro","model_compare_deepseek-v4-pro.json")]:
        d=json.load(open(ROOT/f"artifacts/exp1/{f}",encoding="utf-8"))
        pp=d["per_persona"]
        M[m]={p:{"JSD":pp[p]["JSD_log2"],"cos":pp[p]["cosine"],"W1":pp[p]["W1"],
                 "D3late":pp[p]["D3"]["late"]["err_pct"]} for p in ["P1","P2","P3"]}
        M[m]["cost_ep"]=d["cost"]["per_ep_real"]
    return M

def late_std(model):
    """P3 late 段跨-ep std（多样性）。4o 用全量 steps，flash/pro 用复现轨迹。"""
    if model=="gpt-4o":
        d=json.load(open(ROOT/"artifacts/exp1/v2_official_N30_steps.json",encoding="utf-8"))
        EP={}
        for r in d["P3"]: EP.setdefault(r["ep"],[]).append((r["step"],r["fin"]))
        eps=[[e for _,e in sorted(v)] for _,v in sorted(EP.items())]
    else:
        f=ROOT/f"artifacts/exp1/traj_{model}_P3.json"
        if not f.exists(): return None,None
        eps=json.load(open(f))["eps"]
    A=np.array([e for e in eps if len(e)==30]); late=A[:,20:]
    return round(float(late.std(0).mean()),4), late   # std均值, late矩阵

def late_traj(model):
    s,late=late_std(model); return late

def div_stats(late):
    """late 多样性统计：std、唯一轨迹数、与众数轨迹相同的条数、总条数。"""
    rows=[tuple(np.round(r,6)) for r in late]
    from collections import Counter
    c=Counter(rows); mode_n=c.most_common(1)[0][1]
    return {"std":round(float(late.std(0).mean()),4),"n_unique":len(c),"n_mode":mode_n,"n_total":len(rows)}

M=load_metrics()
STD={m:late_std(m)[0] for m in ["gpt-4o","deepseek-v4-pro","deepseek-v4-flash"]}  # 5.4 未测轨迹
print("实读 JSD(log2):",{m:{p:M[m][p]["JSD"] for p in["P1","P2","P3"]} for m in MODELS})
print("实读 late std:",STD," | P3 D3late:",{m:M[m]["P3"]["D3late"] for m in MODELS})

# ================= 图1：分布保真度（JSD 分组柱状 + cosine/W1 子图）=================
def fig1():
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    personas=["P1","P2","P3"]; x=np.arange(3); w=0.2
    metrics=[("JSD","JSD (log2)  ↓ lower=better",None),
             ("cos","Cosine similarity  ↑ higher=better",(0.97,1.001)),
             ("W1","Wasserstein-1  ↓ lower=better",None)]
    handles=None
    for ax,(key,ylab,ylim) in zip(axes,metrics):
        for i,m in enumerate(MODELS):
            vals=[M[m][p][key] for p in personas]
            ax.bar(x+(i-1.5)*w,vals,w,label=LAB[m],color=COL[m],edgecolor="white",linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(personas)
        ax.set_ylabel(ylab); ax.set_xlabel("Persona")
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y",alpha=0.25,linewidth=0.6)
        if handles is None: handles,_=ax.get_legend_handles_labels()
    fig.suptitle("Fig.1  Distribution fidelity across 4 models (N=30/persona, same V2 prompt & config)",
                 y=1.13,fontsize=13,fontweight="bold")
    fig.legend(handles,[LAB[m] for m in MODELS],ncol=4,loc="upper center",
               bbox_to_anchor=(0.5,1.07),frameon=False)
    fig.text(0.5,-0.04,"Differences are at small-sample run-to-run scale; all models reach high fidelity "
             "(cosine ≥0.983, small W1). No single model dominates.",ha="center",fontsize=9,style="italic")
    fig.subplots_adjust(wspace=0.28)
    save(fig,"fig1_distribution_fidelity"); plt.close(fig)

# ================= 图2：★偏差-vs-多样性散点（诚实，不误导）=================
def fig2():
    fig,ax=plt.subplots(figsize=(8.4,6.2))
    # 诚实模型（std 已测）
    pts={"gpt-4o":(STD["gpt-4o"],M["gpt-4o"]["P3"]["D3late"]),
         "deepseek-v4-pro":(STD["deepseek-v4-pro"],M["deepseek-v4-pro"]["P3"]["D3late"]),
         "deepseek-v4-flash":(STD["deepseek-v4-flash"],M["deepseek-v4-flash"]["P3"]["D3late"])}
    # 模型名标签：放在各自点的空白侧，避免互压
    name_off={"gpt-4o":(0.0013,1.4),"deepseek-v4-pro":(0.0013,1.4),"deepseek-v4-flash":(0.0013,-3.0)}
    for m,(sx,dy) in pts.items():
        ax.scatter(sx,dy,s=240,color=COL[m],edgecolor="black",linewidth=1.2,zorder=5)
        ox,oy=name_off[m]
        ax.annotate(LAB[m],(sx,dy),xytext=(sx+ox,dy+oy),fontsize=11,fontweight="bold",zorder=6)
    # flash 近塌缩：红圈 + 标注（移到右上大片空白区，箭头指向 flash 点）
    fx,fy=pts["deepseek-v4-flash"]
    fdv=div_stats(late_traj("deepseek-v4-flash"))
    ax.add_patch(Circle((fx,fy),0.0016,fill=False,edgecolor="#C44E52",linewidth=2.2,linestyle="--",zorder=4))
    ax.annotate(f"NEAR-COLLAPSE: lowest diversity\n({fdv['n_mode']}/{fdv['n_total']} identical, only {fdv['n_unique']} unique trajectories;\nstd={fdv['std']:.3f}, still the lowest)\n→ low deviation is inflated, not more faithful",
                (fx,fy),xytext=(0.0035,30),fontsize=9.5,color="#C44E52",fontweight="bold",
                ha="left",va="center",arrowprops=dict(arrowstyle="->",color="#C44E52",lw=1.6,
                connectionstyle="arc3,rad=-0.15"),zorder=6)
    # gpt-5.4：std 未测，画水平参考线
    d54=M["gpt-5.4"]["P3"]["D3late"]
    ax.axhline(d54,color=COL["gpt-5.4"],linestyle=":",linewidth=1.6,alpha=0.8)
    ax.text(0.0565,d54+0.7,f"gpt-5.4: deviation {d54}%  (diversity not measured)",
            color=COL["gpt-5.4"],fontsize=9.5,fontweight="bold",ha="right")
    # 注解：flash 低偏差源于低多样性；pro/4o 多样性相当(同 std)、偏差反映真实拟合(pro<4o)
    ax.text(0.030,2.0,"flash's low deviation is inflated by low diversity (near-collapse); pro & gpt-4o have similar genuine diversity, "
            "where deviation reflects real fit (pro < gpt-4o)",
            fontsize=8.4,style="italic",color="#555",ha="center")
    ax.set_xlabel("P3 late-phase trajectory diversity  (cross-episode std)  →  more diverse",fontweight="bold")
    ax.set_ylabel("P3 D3 late deviation (%)  ↓ lower=closer to real marginal",fontweight="bold",labelpad=8)
    ax.set_xlim(-0.004,0.058); ax.set_ylim(0,57)
    ax.set_title("Fig.2  Deviation vs. Diversity — Lower deviation is NOT necessarily better",
                 fontsize=13,pad=16,loc="center")
    fig.subplots_adjust(left=0.13,top=0.9)
    ax.grid(alpha=0.25,linewidth=0.6)
    fig.text(0.5,-0.02,"flash reaches the lowest D3-late (8.3%) with the lowest trajectory diversity — most episodes converge to one near-identical canned "
             "curve (spike at the same step).\nAmong the more diverse models, pro attains the lowest deviation. D3 alone, a marginal-distribution metric, does not "
             "penalize near-degeneracy. (DeepSeek at temp=0 is not fully deterministic, so flash's diversity is near-collapse, not an absolute freeze.)",
             ha="center",fontsize=8.4,style="italic")
    save(fig,"fig2_deviation_vs_diversity"); plt.close(fig)

# ================= 图3：轨迹塌缩可视化（flash 全重叠 vs 4o 散开）=================
def fig3():
    flash=late_traj("deepseek-v4-flash"); o4=late_traj("gpt-4o")
    nf,no=flash.shape[0],o4.shape[0]
    fdv=div_stats(flash); odv=div_stats(o4)
    fig,axes=plt.subplots(1,2,figsize=(13,5.0),sharey=True)
    steps=np.arange(21,31)
    # flash：~90% 重叠为同一罐头曲线 + 少数逃逸
    for i in range(nf):
        axes[0].plot(steps,flash[i],color=COL["deepseek-v4-flash"],alpha=0.45,linewidth=1.4)
    axes[0].set_title(f"deepseek-v4-flash  (P3 late, {nf} episodes)\n{fdv['n_mode']}/{nf} identical — only {fdv['n_unique']} unique trajectories (near-collapse)",
                      color="#C44E52")
    axes[0].text(21.2,0.435,f"std = {fdv['std']:.3f}  (lowest)\n~{round(100*fdv['n_mode']/nf)}% on one canned curve\n(spike at step 26)",
                 ha="left",va="top",fontsize=9.5,color="#C44E52",fontweight="bold")
    # 4o：30 条真实散开
    for i in range(no):
        axes[1].plot(steps,o4[i],color=COL["gpt-4o"],alpha=0.32,linewidth=1.0)
    axes[1].set_title(f"gpt-4o  (P3 late, {no} episodes)\nGENUINE episode-to-episode variation ({odv['n_unique']} unique)",color=COL["gpt-4o"])
    axes[1].text(21.2,0.435,f"std = {odv['std']:.3f}  (highest)\nspikes scattered",ha="left",va="top",fontsize=9.5,
                 color=COL["gpt-4o"],fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Turn (late phase, steps 21–30)"); ax.grid(alpha=0.25,linewidth=0.6)
        ax.axhspan(0.34,0.66,alpha=0.06,color="orange")
        ax.set_xlim(20.8,30.2)
    axes[0].set_ylabel("Engagement"); axes[0].set_ylim(0,0.46)
    fig.suptitle("Fig.3  Near-collapse vs. genuine fluctuation (visual evidence for the D3 blind spot)",
                 y=1.02,fontsize=13,fontweight="bold")
    fig.text(0.5,-0.03,f"Both panels show all {nf} P3 late-phase episodes (same N). flash's episodes mostly overlap onto one canned curve "
             f"({fdv['n_mode']}/{nf} identical, {fdv['n_unique']} unique), while gpt-4o's vary genuinely.\n"
             "Shaded band = State-1 region (0.34–0.66). DeepSeek at temp=0 is not fully deterministic, so flash is near-collapse (lowest diversity), not an absolute freeze.",
             ha="center",fontsize=8.5,style="italic")
    save(fig,"fig3_trajectory_collapse"); plt.close(fig)

# ================= 图4：成本对比 =================
def fig4():
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    costs=[M[m]["cost_ep"] for m in MODELS]
    bars=ax.bar([LAB[m] for m in MODELS],costs,color=[COL[m] for m in MODELS],edgecolor="white",linewidth=0.6)
    for b,c in zip(bars,costs):
        ax.text(b.get_x()+b.get_width()/2,c+0.004,f"${c:.3f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_ylabel("Cost per episode (USD, real est.)"); ax.set_ylim(0,0.25)
    ax.grid(axis="y",alpha=0.25,linewidth=0.6)
    ax.set_title("Fig.4  Cost per episode (real estimate; DeepSeek prices are placeholders)")
    # 标注 flash 退化解
    ax.annotate("cheapest, but lowest diversity\n(near-collapse; see Fig.2/3)",xy=(3,M["deepseek-v4-flash"]["cost_ep"]),
                xytext=(2.3,0.10),fontsize=9,color="#C44E52",fontweight="bold",
                arrowprops=dict(arrowstyle="->",color="#C44E52",lw=1.4))
    fig.text(0.5,-0.03,"DeepSeek ~5–10× cheaper than GPT, but flash has the lowest trajectory diversity (near-collapse). "
             "Costs are token×list-price ×0.6 estimates; platform billing is ground truth.",ha="center",fontsize=8.6,style="italic")
    save(fig,"fig4_cost"); plt.close(fig)

print("\n生成图表 ->",FIG)
fig1(); fig2(); fig3(); fig4()
print("完成。")
