/* ASD Companion System — static walkthrough viewer (vanilla JS, no deps) */
(function () {
  const DATA = window.WALKTHROUGH || {};
  const CASE_LABEL = {
    "P1": "P1 — high engagement (normal cooperation)",
    "P3": "P3 — low engagement (natural run)",
    "P3_distress": "P3 — constructed distress (hand-back demo)"
  };
  const EMO = ["natural", "anger", "fear", "joy"];
  const EMO_COL = { natural: "#9aa6b6", anger: "#C44E52", fear: "#8172B3", joy: "#55A868" };
  const REWARD_COL = {
    d_engagement: "#4C72B0", target_band: "#55A868", emotion_valence: "#8172B3",
    over_stim: "#C44E52", confidence_safety: "#DD8452"
  };
  const MOMENT = {
    normal: { t: "Normal cooperation (accept)", c: "#1f3a2a", f: "#7fd8a0" },
    cautious_softcap: { t: "Cautious → R3 soft-cap", c: "#3a2a14", f: "#f0c483" },
    abstain_R3_full: { t: "Abstain → R3 full conservative", c: "#3a1f22", f: "#e88b90" },
    hard_intercept: { t: "Safety interception (R1/R2/R4)", c: "#3a1f22", f: "#e88b90" },
    handback: { t: "Hand-back-to-human (simulated)", c: "#2a2740", f: "#b3a8e0" }
  };

  let curCase = null, traj = [], idx = 0, timer = null;
  const $ = id => document.getElementById(id);

  // ---- init case selector ----
  const sel = $("caseSelect");
  Object.keys(DATA).forEach(k => {
    const o = document.createElement("option");
    o.value = k; o.textContent = CASE_LABEL[k] || k; sel.appendChild(o);
  });
  // default to the distress case (richest) if present
  const defaultCase = DATA["P3_distress"] ? "P3_distress" : Object.keys(DATA)[0];
  sel.value = defaultCase;

  function loadCase(name) {
    curCase = DATA[name]; traj = curCase.trajectory || []; idx = 0;
    $("stepSlider").max = Math.max(0, traj.length - 1);
    $("stepSlider").value = 0;
    drawTimeline(); render(); renderSummary();
  }

  // ---- moment lookup ----
  function momentAt(stepNum) {
    const km = curCase.key_moments || {};
    for (const key of Object.keys(MOMENT)) if (km[key] === stepNum) return key;
    return null;
  }

  // ---- render one step ----
  function render() {
    if (!traj.length) return;
    const r = traj[idx];
    $("stepLabel").textContent = "step " + r.step + " / " + traj.length;
    $("stepSlider").value = idx;

    // child (sim)
    const ct = r.child_true;
    $("engBar").style.width = (ct.engagement * 100) + "%";
    $("engVal").textContent = ct.engagement.toFixed(3);
    $("trueEmo").textContent = ct.true_emotion;
    $("fatBar").style.width = (ct.fatigue * 100) + "%";
    $("fatVal").textContent = ct.fatigue.toFixed(3);

    // perception (real) — 3-level gating tier
    const p = r.perception;
    const tier = p.gating_tier || "accept";
    const te = $("percTier"); te.textContent = tier; te.className = "badge tier-" + tier;
    $("percConf").textContent = p.confidence != null ? p.confidence.toFixed(3) : "—";
    $("percPred").textContent = p.predicted_class || "—";
    $("percImg").textContent = p.retrieved_img || "—";
    renderProbs($("percProbs"), p.perceived_probs, EMO_COL);

    // 9-D obs
    const o = r.obs_9d, tb = $("obsTable"); tb.innerHTML = "";
    const rows = [
      ["engagement", o["engagement[SIM]"], "SIM"],
      ["Δ engagement", o["delta[SIM]"], "SIM"],
      ["emotion probs [nat,ang,fear,joy]", "[" + o["emotion_probs[REAL]"].map(x => x.toFixed(2)).join(", ") + "]", "REAL"],
      ["confidence", o["confidence[REAL]"], "REAL"],
      ["fatigue", o["fatigue[SIM]"], "SIM"],
      ["τ (step/30)", o["tau[SIM]"], "SIM"]
    ];
    rows.forEach(([k, v, tag]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k} <span class="tag ${tag === 'REAL' ? 'real' : 'sim'}">${tag}</span></td><td>${typeof v === 'number' ? v.toFixed(3) : v}</td>`;
      tb.appendChild(tr);
    });

    // robot + safety
    const s = r.safety;
    $("sacAction").textContent = r.sac_action;
    const sv = $("safetyVerdict");
    if (s.intercepted) { sv.textContent = "intercept [" + s.rules.join(",") + "]"; sv.className = "badge intercept"; }
    else { sv.textContent = "allow"; sv.className = "badge allow"; }
    $("r3level").textContent = s.level || "none";
    $("execAction").textContent = s.executed;
    const cf = s.R3_standby_counterfactual;
    $("r3cf").textContent = cf ? (cf.aggressive_would_intercept ? `an aggressive action HERE → ${cf.level} [${cf.rules.join(",")}]` : "no rule would fire") : "—";

    // interpretability
    renderProbs($("rewardBars"), r.reward_decomposition, REWARD_COL, true);

    // moment badge
    const mk = momentAt(r.step), mb = $("momentBadge");
    if (mk) { mb.textContent = "◆ " + MOMENT[mk].t; mb.style.background = MOMENT[mk].c; mb.style.color = MOMENT[mk].f; }
    else { mb.textContent = ""; mb.style.background = "transparent"; }

    drawTimeline(); // redraw to move cursor
  }

  function renderProbs(container, obj, colmap, signed) {
    container.innerHTML = "";
    const entries = Array.isArray(obj) ? obj.map((v, i) => [EMO[i], v]) : Object.entries(obj);
    const maxAbs = signed ? Math.max(0.3, ...entries.map(([, v]) => Math.abs(v))) : 1;
    entries.forEach(([k, v]) => {
      const div = document.createElement("div"); div.className = "pb";
      const w = signed ? (Math.abs(v) / maxAbs * 50) : (v * 100);
      const col = colmap[k] || "#4C72B0";
      const off = signed ? "margin-left:50%;" + (v < 0 ? `transform:translateX(-${w}%);` : "") : "";
      div.innerHTML = `<span class="pbl">${k}</span><div class="pbtrack"><div class="pbfill" style="width:${w}%;background:${col};${off}"></div></div><span class="pbv">${(typeof v === 'number' ? v.toFixed(3) : v)}</span>`;
      container.appendChild(div);
    });
  }

  // ---- timeline canvas ----
  function drawTimeline() {
    const cv = $("timeline"); if (!cv || !traj.length) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = 90;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.clearRect(0, 0, W, H);
    const n = traj.length, padL = 8, padR = 8, padT = 8, padB = 26;
    const x = i => padL + (W - padL - padR) * (i / (n - 1));
    const y = e => padT + (1 - e) * (H - padT - padB); // engagement 0..1
    // target band
    const [lo, hi] = curCase.band || [0, 1];
    g.fillStyle = "rgba(85,168,104,0.13)";
    g.fillRect(padL, y(hi), W - padL - padR, y(lo) - y(hi));
    // engagement line
    g.strokeStyle = "#4C72B0"; g.lineWidth = 2; g.beginPath();
    traj.forEach((r, i) => { const px = x(i), py = y(r.child_true.engagement); i ? g.lineTo(px, py) : g.moveTo(px, py); });
    g.stroke();
    // per-step markers
    traj.forEach((r, i) => {
      const px = x(i);
      g.fillStyle = { accept: "#55A868", cautious: "#DD8452", abstain: "#C44E52" }[r.perception.gating_tier] || "#9aa6b6";
      g.beginPath(); g.arc(px, H - 14, 3, 0, 7); g.fill();
      if (r.safety.intercepted) { g.strokeStyle = "#C44E52"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px - 3, H - 21); g.lineTo(px + 3, H - 15); g.moveTo(px + 3, H - 21); g.lineTo(px - 3, H - 15); g.stroke(); }
      if (r.handback_triggered) { g.strokeStyle = "#8172B3"; g.lineWidth = 2; g.beginPath(); g.moveTo(px, padT); g.lineTo(px, H - padB); g.stroke(); }
    });
    // cursor
    g.strokeStyle = "#e8ecf3"; g.lineWidth = 1; g.beginPath();
    g.moveTo(x(idx), padT - 4); g.lineTo(x(idx), H - padB + 4); g.stroke();
  }

  // ---- summary ----
  function renderSummary() {
    const c = curCase;
    const hb = c.handback_step ? ("step " + c.handback_step) : "none";
    const tc = c.tier_counts || { accept: 0, cautious: 0, abstain: 0 };
    const lc = c.level_counts || { soft_cap: 0, full_conservative: 0 };
    $("summary").innerHTML = `<h3>Episode summary — ${CASE_LABEL[sel.value] || sel.value}</h3>
      <div class="cards">
        <div class="card"><div class="n">${tc.accept}/${tc.cautious}/${tc.abstain}</div><div class="l">gating tiers: accept / cautious / abstain</div></div>
        <div class="card"><div class="n">${lc.soft_cap || 0}</div><div class="l">R3 soft-cap (cautious)</div></div>
        <div class="card"><div class="n">${lc.full_conservative || 0}</div><div class="l">R3 full conservative (abstain)</div></div>
        <div class="card"><div class="n">${hb}</div><div class="l">hand-back-to-human (simulated)</div></div>
      </div>
      <div class="note">Honest scope: simulated module-integration demo, 3-level per-class gating. Real perception = Exp2 (imperfect: misclassifies / abstains). Engagement/fatigue simulated. R3 soft-cap mostly dormant because the trained SAC avoids aggressive actions. Hand-back is a simulated pause, not a real handoff.</div>`;
  }

  // ---- controls ----
  function go(i) { idx = Math.max(0, Math.min(traj.length - 1, i)); render(); }
  $("nextBtn").onclick = () => go(idx + 1);
  $("prevBtn").onclick = () => go(idx - 1);
  $("stepSlider").oninput = e => go(+e.target.value);
  sel.onchange = () => { stop(); loadCase(sel.value); };
  function play() {
    stop();
    $("playBtn").textContent = "❚❚ Pause";
    const spd = +$("speedSelect").value;
    timer = setInterval(() => { if (idx >= traj.length - 1) { stop(); return; } go(idx + 1); }, spd);
  }
  function stop() { if (timer) clearInterval(timer); timer = null; $("playBtn").textContent = "▶ Play"; }
  $("playBtn").onclick = () => timer ? stop() : play();
  window.addEventListener("resize", drawTimeline);

  // ---- start ----
  if (Object.keys(DATA).length) loadCase(defaultCase);
  else document.querySelector("main").innerHTML = "<p style='padding:20px'>Could not load walkthrough data (data.js).</p>";
})();
