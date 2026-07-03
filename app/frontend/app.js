/* ASD Companion System — live demo frontend */
(function () {
  const $ = id => document.getElementById(id);
  const EMO = ["natural", "anger", "fear", "joy"];
  const EMO_COL = { natural: "#9aa6b6", anger: "#C44E52", fear: "#8172B3", joy: "#55A868" };
  const EMO_EMOJI = { neutral: "😐", natural: "😐", Natural: "😐", anger: "😠", Anger: "😠",
    fear: "😨", Fear: "😨", joy: "😊", Joy: "😊", happy: "😊", excited: "🤩", sad: "😢", frustrated: "😣", anxious: "😟" };
  const REWARD_COL = { d_engagement: "#4C72B0", target_band: "#55A868", emotion_valence: "#8172B3", over_stim: "#C44E52", confidence_safety: "#DD8452" };
  let sid = null, auto = null, busy = false;

  // --- pure translation: 4-D action code -> plain-language sentence ---
  function actionToSentence(actStr) {
    const parts = (actStr || "").split("/"); if (parts.length !== 4) return actStr || "—";
    const [sp, st, tp, en] = parts;
    const SP = { slow: "speaks slowly", normal: "speaks at a normal pace", fast: "speaks quickly" };
    const ST = { low: "keeps stimulation low", medium: "uses moderate stimulation", high: "uses high stimulation" };
    const TP = { maintain: "stays on the current topic", switch: "switches to a new topic" };
    const EN = { none: "gives no extra encouragement", moderate: "gives moderate encouragement", frequent: "gives frequent encouragement" };
    return `The robot ${SP[sp] || sp}, ${ST[st] || st}, ${TP[tp] || tp}, and ${EN[en] || en}.`;
  }

  // --- data-based state interpretation (engagement + emotion ONLY; no fabricated behaviour) ---
  function childStateSentence(eng, emo) {
    const lvl = eng < 0.34 ? "Low engagement" : eng < 0.66 ? "Moderate engagement" : "High engagement";
    const dis = eng < 0.34 ? "the child appears disengaged" : eng < 0.66 ? "the child is partially engaged" : "the child is highly engaged";
    const AFF = {
      Joy: "positive affect", joy: "positive affect", happy: "positive affect", excited: "high positive arousal",
      Natural: "neutral affect", neutral: "neutral affect",
      Anger: "negative affect (anger)", anger: "negative affect (anger)", frustrated: "frustration",
      Fear: "anxiety/fear", fear: "anxiety/fear", anxious: "anxiety", sad: "low/negative affect"
    };
    const aff = AFF[emo] || "neutral affect";
    return `${lvl} — ${dis}, with ${aff}.`;
  }

  async function api(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    if (!r.ok) { const t = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(t.detail || r.statusText); }
    return r.json();
  }

  async function health() {
    try {
      const h = await (await fetch("/api/health")).json();
      $("status").textContent = h.key_configured ? `backend ready · child model: ${h.model}` : "⚠️ no OPENAI_API_KEY (set app/.env)";
      $("status").style.color = h.key_configured ? "#7fd8a0" : "#f0c483";
    } catch { $("status").textContent = "⚠️ backend not reachable"; $("status").style.color = "#e88b90"; }
  }

  async function start() {
    stopAuto(); $("summary").style.display = "none";
    try {
      const res = await api("/api/start", { persona: $("personaSelect").value });
      sid = res.session_id;
      $("stepBtn").disabled = false; $("autoBtn").disabled = false;
      $("stepLabel").textContent = "session " + sid + " · step 0/30";
      $("startBtn").textContent = "Restart";
    } catch (e) { alert("Start failed: " + e.message); }
  }

  async function step() {
    if (!sid || busy) return; busy = true;
    try {
      const r = await api("/api/step", { session_id: sid });
      if (r.done && !r.step) return;
      render(r);
      $("stepLabel").textContent = `session ${sid} · step ${r.step}/30`;
      if (r.handback) { stopAuto(); showHandback(); }
      if (r.done) { stopAuto(); await showSummary(); $("stepBtn").disabled = true; $("autoBtn").disabled = true; }
    } catch (e) { stopAuto(); $("status").textContent = "step error: " + e.message; $("status").style.color = "#e88b90"; }
    finally { busy = false; }
  }

  function render(r) {
    const c = r.child;
    $("engBar").style.width = (c.engagement * 100) + "%"; $("engVal").textContent = c.engagement.toFixed(3);
    $("dispEmo").textContent = c.displayed_emotion; $("rawEmo").textContent = c.true_emotion_raw;
    $("emoIcon").textContent = EMO_EMOJI[c.displayed_emotion] || EMO_EMOJI[c.true_emotion_raw] || "😐";
    $("childInterp").textContent = childStateSentence(c.engagement, c.true_emotion_raw || c.displayed_emotion);
    $("fatBar").style.width = (c.fatigue * 100) + "%"; $("fatVal").textContent = c.fatigue.toFixed(3);

    const p = r.perception, tierEl = $("percTier");
    const tier = p.gating_tier || "accept";
    tierEl.textContent = tier; tierEl.className = "badge tier-" + tier;
    $("percConf").textContent = p.confidence != null ? p.confidence.toFixed(3) : "—";
    $("percPred").textContent = p.predicted_class || "—";
    $("percImg").textContent = p.retrieved_img || "—";
    probs($("percProbs"), p.probs, EMO_COL);

    const o = r.obs_9d, tb = $("obsTable"); tb.innerHTML = "";
    [["engagement", o["engagement[SIM]"], "SIM"], ["Δ engagement", o["delta[SIM]"], "SIM"],
     ["emotion probs", "[" + o["emotion_probs[REAL]"].map(x => x.toFixed(2)).join(", ") + "]", "REAL"],
     ["confidence", o["confidence[REAL]"], "REAL"], ["fatigue", o["fatigue[SIM]"], "SIM"], ["τ", o["tau[SIM]"], "SIM"]]
      .forEach(([k, v, tag]) => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${k} <span class="tag ${tag === 'REAL' ? 'real' : 'sim'}">${tag}</span></td><td>${typeof v === 'number' ? v.toFixed(3) : v}</td>`; tb.appendChild(tr); });

    const s = r.safety, sv = $("safetyVerdict");
    $("sacAction").textContent = r.sac_action;
    if (s.intercepted) { sv.textContent = "intercept [" + s.rules.join(",") + "]"; sv.className = "badge intercept"; }
    else { sv.textContent = "allow"; sv.className = "badge allow"; }
    $("r3level").textContent = s.level || "none";
    $("execAction").textContent = s.executed;
    const cf = s.r3_standby; $("r3cf").textContent = cf && cf.aggressive_would_intercept ? `aggressive action here → ${cf.level} [${cf.rules.join(",")}]` : "no rule would fire";
    const execSent = actionToSentence(s.executed);
    let prefix = "";
    if (s.level === "full_conservative") prefix = "Safety forced a fully conservative action → ";
    else if (s.level === "soft_cap") prefix = "Safety toned down stimulation (cautious tier: fast→normal / high→medium) → ";
    $("actionInterp").textContent = prefix + execSent;
    probs($("rewardBars"), r.reward_decomposition, REWARD_COL, true);

    const mb = $("momentBadge");
    let m = "";
    if (s.level === "soft_cap") m = "Cautious → R3 soft-cap";
    else if (s.level === "full_conservative" && (s.rules || []).includes("R3")) m = "Abstain → R3 conservative";
    else if (s.intercepted) m = "Safety interception (R1/R2/R4)";
    else if (r.handback) m = "Hand-back-to-human (simulated)";
    if (m) { mb.textContent = "◆ " + m; mb.style.background = "#2a2740"; mb.style.color = "#b3a8e0"; } else { mb.textContent = ""; mb.style.background = "transparent"; }
    if (r.llm_note) { $("status").textContent = r.llm_note; $("status").style.color = "#f0c483"; }
  }

  function probs(container, obj, colmap, signed) {
    container.innerHTML = "";
    const entries = Array.isArray(obj) ? obj.map((v, i) => [EMO[i], v]) : Object.entries(obj);
    const maxAbs = signed ? Math.max(0.3, ...entries.map(([, v]) => Math.abs(v))) : 1;
    entries.forEach(([k, v]) => {
      const div = document.createElement("div"); div.className = "pb";
      const w = signed ? (Math.abs(v) / maxAbs * 50) : (v * 100);
      const off = signed ? "margin-left:50%;" + (v < 0 ? `transform:translateX(-${w}%);` : "") : "";
      div.innerHTML = `<span class="pbl">${k}</span><div class="pbtrack"><div class="pbfill" style="width:${w}%;background:${colmap[k] || '#4C72B0'};${off}"></div></div><span class="pbv">${typeof v === 'number' ? v.toFixed(3) : v}</span>`;
      container.appendChild(div);
    });
  }

  function showHandback() { $("handbackModal").style.display = "flex"; }
  $("hbContinue").onclick = () => { $("handbackModal").style.display = "none"; };
  $("hbEnd").onclick = async () => { $("handbackModal").style.display = "none"; stopAuto(); await showSummary(); $("stepBtn").disabled = true; $("autoBtn").disabled = true; };

  async function showSummary() {
    try {
      const s = await api("/api/summary", { session_id: sid });
      $("summary").style.display = "block";
      $("summary").innerHTML = `<h3>Session summary — ${s.persona}</h3><div class="cards">
        <div class="card"><div class="n">${s.tier_accept}/${s.tier_cautious}/${s.tier_abstain}</div><div class="l">gating tiers: accept / cautious / abstain</div></div>
        <div class="card"><div class="n">${s.R3_soft_cap}</div><div class="l">R3 soft-cap (cautious)</div></div>
        <div class="card"><div class="n">${s.R3_full_conservative}</div><div class="l">R3 full conservative (abstain)</div></div>
        <div class="card"><div class="n">${s.handback_step || "none"}</div><div class="l">hand-back (simulated)</div></div>
        <div class="card"><div class="n">${s.mean_engagement ?? "—"}</div><div class="l">mean engagement (band ${s.target_band[0]}–${s.target_band[1]})</div></div></div>
        <div class="note">Simulated module-integration demo. 3-level per-class gating. Real perception = Exp2 (imperfect). engagement/fatigue simulated. R3 soft-cap mostly dormant because the trained SAC avoids aggressive actions. Hand-back = simulated pause.</div>`;
    } catch (e) { /* ignore */ }
  }

  function startAuto() { if (auto) return; $("autoBtn").textContent = "❚❚ Pause"; auto = setInterval(() => { if (!busy) step(); }, 1600); }
  function stopAuto() { if (auto) clearInterval(auto); auto = null; $("autoBtn").textContent = "▶ Auto"; }

  $("startBtn").onclick = start;
  $("stepBtn").onclick = step;
  $("autoBtn").onclick = () => auto ? stopAuto() : startAuto();
  health();
})();
