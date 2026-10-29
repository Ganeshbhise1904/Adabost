"""
AdaBoost Predictor - Flask app (Vercel ready)
Put this file, requirements.txt and your model file (Adabost.pkl) in the SAME folder (project root).
"""
import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ----------------------------------------------------------------------------
# 1. Load model (accepts either spelling of the file name)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = None
MODEL_ERROR = None
for _name in ("Adabost.pkl", "adaboost.pkl", "Adaboost.pkl", "AdaBoost.pkl"):
    _path = os.path.join(BASE_DIR, _name)
    if os.path.exists(_path):
        try:
            with open(_path, "rb") as f:
                MODEL = pickle.load(f)
        except Exception as exc:  # noqa
            MODEL_ERROR = str(exc)
        break
else:
    MODEL_ERROR = "Model file not found. Keep Adabost.pkl next to app.py."

FEATURES = list(MODEL.feature_names_in_) if MODEL is not None else []
POS_INDEX = list(MODEL.classes_).index(1) if MODEL is not None else 1

# Change these two words if your target means something else (e.g. "Approved" / "Not approved")
POSITIVE_WORD = "Likely"
NEGATIVE_WORD = "Unlikely"

# ----------------------------------------------------------------------------
# 2. Form definition (order must match the model's training columns)
# ----------------------------------------------------------------------------
US_STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
             "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
             "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
             "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
             "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
             "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
             "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
             "Wisconsin", "Wyoming"]

FIELDS = [
    # name, label, section, kind, min, max, step, default, unit, help
    dict(name="age", label="Age", section="Personal", kind="slider", min=14, max=40, step=1, default=18, unit="yrs", help="Applicant's age."),
    dict(name="gender", label="Gender", section="Personal", kind="select", options=["Male", "Female", "Other"], default=0, help="Encoded as 0/1/2."),
    dict(name="state", label="State", section="Personal", kind="select", options=US_STATES, default=4, help="Encoded as the list position."),
    dict(name="family_income", label="Family income", section="Personal", kind="number", min=0, max=1000000, step=1000, default=60000, unit="$/yr", help="Yearly family income."),

    dict(name="high_school_gpa", label="High school GPA", section="Academics", kind="slider", min=0, max=4, step=0.01, default=3.4, unit="/ 4.0", help="Unweighted GPA on a 4.0 scale."),
    dict(name="sat_score", label="SAT score", section="Academics", kind="slider", min=400, max=1600, step=10, default=1150, unit="/ 1600", help="Total SAT score."),
    dict(name="act_score", label="ACT score", section="Academics", kind="slider", min=1, max=36, step=1, default=25, unit="/ 36", help="Composite ACT score."),
    dict(name="attendance_rate", label="Attendance rate", section="Academics", kind="slider", min=0, max=100, step=1, default=92, unit="%", help="School attendance percentage."),
    dict(name="ap_courses", label="AP courses taken", section="Academics", kind="slider", min=0, max=15, step=1, default=3, unit="", help="Advanced Placement courses."),

    dict(name="extracurricular_count", label="Extracurricular activities", section="Activities", kind="slider", min=0, max=15, step=1, default=4, unit="", help="Clubs, sports, arts etc."),
    dict(name="volunteer_hours", label="Volunteer hours", section="Activities", kind="slider", min=0, max=500, step=5, default=60, unit="hrs", help="Total community service hours."),
    dict(name="leadership_positions", label="Leadership positions", section="Activities", kind="slider", min=0, max=10, step=1, default=1, unit="", help="Captain, president, lead etc."),
    dict(name="coding_projects", label="Coding projects", section="Activities", kind="slider", min=0, max=20, step=1, default=2, unit="", help="Personal / school coding projects."),
    dict(name="social_media_hours", label="Social media hours", section="Activities", kind="slider", min=0, max=12, step=0.5, default=3, unit="hrs/day", help="Average daily usage."),
    dict(name="online_certifications", label="Online certifications", section="Activities", kind="slider", min=0, max=20, step=1, default=2, unit="", help="Completed online certificates."),

    dict(name="essay_score", label="Essay score", section="Application", kind="slider", min=0, max=100, step=1, default=75, unit="/ 100", help="Quality score of the essay."),
    dict(name="recommendation_score", label="Recommendation score", section="Application", kind="slider", min=0, max=100, step=1, default=75, unit="/ 100", help="Strength of recommendation letters."),
]

LABELS = {f["name"]: f["label"] for f in FIELDS}
SECTIONS = []
for _f in FIELDS:
    if _f["section"] not in SECTIONS:
        SECTIONS.append(_f["section"])

# what-if tweaks used for "how to improve" tips: feature -> (step, max value)
IMPROVABLE = {
    "high_school_gpa": (0.3, 4.0), "sat_score": (100, 1600), "act_score": (2, 36),
    "attendance_rate": (5, 100), "ap_courses": (2, 15), "extracurricular_count": (2, 15),
    "volunteer_hours": (30, 500), "leadership_positions": (1, 10), "coding_projects": (2, 20),
    "online_certifications": (2, 20), "essay_score": (10, 100), "recommendation_score": (10, 100),
}


# ----------------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------------
def parse_input(payload):
    values, errors = [], []
    for f in FIELDS:
        raw = payload.get(f["name"])
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{f['label']} must be a number.")
            continue
        if f["kind"] == "select":
            if not 0 <= val < len(f["options"]):
                errors.append(f"{f['label']} is invalid.")
        elif not f["min"] <= val <= f["max"]:
            errors.append(f"{f['label']} must be between {f['min']} and {f['max']}.")
        values.append(val)
    return values, errors


def prob_of(vec):
    return float(MODEL.predict_proba(np.array([vec], dtype=float))[0][POS_INDEX])


def verdict(p):
    if p >= 0.80: return "Very strong chance", "great"
    if p >= 0.65: return "Good chance", "good"
    if p >= 0.50: return "Slightly favourable", "ok"
    if p >= 0.35: return "Slightly below the bar", "warn"
    if p >= 0.20: return "Low chance", "low"
    return "Very low chance", "low"


def explain(vec):
    """Per-feature push (+ / -) from the AdaBoost stumps, as % of the total vote."""
    X = np.array([vec], dtype=float)
    weights = MODEL.estimator_weights_
    total = float(np.sum(weights)) or 1.0
    push = {n: 0.0 for n in FEATURES}
    for est, w in zip(MODEL.estimators_, weights):
        feat_idx = est.tree_.feature[0]
        if feat_idx < 0:
            continue
        vote = 1 if est.predict(X)[0] == MODEL.classes_[POS_INDEX] else -1
        push[FEATURES[feat_idx]] += vote * float(w) / total
    return push


def build_tips(vec, base_p):
    tips = []
    for name, (step, cap) in IMPROVABLE.items():
        i = FEATURES.index(name)
        new = min(vec[i] + step, cap)
        if new <= vec[i]:
            continue
        v2 = list(vec)
        v2[i] = new
        gain = prob_of(v2) - base_p
        if gain > 0.004:
            tips.append((gain, name, new))
    tips.sort(reverse=True)
    out = []
    for gain, name, new in tips[:3]:
        shown = f"{new:.2f}" if name == "high_school_gpa" else f"{new:g}"
        out.append(f"Raising {LABELS[name]} to {shown} could add about {gain * 100:.1f} percentage points.")
    return out


def lower_label(name):
    text = LABELS[name].lower()
    for acr in ("sat", "act", "gpa", "ap"):
        text = text.replace(acr, acr.upper()) if f" {acr} " in f" {text} " else text
    return text


def human_text(p, push):
    title, _ = verdict(p)
    pos = sorted([(v, k) for k, v in push.items() if v > 0.005], reverse=True)[:2]
    neg = sorted([(v, k) for k, v in push.items() if v < -0.005])[:2]
    parts = [f"The model gives this profile a {p * 100:.0f}% chance of a positive result - {title.lower()}."]
    if pos:
        parts.append("It helps most that your " + " and ".join(lower_label(k) for _, k in pos) + (" are" if len(pos) > 1 else " is") + " strong.")
    if neg:
        parts.append("It holds you back that your " + " and ".join(lower_label(k) for _, k in neg) + (" are" if len(neg) > 1 else " is") + " on the weaker side.")
    parts.append("This is a statistical estimate, not a guarantee.")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# 4. Routes
# ----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE, fields=FIELDS, sections=SECTIONS, model_ok=MODEL is not None, error=MODEL_ERROR)


@app.route("/health")
def health():
    return jsonify(status="ok" if MODEL else "model_not_loaded", error=MODEL_ERROR)


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify(error=f"Model not loaded: {MODEL_ERROR}"), 500
    values, errors = parse_input(request.get_json(silent=True) or {})
    if errors:
        return jsonify(error=" ".join(errors)), 400

    p = prob_of(values)
    label = POSITIVE_WORD if p >= 0.5 else NEGATIVE_WORD
    title, tone = verdict(p)
    push = explain(values)
    factors = sorted(
        [{"name": LABELS[k], "impact": round(v * 100, 1)} for k, v in push.items() if abs(v) >= 0.002],
        key=lambda d: abs(d["impact"]), reverse=True)[:8]

    return jsonify(
        probability=round(p * 100, 1), label=label, title=title, tone=tone,
        message=human_text(p, push), factors=factors, tips=build_tips(values, p),
    )


# ----------------------------------------------------------------------------
# 5. UI (single page, light + dark theme)
# ----------------------------------------------------------------------------
PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AdaBoost Predictor</title>
<script>
  (function(){var t=localStorage.getItem('theme');
  if(!t){t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}
  document.documentElement.dataset.theme=t;})();
</script>
<style>
:root{--bg:#f4f6fb;--card:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#4f46e5;--accent2:#06b6d4;
--good:#16a34a;--warn:#f59e0b;--low:#dc2626;--shadow:0 10px 30px rgba(15,23,42,.08);--input:#f8fafc}
[data-theme=dark]{--bg:#0b1020;--card:#131a2e;--text:#e5e9f5;--muted:#94a3b8;--line:#243050;--accent:#818cf8;--accent2:#22d3ee;
--shadow:0 10px 30px rgba(0,0,0,.45);--input:#0e1426}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);transition:background .3s,color .3s}
.bg-glow{position:fixed;inset:0;z-index:-1;background:radial-gradient(600px 300px at 10% -5%,color-mix(in srgb,var(--accent) 25%,transparent),transparent),radial-gradient(500px 300px at 95% 0,color-mix(in srgb,var(--accent2) 20%,transparent),transparent)}
header{display:flex;justify-content:space-between;align-items:center;max-width:1200px;margin:0 auto;padding:22px 20px}
.brand{display:flex;gap:12px;align-items:center;font-weight:700;font-size:20px}
.logo{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;color:#fff;font-size:20px}
.brand small{display:block;font-weight:400;color:var(--muted);font-size:12px}
button{font:inherit;cursor:pointer}
.icon-btn{border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:12px;padding:9px 14px;box-shadow:var(--shadow)}
main{max-width:1200px;margin:0 auto;padding:0 20px 60px;display:grid;grid-template-columns:1.35fr 1fr;gap:22px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:var(--shadow)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px;align-items:center}
.chip{border:1px solid var(--line);background:var(--input);color:var(--text);border-radius:999px;padding:7px 14px;font-size:13px;transition:.2s}
.chip:hover{border-color:var(--accent);color:var(--accent)}
.progress{height:6px;background:var(--line);border-radius:9px;overflow:hidden;margin:12px 0 4px}
.progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .3s}
h3.sec{margin:22px 0 10px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px 18px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.field label{display:flex;justify-content:space-between;font-size:14px;font-weight:600;margin-bottom:6px}
.field label span{color:var(--muted);font-weight:400;font-size:12px}
.row{display:flex;gap:10px;align-items:center}
input[type=range]{flex:1;accent-color:var(--accent);height:6px}
input[type=number],select{width:100%;background:var(--input);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:9px 10px;font:inherit;outline:none}
.row input[type=number]{width:88px}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent)}
.help{font-size:11.5px;color:var(--muted);margin-top:4px}
.primary{width:100%;margin-top:24px;border:0;border-radius:14px;padding:15px;font-weight:700;font-size:16px;color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 20px color-mix(in srgb,var(--accent) 40%,transparent);transition:transform .15s}
.primary:hover{transform:translateY(-2px)}
.primary:disabled{opacity:.6}
.sticky{position:sticky;top:16px;align-self:start;display:flex;flex-direction:column;gap:22px}
.empty{text-align:center;color:var(--muted);padding:30px 10px}
.empty .big{font-size:46px}
.gauge{position:relative;width:190px;height:190px;margin:0 auto}
.gauge svg{transform:rotate(-90deg)}
.gauge .bgc{stroke:var(--line)}
.gauge .fg{transition:stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1),stroke .4s}
.gauge .num{position:absolute;inset:0;display:grid;place-items:center;text-align:center;font-size:40px;font-weight:800}
.gauge .num small{display:block;font-size:12px;font-weight:500;color:var(--muted)}
.badge{display:inline-block;padding:6px 14px;border-radius:999px;font-weight:700;font-size:14px;color:#fff}
.great,.good{--c:var(--good)}.ok{--c:#84cc16}.warn{--c:var(--warn)}.low{--c:var(--low)}
.badge{background:var(--c)}
.center{text-align:center;margin-top:12px}
.msg{margin:14px 0;line-height:1.6;font-size:15px}
.factor{display:grid;grid-template-columns:130px 1fr 52px;gap:8px;align-items:center;font-size:13px;margin:7px 0}
.bar{height:9px;background:var(--line);border-radius:9px;position:relative;overflow:hidden}
.bar i{position:absolute;top:0;bottom:0;border-radius:9px;transition:width .8s}
.pos i{left:50%;background:var(--good)}.neg i{right:50%;background:var(--low)}
.bar:after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--muted);opacity:.5}
.tips li{margin:6px 0;line-height:1.5}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.hist{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px dashed var(--line);font-size:13px}
.hist:last-child{border:0}
.err{background:color-mix(in srgb,var(--low) 14%,transparent);color:var(--low);padding:10px 14px;border-radius:12px;margin-top:14px;display:none}
.fade{animation:fade .5s ease}
@keyframes fade{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.spin{display:inline-block;width:16px;height:16px;border:2px solid #fff6;border-top-color:#fff;border-radius:50%;animation:s .7s linear infinite;vertical-align:-3px;margin-right:8px}
@keyframes s{to{transform:rotate(360deg)}}
footer{text-align:center;color:var(--muted);font-size:12px;padding-bottom:30px}
@media print{header,.toolbar,.primary,.actions,#historyCard,.bg-glow{display:none}main{display:block}}
</style>
</head>
<body>
<div class="bg-glow"></div>
<header>
  <div class="brand"><div class="logo">&#9889;</div><div>AdaBoost Predictor<small>Machine-learning outcome estimator</small></div></div>
  <button class="icon-btn" id="themeBtn" aria-label="Toggle theme">&#127769; Theme</button>
</header>

<main>
  <section class="card">
    <div class="toolbar">
      <button class="chip" data-preset="strong">&#127942; Strong profile</button>
      <button class="chip" data-preset="average">&#9878;&#65039; Average profile</button>
      <button class="chip" data-preset="weak">&#128201; Weak profile</button>
      <button class="chip" id="randomBtn">&#127922; Random</button>
      <button class="chip" id="resetBtn">&#8634; Reset</button>
    </div>
    <div class="progress"><i id="prog"></i></div>
    <div class="help">Move the sliders or type exact values. Prediction updates when you press Predict.</div>

    <form id="form" autocomplete="off">
    {% for sec in sections %}
      <h3 class="sec">{{ sec }}</h3>
      <div class="grid">
      {% for f in fields if f.section == sec %}
        <div class="field">
          <label for="{{ f.name }}">{{ f.label }} <span>{{ f.unit or '' }}</span></label>
          {% if f.kind == 'select' %}
            <select id="{{ f.name }}" name="{{ f.name }}" data-default="{{ f.default }}">
              {% for o in f.options %}<option value="{{ loop.index0 }}" {% if loop.index0 == f.default %}selected{% endif %}>{{ o }}</option>{% endfor %}
            </select>
          {% elif f.kind == 'slider' %}
            <div class="row">
              <input type="range" id="{{ f.name }}_r" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
              <input type="number" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}" data-default="{{ f.default }}">
            </div>
          {% else %}
            <input type="number" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}" data-default="{{ f.default }}">
          {% endif %}
          <div class="help">{{ f.help }}</div>
        </div>
      {% endfor %}
      </div>
    {% endfor %}
      <div class="err" id="err"></div>
      <button class="primary" id="go" type="submit" {% if not model_ok %}disabled{% endif %}>Predict result</button>
      {% if not model_ok %}<div class="err" style="display:block">{{ error }}</div>{% endif %}
    </form>
  </section>

  <aside class="sticky">
    <section class="card" id="resultCard">
      <div class="empty" id="empty"><div class="big">&#128302;</div><h3>Your result appears here</h3><p>Fill in the details and press <b>Predict result</b>.</p></div>
      <div id="result" style="display:none">
        <div class="gauge">
          <svg width="190" height="190" viewBox="0 0 190 190">
            <circle class="bgc" cx="95" cy="95" r="82" fill="none" stroke-width="14"/>
            <circle class="fg" id="arc" cx="95" cy="95" r="82" fill="none" stroke-width="14" stroke-linecap="round" stroke-dasharray="515.2" stroke-dashoffset="515.2"/>
          </svg>
          <div class="num"><div><span id="pct">0</span>%<small>chance</small></div></div>
        </div>
        <div class="center"><span class="badge" id="badge"></span></div>
        <p class="msg" id="msg"></p>
        <h3 class="sec" style="margin-top:8px">What influenced it</h3>
        <div id="factors"></div>
        <div id="tipsBox"><h3 class="sec">How to improve</h3><ul class="tips" id="tips"></ul></div>
        <div class="actions">
          <button class="chip" id="copyBtn">&#128203; Copy summary</button>
          <button class="chip" onclick="window.print()">&#128424; Print</button>
        </div>
      </div>
    </section>
    <section class="card" id="historyCard">
      <div style="display:flex;justify-content:space-between;align-items:center"><b>Recent predictions</b><button class="chip" id="clearHist">Clear</button></div>
      <div id="history"><div class="help" style="margin-top:10px">Nothing yet.</div></div>
    </section>
  </aside>
</main>
<footer>Predictions are statistical estimates and not guaranteed outcomes.</footer>

<script>
const $ = s => document.querySelector(s);
const form = $('#form');
const nums = [...form.querySelectorAll('input[type=number],select')];

// theme toggle
function setTheme(t){document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);$('#themeBtn').innerHTML=t==='dark'?'&#9728;&#65039; Light':'&#127769; Dark'}
setTheme(document.documentElement.dataset.theme);
$('#themeBtn').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');

// slider <-> number sync
nums.forEach(n=>{
  const r=$('#'+n.id+'_r');
  if(r){r.oninput=()=>{n.value=r.value;progress()};n.oninput=()=>{r.value=n.value;progress()}}
  else n.oninput=progress;
});
function progress(){
  const ok=nums.filter(n=>n.value!=='').length;
  $('#prog').style.width=(ok/nums.length*100)+'%';
}
progress();
function setVal(id,v){const n=$('#'+id);n.value=v;const r=$('#'+id+'_r');if(r)r.value=v}

// presets
const P={
 strong:{age:18,high_school_gpa:3.9,sat_score:1500,act_score:34,attendance_rate:98,ap_courses:8,extracurricular_count:6,volunteer_hours:120,leadership_positions:3,coding_projects:5,social_media_hours:1.5,online_certifications:5,essay_score:90,recommendation_score:90},
 average:{age:18,high_school_gpa:3.3,sat_score:1150,act_score:25,attendance_rate:92,ap_courses:3,extracurricular_count:4,volunteer_hours:60,leadership_positions:1,coding_projects:2,social_media_hours:3,online_certifications:2,essay_score:72,recommendation_score:72},
 weak:{age:18,high_school_gpa:2.2,sat_score:850,act_score:16,attendance_rate:75,ap_courses:0,extracurricular_count:1,volunteer_hours:10,leadership_positions:0,coding_projects:0,social_media_hours:7,online_certifications:0,essay_score:45,recommendation_score:45}};
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{Object.entries(P[b.dataset.preset]).forEach(([k,v])=>setVal(k,v));progress()});
$('#resetBtn').onclick=()=>{nums.forEach(n=>setVal(n.id,n.dataset.default));progress();$('#err').style.display='none'};
$('#randomBtn').onclick=()=>{nums.forEach(n=>{
  if(n.tagName==='SELECT'){n.value=Math.floor(Math.random()*n.options.length);return}
  if(n.id==='family_income'){n.value=Math.round((20000+Math.random()*180000)/1000)*1000;return}
  const mn=+n.min,mx=+n.max,st=+n.step||1;const v=mn+Math.round(Math.random()*(mx-mn)/st)*st;setVal(n.id,+v.toFixed(2))});progress()};

// predict
form.onsubmit=async e=>{
  e.preventDefault();
  const body={};nums.forEach(n=>body[n.name]=n.value);
  const btn=$('#go'),err=$('#err');err.style.display='none';
  btn.disabled=true;btn.innerHTML='<span class="spin"></span>Analysing...';
  try{
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await res.json();
    if(!res.ok)throw new Error(d.error||'Something went wrong');
    show(d);saveHistory(d);
  }catch(x){err.textContent=x.message;err.style.display='block'}
  btn.disabled=false;btn.textContent='Predict result';
};

function show(d){
  $('#empty').style.display='none';const r=$('#result');r.style.display='block';r.classList.remove('fade');void r.offsetWidth;r.classList.add('fade');
  const C=515.2,arc=$('#arc');
  const color={great:'#16a34a',good:'#16a34a',ok:'#84cc16',warn:'#f59e0b',low:'#dc2626'}[d.tone];
  arc.style.stroke=color;arc.style.strokeDashoffset=C;
  requestAnimationFrame(()=>requestAnimationFrame(()=>arc.style.strokeDashoffset=C*(1-d.probability/100)));
  // count-up
  let cur=0;const t=setInterval(()=>{cur+=Math.max(.5,d.probability/40);if(cur>=d.probability){cur=d.probability;clearInterval(t)}$('#pct').textContent=cur.toFixed(0)},20);
  const b=$('#badge');b.className='badge '+d.tone;b.textContent=d.label+' - '+d.title;
  $('#msg').textContent=d.message;
  $('#factors').innerHTML=d.factors.map(f=>{const w=Math.min(50,Math.abs(f.impact)*1.6);
    return `<div class="factor"><span>${f.name}</span><div class="bar ${f.impact>=0?'pos':'neg'}"><i style="width:${w}%"></i></div><b style="color:var(${f.impact>=0?'--good':'--low'})">${f.impact>0?'+':''}${f.impact}</b></div>`}).join('')||'<div class="help">No single factor dominated.</div>';
  $('#tips').innerHTML=d.tips.map(t=>`<li>${t}</li>`).join('');
  $('#tipsBox').style.display=d.tips.length?'block':'none';
  window._last=d;
  if(innerWidth<900)r.scrollIntoView({behavior:'smooth'});
}
$('#copyBtn').onclick=async()=>{const d=window._last;await navigator.clipboard.writeText(`Result: ${d.label} (${d.probability}%) - ${d.title}\n${d.message}`);$('#copyBtn').textContent='Copied!';setTimeout(()=>$('#copyBtn').innerHTML='&#128203; Copy summary',1500)};

// history
function getH(){try{return JSON.parse(localStorage.getItem('hist')||'[]')}catch(e){return[]}}
function saveHistory(d){const h=getH();h.unshift({p:d.probability,l:d.label,t:d.title,at:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})});localStorage.setItem('hist',JSON.stringify(h.slice(0,5)));renderH()}
function renderH(){const h=getH();$('#history').innerHTML=h.length?h.map(x=>`<div class="hist"><span>${x.at} &middot; ${x.t}</span><b>${x.p}%</b></div>`).join(''):'<div class="help" style="margin-top:10px">Nothing yet.</div>'}
$('#clearHist').onclick=()=>{localStorage.removeItem('hist');renderH()};
renderH();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)