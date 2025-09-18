\
import os, tarfile, tempfile, logging, json, math, re
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

MOUNT_DIR = os.getenv("MODEL_MOUNT_DIR", "/mnt/models")
ENV_MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
DEFAULT_CANDIDATES = ["model.pkl","model.pkl.tgz","cpiflowmodel","cpiflowmodel.tgz","cpimodel","cpimodel.tgz"]
LABELS_DIR = os.getenv("LABELS_DIR", "/mnt/labels")
os.makedirs(LABELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("cpiflow-serve")

def _is_archive(p: str) -> bool:
    p = p.lower()
    return p.endswith((".tgz",".tar.gz",".tar"))

def _listdir(path: str):
    try: return sorted(os.listdir(path))
    except: return []

def _load_archive(path: str):
    log.info(f"Loading model from archive: {path}")
    with tarfile.open(path, "r:*") as tar, tempfile.TemporaryDirectory() as tmp:
        member = next((m for m in tar.getmembers() if m.name.lower().endswith(".pkl")), None)
        if not member: raise FileNotFoundError(f"No .pkl in archive: {path}")
        with tar.extractfile(member) as fi, open(os.path.join(tmp, os.path.basename(member.name)), "wb") as fo:
            fo.write(fi.read())
            return joblib.load(fo.name)

def _resolve_env_path() -> Optional[str]:
    if not ENV_MODEL_PATH: return None
    p = ENV_MODEL_PATH if os.path.isabs(ENV_MODEL_PATH) else os.path.join(MOUNT_DIR, ENV_MODEL_PATH)
    return p if os.path.exists(p) else None

def _load_pipeline():
    envp = _resolve_env_path()
    if envp:
        if os.path.isdir(envp):
            pkls = [f for f in _listdir(envp) if f.lower().endswith(".pkl")]
            if pkls: return joblib.load(os.path.join(envp, pkls[0]))
        if _is_archive(envp): return _load_archive(envp)
        return joblib.load(envp)

    for name in DEFAULT_CANDIDATES:
        p = os.path.join(MOUNT_DIR, name)
        if os.path.exists(p) and not os.path.isdir(p):
            return _load_archive(p) if _is_archive(p) else joblib.load(p)

    if os.path.isdir(MOUNT_DIR):
        pkls = [f for f in _listdir(MOUNT_DIR) if f.lower().endswith(".pkl")]
        if pkls: return joblib.load(os.path.join(MOUNT_DIR, pkls[0]))
        archives = [f for f in _listdir(MOUNT_DIR) if _is_archive(f)]
        if archives: return _load_archive(os.path.join(MOUNT_DIR, archives[0]))

    raise FileNotFoundError(f"Model not found in {MOUNT_DIR}. Contents: { _listdir(MOUNT_DIR) }")

pipe = _load_pipeline()
META = getattr(pipe, "meta", {}) or {}
ART_KPIS: Dict[str, Dict[str, Any]] = META.get("artifact_kpis", {})
GLOBAL_HOTSPOTS = META.get("hotspots_top10", [])
BASELINE_FREQS: Dict[str, Dict[str,float]] = META.get("baseline_freqs", {})
BASELINE_TARGET: Dict[str,float] = META.get("baseline_target", {})
DAILY_SERIES: List[Dict[str,Any]] = META.get("daily_error_series", [])

app = FastAPI(title="CPI Logs Classifier (cpiflow-pro)", version="3.0")

# --------- Schemas ---------
class Instance(BaseModel):
    ARTIFACT_NAME: Optional[str] = ""
    ORIGIN_COMPONENT_NAME: Optional[str] = ""
    LOG_LEVEL: Optional[str] = ""
    MESSAGE: Optional[str] = None

class PredictRequest(BaseModel):
    instances: List[Instance]

class AnalyzeRequest(BaseModel):
    artifact_name: str
    top_k: Optional[int] = 4

class AnalyzeManyRequest(BaseModel):
    artifact_names: List[str]
    top_k: Optional[int] = 4

class DriftRequest(BaseModel):
    batch: List[Instance] = Field(...)
    targets: Optional[List[str]] = None

class SignaturesRequest(BaseModel):
    rows: List[Instance]
    top_k: Optional[int] = 10

class ForecastRequest(BaseModel):
    horizon_days: Optional[int] = 7
    artifact_name: Optional[str] = None

class ExplainRequest(BaseModel):
    instance: Instance
    top_k: Optional[int] = 6

class PrioritizeRequest(BaseModel):
    top_k: Optional[int] = 10

class LabeledRow(BaseModel):
    features: Instance
    label: str

class LabelBatch(BaseModel):
    items: List[LabeledRow]

# --------- Helpers ---------
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

def _clf():
    c = pipe.named_steps.get("clf")
    if not hasattr(c, "predict_proba"):
        raise RuntimeError("Model does not support probabilities.")
    return c

def _vec():
    v = pipe.named_steps.get("vec")
    if v is None:
        raise RuntimeError("DictVectorizer missing in pipeline.")
    return v

def _predict_proba_for(record: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    clf = _clf()
    v = _vec()
    X = v.transform([record])
    probs = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    return classes, probs

def _is_known_artifact(name: str) -> bool:
    key = f"ARTIFACT_NAME={name}"
    return key in getattr(_vec(), "vocabulary_", {})

def _known_artifacts() -> List[str]:
    vocab = getattr(_vec(), "vocabulary_", {}) or {}
    prefix = "ARTIFACT_NAME="
    names = [k[len(prefix):] for k in vocab.keys() if k.startswith(prefix)]
    names.sort()
    return names

def _psi(expected: Dict[str,float], actual: Dict[str,float], eps: float=1e-8) -> float:
    keys = set(expected) | set(actual)
    psi = 0.0
    for k in keys:
        e = expected.get(k, eps)
        a = actual.get(k, eps)
        psi += (a - e) * math.log((a + eps) / (e + eps))
    return float(psi)

def _freqs(records: List[Instance], field: str) -> Dict[str,float]:
    vals = [str(getattr(r, field) or "") for r in records]
    total = max(1, len(vals))
    c = Counter(vals)
    return {k: v/total for k,v in c.items()}

def _tokenize(text: str):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]

def _analyze_one(name: str, top_k: int = 4) -> Dict[str, Any]:
    rec = {"ARTIFACT_NAME": name, "ORIGIN_COMPONENT_NAME": "", "LOG_LEVEL": ""}
    classes, probs = _predict_proba_for(rec)
    items = sorted(zip(classes, probs), key=lambda t: t[1], reverse=True)
    top_k = max(1, min(top_k, len(items)))
    top = [{"label": str(lbl), "prob": round(float(p), 4)} for lbl, p in items[:top_k]]
    out = {
        "artifact": name,
        "known_artifact": _is_known_artifact(name),
        "prediction": top[0]["label"],
        "top_probs": top
    }
    if ART_KPIS.get(name):
        out["artifact_kpis"] = ART_KPIS[name]
    return out

def _feature_contributions(rec: Dict[str,str], k: int=6) -> List[Dict[str,Any]]:
    clf = _clf()
    v = _vec()
    X = v.transform([rec])
    classes = list(clf.classes_)
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return []
    w = coef[pred_idx]
    idxs = X.nonzero()[1]
    names = getattr(v, "feature_names_", None)
    if names is None and hasattr(v, "get_feature_names_out"):
        names = v.get_feature_names_out()
    out = [{"feature": str(names[j]), "weight": float(w[j])} for j in idxs]
    out.sort(key=lambda d: abs(d["weight"]), reverse=True)
    return out[:k]

def _risk_score(kpi: Dict[str,Any]) -> float:
    events = max(1, float(kpi.get("events", 0)))
    er = float(kpi.get("error_rate", 0.0))
    return float(er * math.log1p(events) * (1.0 + float(kpi.get("top_error_share", 0.0))))

# --------- Routes ---------
@app.get("/greet")
@app.get("/v2/greet")
def greet():
    return {
        "status": "ok",
        "mount_dir": MOUNT_DIR,
        "env_model_path": ENV_MODEL_PATH or None,
        "dir_listing": _listdir(MOUNT_DIR),
        "has_meta": bool(ART_KPIS or GLOBAL_HOTSPOTS),
        "endpoints": ["/greet","/analyze","/analyze_many","/analyze_all","/drift","/signatures","/forecast","/explain","/prioritize","/label"]
    }

@app.post("/analyze")
@app.post("/v2/analyze")
def analyze(req: AnalyzeRequest):
    name = (req.artifact_name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="artifact_name must be non-empty")
    resp = _analyze_one(name, top_k=req.top_k or 4)
    if GLOBAL_HOTSPOTS:
        resp["global_hotspots"] = GLOBAL_HOTSPOTS
    return resp

@app.get("/analyze_all")
@app.get("/v2/analyze_all")
def analyze_all(
    top_k: int = Query(4, ge=1, le=10),
    sort: str = Query("alpha", pattern="^(alpha|error_rate)$")
):
    names = _known_artifacts()
    results = [_analyze_one(n, top_k=top_k) for n in names]
    if sort == "error_rate" and ART_KPIS:
        results.sort(key=lambda r: ART_KPIS.get(r["artifact"], {}).get("error_rate", 0), reverse=True)
    else:
        results.sort(key=lambda r: r["artifact"])
    resp = {"count": len(results), "results": results}
    if GLOBAL_HOTSPOTS:
        resp["global_hotspots"] = GLOBAL_HOTSPOTS
    return resp

@app.post("/analyze_many")
@app.post("/v2/analyze_many")
def analyze_many(req: AnalyzeManyRequest):
    names = [n.strip() for n in (req.artifact_names or []) if n and n.strip()]
    if not names:
        raise HTTPException(status_code=422, detail="artifact_names must be a non-empty list")
    results = [_analyze_one(n, top_k=req.top_k or 4) for n in names]
    return { "count": len(results), "results": results }

@app.post("/drift")
def drift(req: DriftRequest):
    if not req.batch:
        raise HTTPException(status_code=422, detail="batch must be a non-empty list")
    drift_scores = {}
    for col in ("ARTIFACT_NAME","ORIGIN_COMPONENT_NAME","LOG_LEVEL"):
        if col in BASELINE_FREQS:
            cur = _freqs(req.batch, col)
            keys = set(BASELINE_FREQS[col]) | set(cur)
            psi = 0.0
            eps = 1e-8
            for k in keys:
                e = BASELINE_FREQS[col].get(k, eps)
                a = cur.get(k, eps)
                psi += (a - e) * math.log((a + eps) / (e + eps))
            drift_scores[col] = round(float(psi), 4)
    target_drift = None
    if req.targets and BASELINE_TARGET:
        total = len(req.targets)
        c = Counter([str(t) for t in req.targets])
        cur = {k: v/total for k,v in c.items()}
        keys = set(BASELINE_TARGET) | set(cur)
        psi = 0.0
        eps = 1e-8
        for k in keys:
            e = BASELINE_TARGET.get(k, eps)
            a = cur.get(k, eps)
            psi += (a - e) * math.log((a + eps) / (e + eps))
        target_drift = round(float(psi), 4)
    recommend_retrain = any(v > 0.2 for v in drift_scores.values()) or (target_drift is not None and target_drift > 0.1)
    return {"drift_scores": drift_scores, "target_drift": target_drift, "recommend_retrain": recommend_retrain}

@app.post("/signatures")
def signatures(req: SignaturesRequest):
    if not req.rows:
        raise HTTPException(status_code=422, detail="rows must be a non-empty list")
    tokens = []
    for r in req.rows:
        if r.MESSAGE:
            toks = _tokenize(r.MESSAGE)
            tokens.extend([" ".join(toks[i:i+2]) for i in range(len(toks)-1)])
        else:
            combo = f"{r.ORIGIN_COMPONENT_NAME}|{r.LOG_LEVEL}".strip("|")
            if combo:
                tokens.append(combo)
    cnt = Counter(tokens)
    top_k = req.top_k or 10
    top = [{"signature": s, "count": int(n)} for s,n in cnt.most_common(top_k)]
    return {"count": len(tokens), "top_signatures": top}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    horizon = max(1, int(req.horizon_days or 7))
    if DAILY_SERIES and len(DAILY_SERIES) >= 3:
        vals = [float(x["error_rate"]) for x in DAILY_SERIES[-14:]]
        trend = float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0.0
        trend *= 0.3
        out = []
        cur = vals[-1]
        for d in range(1, horizon+1):
            cur = max(0.0, min(1.0, cur + trend*0.5))
            out.append({"day": d, "forecast_error_rate": round(cur, 4)})
        return {"method":"naive-damped","history_days": len(vals), "forecast": out}
    else:
        if ART_KPIS:
            tot_events = sum(v.get("events",0) for v in ART_KPIS.values())
            weighted_er = sum((v.get("error_rate",0.0)*v.get("events",0)) for v in ART_KPIS.values()) / max(1, tot_events)
        else:
            weighted_er = 0.1
        out = [{"day": d, "forecast_error_rate": round(float(weighted_er),4)} for d in range(1, horizon+1)]
        return {"method":"flat-baseline","forecast": out}

@app.post("/explain")
def explain(req: ExplainRequest):
    inst = req.instance
    rec = {"ARTIFACT_NAME": inst.ARTIFACT_NAME or "",
           "ORIGIN_COMPONENT_NAME": inst.ORIGIN_COMPONENT_NAME or "",
           "LOG_LEVEL": inst.LOG_LEVEL or ""}
    classes, probs = _predict_proba_for(rec)
    pred_idx = int(np.argmax(probs))
    contributions = _feature_contributions(rec, k=req.top_k or 6)
    return {
        "prediction": str(classes[pred_idx]),
        "prob": round(float(probs[pred_idx]),4),
        "top_feature_contributions": contributions
    }

@app.post("/prioritize")
def prioritize(req: PrioritizeRequest):
    k = max(1, int(req.top_k or 10))
    items = []
    for name, kpi in ART_KPIS.items():
        rs = _risk_score(kpi)
        d = {"artifact": name, "risk_score": round(rs,4)}
        d.update(kpi)
        items.append(d)
    items.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"count": len(items[:k]), "results": items[:k]}

@app.post("/label")
def label(batch: LabelBatch):
    if not batch.items:
        raise HTTPException(status_code=422, detail="items must be non-empty")
    path = os.path.join(LABELS_DIR, "labels.jsonl")
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for item in batch.items:
            row = {"features": item.features.model_dump(), "label": item.label}
            f.write(json.dumps(row) + "\\n")
            n += 1
    return {"stored": n, "file": path}

# Back-compat
@app.post("/v2/predict")
def predict_v2(req: PredictRequest):
    if not req.instances:
        raise HTTPException(status_code=422, detail="instances must be a non-empty list")
    records = [{
        "ARTIFACT_NAME": (x.ARTIFACT_NAME or ""),
        "ORIGIN_COMPONENT_NAME": (x.ORIGIN_COMPONENT_NAME or ""),
        "LOG_LEVEL": (x.LOG_LEVEL or ""),
    } for x in req.instances]
    preds = pipe.predict(records)
    return {"predictions": [str(p) for p in preds]}
