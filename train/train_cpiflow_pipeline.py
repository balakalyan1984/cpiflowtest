import os, json, pathlib, pandas as pd, joblib, sys
from typing import List, Dict, Any
from dateutil import parser as dateparser

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- Config ---
DATA_PATH        = os.getenv("DATA_PATH", "/app/data/cpi_logs_500.csv")
TARGET_COLUMN    = os.getenv("TARGET_COLUMN", "CUSTOM_STATUS")
MODEL_DIR        = os.getenv("MODEL_DIR", "/app/model")
MODEL_PATH       = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))
TRAINER          = os.getenv("TRAINER", "LR").upper()           # "LR" or "SGD"
FINETUNE_MODE    = os.getenv("FINETUNE_MODE", "0") == "1"
PREV_MODEL_PATH  = os.getenv("PREV_MODEL_PATH", "/app/prev/prev_model.pkl")
CLASS_WEIGHT     = os.getenv("CLASS_WEIGHT", "balanced")
ALPHA            = float(os.getenv("ALPHA", "0.0001"))
EPOCHS           = int(os.getenv("EPOCHS", "1"))
RANDOM_STATE     = 42

EXTRA_DATA_PATHS = os.getenv("EXTRA_DATA_PATHS", "")
CONCAT_MODE      = os.getenv("CONCAT_MODE", "append")

INSIGHTS_CSV  = os.path.join(MODEL_DIR, "cpilog_artifact_insights.csv")
HOTSPOTS_CSV  = os.path.join(MODEL_DIR, "cpilog_hotspots.csv")
REPORT_MD     = os.path.join(MODEL_DIR, "cpilog_insights_report.md")
METRICS_JSON  = os.path.join(MODEL_DIR, "metrics.json")

ART_COL = "ARTIFACT_NAME"
COMP_COL = "ORIGIN_COMPONENT_NAME"
LVL_COL  = "LOG_LEVEL"

pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

def _read_all() -> pd.DataFrame:
    paths = [DATA_PATH] + [p.strip() for p in (EXTRA_DATA_PATHS.split(",") if EXTRA_DATA_PATHS else []) if p.strip()]
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"[warn] data file not found: {p}", file=sys.stderr)
    if not frames:
        raise ValueError("No training data found (DATA_PATH/EXTRA_DATA_PATHS).")
    if CONCAT_MODE.lower() == "replace":
        return frames[-1]
    return pd.concat(frames, ignore_index=True)

df = _read_all()

feat_candidates = [ART_COL, COMP_COL, LVL_COL]
feat_cols = [c for c in feat_candidates if c in df.columns]
if not feat_cols:
    raise ValueError(f"Missing feature columns; expected any of {feat_candidates}")
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Missing target column '{TARGET_COLUMN}'")

X_dict = df[feat_cols].fillna("").astype(str).to_dict(orient="records")
y = df[TARGET_COLUMN].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_dict, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(y.unique())>1 else None
)

def _make_lr() -> Pipeline:
    vec = DictVectorizer(sparse=True)
    clf = LogisticRegression(max_iter=500, solver="saga", multi_class="multinomial", random_state=RANDOM_STATE)
    return Pipeline([("vec", vec), ("clf", clf)])

def _make_sgd() -> Pipeline:
    vec = DictVectorizer(sparse=True)
    cw = None if CLASS_WEIGHT.lower()=="none" else "balanced"
    clf = SGDClassifier(loss="log_loss", alpha=ALPHA, class_weight=cw, random_state=RANDOM_STATE)
    return Pipeline([("vec", vec), ("clf", clf)])

def _fit_fresh(pipe: Pipeline, X_dict: List[Dict[str,Any]], y):
    if isinstance(pipe.named_steps["clf"], SGDClassifier):
        X = pipe.named_steps["vec"].fit_transform(X_dict)
        pipe.named_steps["clf"].partial_fit(X, y, classes=sorted(pd.Series(y).unique()))
    else:
        pipe.fit(X_dict, y)
    return pipe

def _continue_with_prev(prev_model_path: str, X_dict_new: List[Dict[str,Any]], y_new):
    old = joblib.load(prev_model_path)
    vec_prev = old.named_steps.get("vec")
    clf_prev = old.named_steps.get("clf")
    from sklearn.linear_model import SGDClassifier
    if isinstance(clf_prev, SGDClassifier):
        pipe = old
    else:
        cw = None if CLASS_WEIGHT.lower()=="none" else "balanced"
        pipe = Pipeline([("vec", vec_prev),
                         ("clf", SGDClassifier(loss="log_loss", alpha=ALPHA, class_weight=cw, random_state=RANDOM_STATE))])
    V = pipe.named_steps["vec"]
    if hasattr(V, "vocabulary_") and V.vocabulary_:
        X = V.transform(X_dict_new)
    else:
        X = V.fit_transform(X_dict_new)
    clf = pipe.named_steps["clf"]
    classes = getattr(clf, "classes_", None) or sorted(pd.Series(y_new).unique())
    for _ in range(max(1, EPOCHS)):
        clf.partial_fit(X, y_new, classes=classes)
    return pipe

if FINETUNE_MODE and os.path.exists(PREV_MODEL_PATH):
    print(f"[info] Fine-tuning from previous model: {PREV_MODEL_PATH}")
    pipe = _continue_with_prev(PREV_MODEL_PATH, X_tr, y_tr)
else:
    if FINETUNE_MODE:
        print("[warn] FINETUNE_MODE=1 but PREV_MODEL_PATH missing. Doing fresh fit.")
    pipe = _make_sgd() if TRAINER=="SGD" else _make_lr()
    pipe = _fit_fresh(pipe, X_tr, y_tr)

# Eval
def _predict_any(pipe, recs):
    try:
        return pipe.predict(recs)
    except Exception:
        V = pipe.named_steps["vec"]
        X = V.transform(recs) if hasattr(V,"transform") else V.fit_transform(recs)
        return pipe.named_steps["clf"].predict(X)

y_pred = _predict_any(pipe, X_te)
acc = float(accuracy_score(y_te, y_pred))
print("Accuracy:", acc)
print(classification_report(y_te, y_pred, zero_division=0))

# Insights
def build_artifact_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    total = frame.groupby(ART_COL).size().rename("events")
    status = frame.pivot_table(index=ART_COL, columns=TARGET_COLUMN, values=LVL_COL,
                               aggfunc="count", fill_value=0)
    success = status.get("SUCCESS", pd.Series(0, index=total.index)).rename("success")
    errors  = (status.sum(axis=1) - success).rename("errors")
    out = pd.concat([total, success, errors], axis=1)
    out["success_rate"] = (out["success"] / out["events"]).fillna(0).round(3)
    out["error_rate"]   = (out["errors"]  / out["events"]).fillna(0).round(3)
    err_cols = [c for c in status.columns if c != "SUCCESS"]
    if err_cols:
        out["top_error_type"]  = status[err_cols].idxmax(axis=1)
        out["top_error_share"] = (status[err_cols].max(axis=1) / out["events"]).fillna(0).round(3)
    else:
        out["top_error_type"] = ""
        out["top_error_share"] = 0.0

    errs = frame[frame[TARGET_COLUMN] != "SUCCESS"]
    if not errs.empty:
        comp_lvl = (errs.groupby([ART_COL, COMP_COL, LVL_COL]).size()
                    .rename("count").reset_index())
        top_pairs = (comp_lvl.sort_values([ART_COL, "count"], ascending=[True, False])
                            .groupby(ART_COL).head(2))
        pair_str = (top_pairs
            .assign(pair=lambda x: x[COMP_COL] + "|" + x[LVL_COL] + " (n=" + x["count"].astype(str) + ")")
            .groupby(ART_COL)["pair"].apply(lambda s: "; ".join(s.tolist())))
        out["top_err_components"] = out.index.map(pair_str.to_dict()).fillna("")
    else:
        out["top_err_components"] = ""
    return out.sort_values(["error_rate","events"], ascending=[False, False]).reset_index()

artifact_kpis = build_artifact_kpis(df)

# Global hotspots
hotspots = (df[df[TARGET_COLUMN]!="SUCCESS"]
            .groupby([COMP_COL, LVL_COL]).size()
            .sort_values(ascending=False).reset_index(name="count"))

# Baseline distributions for drift
def _probs(s):
    c = s.astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k,v in c.items()}

baseline_freqs = {}
for col in [ART_COL, COMP_COL, LVL_COL]:
    if col in df.columns:
        baseline_freqs[col] = _probs(df[col])
baseline_target = _probs(df[TARGET_COLUMN])

# Optional daily error series if time column exists
daily_series = []
time_cols = [c for c in df.columns if c.lower() in ("timestamp","date","datetime","log_time")]
if time_cols:
    tcol = time_cols[0]
    def _to_date(x):
        try: return dateparser.parse(str(x)).date()
        except: return None
    tmp = df[[tcol, TARGET_COLUMN]].copy()
    tmp["__d"] = tmp[tcol].map(_to_date)
    tmp = tmp.dropna(subset=["__d"])
    if not tmp.empty:
        g = tmp.groupby("__d")
        total = g.size()
        err = g.apply(lambda s: (s[TARGET_COLUMN]!="SUCCESS").sum())
        ser = (err/total).fillna(0)
        daily_series = [{"date": str(k), "error_rate": float(v)} for k,v in ser.items()]

# Attach meta
try:
    pipe.meta = {
        "artifact_kpis": artifact_kpis.set_index("ARTIFACT_NAME")[
            ["events","success_rate","error_rate","top_error_type","top_err_components"]
        ].to_dict(orient="index"),
        "hotspots_top10": hotspots.head(10).to_dict(orient="records"),
        "baseline_freqs": baseline_freqs,
        "baseline_target": baseline_target,
        "daily_error_series": daily_series
    }
except Exception as e:
    print(f"[warn] could not attach meta: {e}", file=sys.stderr)

# Save artifacts
joblib.dump(pipe, MODEL_PATH)
with open(METRICS_JSON, "w") as f:
    json.dump({"accuracy": acc, "features": feat_cols, "target": TARGET_COLUMN,
               "trainer": TRAINER, "finetune_mode": FINETUNE_MODE}, f)
artifact_kpis.to_csv(INSIGHTS_CSV, index=False)
hotspots.to_csv(HOTSPOTS_CSV, index=False)
with open(REPORT_MD, "w") as f:
    f.write("\\n".join([
        "# CPI Log Insights",
        f"- Source: {os.path.basename(DATA_PATH)}",
        f"- Total rows: {len(df)}",
        "## Top 3 artifacts by error rate:"
    ]))
print(f"[info] Saved model → {MODEL_PATH}")
