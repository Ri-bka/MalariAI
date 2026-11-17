"""MalariAI dashboard (single file).  Run with:  streamlit run dashboard.py
Tabs: Overview (national picture) | Screen a smear (model + save to database) | Validation (AI vs microscopist)."""
import random
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).parent
DB = BASE / "malariai_pilot.db"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
MODELS_DIR = BASE / "models"      # put your trained model (best.pt) in this folder; the newest .pt file is used
TEAL, RED, AMBER, VIOLET, GREY = "#0f8b7d", "#e5484d", "#f5a623", "#6e56cf", "#8a9a96"
SPECIES = ["P. falciparum", "P. vivax", "Mixed / other"]
# (district, region, demo positivity rate). Replace with your real pilot districts.
DISTRICTS = [("Humera", "Tigray", .07), ("Adwa", "Tigray", .04), ("Pawe", "Benishangul-Gumuz", .19), ("Assosa", "Benishangul-Gumuz", .15),
             ("Metema", "Amhara", .12), ("Bahir Dar Zuria", "Amhara", .05), ("Gewane", "Afar", .13), ("Semera", "Afar", .11),
             ("Dire Dawa", "Dire Dawa", .04), ("Itang", "Gambella", .24), ("Abobo", "Gambella", .22), ("Gambella Town", "Gambella", .17),
             ("Arjo", "Oromia", .10), ("Jimma", "Oromia", .08), ("Batu", "Oromia", .09), ("Hawassa", "Sidama", .09),
             ("Jigjiga", "Somali", .05), ("Kebri Dahar", "Somali", .08), ("Bonga", "South West", .14), ("Mizan Aman", "South West", .13),
             ("Kirkos", "Addis Ababa", .02), ("Jinka", "South Ethiopia", .15), ("Arba Minch", "South Ethiopia", .11)]

SCHEMA = """CREATE TABLE IF NOT EXISTS screenings(
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, district TEXT NOT NULL, region TEXT NOT NULL,
  result TEXT NOT NULL, confidence REAL, parasites INTEGER, wbc INTEGER, species TEXT,
  microscopist TEXT,      -- Positive / Negative / NULL (the expert's reading, for validation)
  quality TEXT, image_path TEXT, source TEXT NOT NULL)"""      # source: seed | live | live-demo


# ------------------------------- database -----------------------------------
def init_db():
    with closing(sqlite3.connect(DB)) as c:
        c.execute(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(screenings)")}
        for col in ("parasites", "wbc"):                 # add new columns to an older database
            if col not in cols:
                c.execute(f"ALTER TABLE screenings ADD COLUMN {col} INTEGER")
        c.commit()


def execute(sql, args=()):
    with closing(sqlite3.connect(DB)) as c:
        cur = c.execute(sql, args)
        c.commit()
        return cur.lastrowid


def read(sql, args=()):
    with closing(sqlite3.connect(DB)) as c:
        return pd.read_sql_query(sql, c, params=args)


def load_df(start, end, region=None):
    q, a = "SELECT * FROM screenings WHERE created_at>=? AND created_at<?", [start.isoformat(), end.isoformat()]
    if region:
        q, a = q + " AND region=?", a + [region]
    df = read(q, a)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pos"] = (df["result"] == "Positive").astype(int)
    return df


def seed_demo(n=6000):
    now, rows = datetime.utcnow(), []
    for _ in range(n):
        name, region, base = random.choice(DISTRICTS)
        t = random.random() ** 0.8
        when = now - timedelta(days=90 * (1 - t), seconds=random.randint(0, 86400))
        pos = random.random() < min(base * (0.8 + 0.4 * t), 0.6)
        mic = None
        if random.random() < 0.25:                      # 25% have an expert reading, 93% agreement
            agree = random.random() < 0.93
            mic = ("Positive" if pos else "Negative") if agree else ("Negative" if pos else "Positive")
        rows.append((when.isoformat(), name, region, "Positive" if pos else "Negative", round(random.uniform(.85, .99), 3),
                     random.randint(3, 60) if pos else 0, random.randint(5, 40),
                     random.choices(SPECIES, [68, 27, 5])[0] if pos else None, mic,
                     random.choice(["Good", "Good", "Fair", "Poor"]), None, "seed"))
    with closing(sqlite3.connect(DB)) as c:
        c.execute("DELETE FROM screenings WHERE source='seed'")
        c.executemany("INSERT INTO screenings(created_at,district,region,result,confidence,parasites,wbc,species,microscopist,quality,image_path,source) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        c.commit()


# --------------------------------- model ------------------------------------
TILE, STRIDE = 640, 512        # same tile size as in training; 128 px overlap so every parasite sits fully inside some tile
TARGET_LONG_SIDE = 4096        # training images were 4096 px wide; other sizes are rescaled to match


def find_model():
    files = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime) if MODELS_DIR.exists() else []
    return files[-1] if files else None


@st.cache_resource
def load_model():
    path = find_model()
    if path:
        try:
            from ultralytics import YOLO
            return YOLO(str(path))
        except Exception as e:
            st.sidebar.error(f"Could not load model: {e}")
    return None


def nms(boxes, scores, thr=0.5):
    """Greedy non-maximum suppression: drops duplicate boxes found in overlapping tiles."""
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        if rest.size == 0:
            break
        xx1, yy1 = np.maximum(boxes[i, 0], boxes[rest, 0]), np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2, yy2 = np.minimum(boxes[i, 2], boxes[rest, 2]), np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        order = rest[inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9) < thr]
    return np.array(keep, dtype=int)


def detect(model, img, conf):
    """Cut the image into overlapping tiles, run the model on each, merge the results.
    Returns boxes (Nx4), scores, classes (0 = parasite, 1 = wbc) and the (rescaled) image the boxes refer to."""
    img = img.convert("RGB")
    s = TARGET_LONG_SIDE / max(img.size)
    if abs(s - 1) > 0.02:
        img = img.resize((round(img.width * s), round(img.height * s)))
    if img.width < TILE or img.height < TILE:                     # pad very small images
        pad = Image.new("RGB", (max(img.width, TILE), max(img.height, TILE)), (255, 255, 255))
        pad.paste(img, (0, 0))
        img = pad
    W, H = img.size
    xs, ys = list(range(0, W - TILE + 1, STRIDE)), list(range(0, H - TILE + 1, STRIDE))
    if xs[-1] != W - TILE:
        xs.append(W - TILE)
    if ys[-1] != H - TILE:
        ys.append(H - TILE)
    arr = np.asarray(img)
    origins = [(x, y) for y in ys for x in xs]
    B, S, C = [], [], []
    for i in range(0, len(origins), 16):
        chunk = origins[i:i + 16]
        tiles = [np.ascontiguousarray(arr[y:y + TILE, x:x + TILE, ::-1]) for x, y in chunk]   # RGB -> BGR for ultralytics
        for (x0, y0), r in zip(chunk, model.predict(tiles, imgsz=TILE, conf=conf, verbose=False)):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for (a, b, c, d), p, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                # skip boxes cut off by a tile edge (the neighbouring tile sees them whole)
                if (a < 4 and x0 > 0) or (b < 4 and y0 > 0) or (c > TILE - 4 and x0 + TILE < W) or (d > TILE - 4 and y0 + TILE < H):
                    continue
                B.append([a + x0, b + y0, c + x0, d + y0]); S.append(p); C.append(k)
    if not B:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), img
    B, S, C = np.array(B), np.array(S), np.array(C)
    keep = np.concatenate([np.where(C == k)[0][nms(B[C == k], S[C == k])] for k in set(C.tolist())])
    keep.sort()
    return B[keep], S[keep], C[keep], img


def draw(img, B, C):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), k in zip(B, C):
        d.rectangle([x1, y1, x2, y2], outline="red" if k == 0 else "lime", width=5)
    im.thumbnail((1600, 1600))
    return im


def analyze(model, img, conf, min_parasites):
    if model is None:                                                      # DEMO mode
        n = random.choice([0, 0, 2, 5, 12, 30])
        im = img.convert("RGB")
        im.thumbnail((1600, 1600))
        out = dict(parasites=n, wbc=random.randint(3, 25), mean_conf=round(random.uniform(.7, .95), 3) if n else None, image=im, demo=True)
    else:
        B, S, C, im = detect(model, img, conf)
        n = int((C == 0).sum())
        out = dict(parasites=n, wbc=int((C == 1).sum()), mean_conf=round(float(S[C == 0].mean()), 3) if n else None,
                   image=draw(im, B, C), demo=False)
    out["result"] = "Positive" if n >= min_parasites else "Negative"
    return out


# ------------------------------- analysis -----------------------------------
def hotspots(df):
    g = df.groupby(["district", "region"]).agg(screened=("pos", "size"), positive=("pos", "sum")).reset_index()
    g = g[g.screened >= 30].copy()                       # ignore districts with very few screenings
    g["positivity_%"] = (g.positive / g.screened * 100).round(1)
    g["status"] = g["positivity_%"].apply(lambda r: "🔴 Critical" if r >= 20 else "🟠 High" if r >= 15 else "🟡 Watch" if r >= 10 else "")
    return g[g.status != ""].sort_values("positivity_%", ascending=False)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, centre - half), min(1, centre + half)


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n * 100:.1f}%  (95% CI {lo * 100:.0f}-{hi * 100:.0f}%, n={n})"


# ----------------------------------- UI -------------------------------------
def overview(df, prev):
    if df.empty:
        st.info("No data in this period yet. Click **Load demo data** in the sidebar, or screen a smear.")
        return
    n, p = len(df), int(df.pos.sum())
    pn, pp = len(prev), int(prev.pos.sum())
    rate, prate = p / n * 100, (pp / pn * 100 if pn else 0)
    ok = pn >= 0.5 * n                                   # compare only when the previous period is comparable
    pct = lambda a, b: f"{(a - b) / b * 100:+.1f}%" if ok and b else None
    hot = hotspots(df)
    c = st.columns(4)
    c[0].metric("Total screened", f"{n:,}", pct(n, pn))
    c[1].metric("Positive cases", f"{p:,}", pct(p, pp), delta_color="inverse")
    c[2].metric("Positivity rate", f"{rate:.1f}%", f"{rate - prate:+.1f} pts" if ok else None, delta_color="inverse")
    c[3].metric("Active hotspots", len(hot))

    wk = df.set_index("created_at").resample("W").agg(screened=("pos", "size"), positive=("pos", "sum")).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=wk.created_at, y=wk.screened, name="Total screened", fill="tozeroy", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=wk.created_at, y=wk.positive, name="Positive", line=dict(color=RED, width=3))
    fig.update_layout(title="Screening trend (weekly)", height=320, margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    reg = df.groupby("region").agg(n=("pos", "size"), p=("pos", "sum")).reset_index()
    reg["positivity_%"] = (reg.p / reg.n * 100).round(1)
    f1 = px.bar(reg.sort_values("positivity_%"), x="positivity_%", y="region", orientation="h", color="positivity_%",
                range_color=(0, 20), color_continuous_scale=["#2fa866", "#f5d44a", AMBER, RED], title="Positivity by region (%)")
    f1.update_layout(height=380, margin=dict(t=50), coloraxis_showscale=False)
    a.plotly_chart(f1, use_container_width=True)

    sp = df[df.pos == 1].species.fillna("Not classified").value_counts().reset_index()
    sp.columns = ["species", "count"]
    f2 = px.pie(sp, names="species", values="count", hole=.55, title="Parasite species (positives)",
                color="species", color_discrete_map={SPECIES[0]: RED, SPECIES[1]: AMBER, SPECIES[2]: VIOLET, "Not classified": GREY})
    f2.update_layout(height=380, margin=dict(t=50))
    b.plotly_chart(f2, use_container_width=True)

    top = df.groupby("district").pos.sum().nlargest(8).reset_index()
    f3 = px.bar(top, x="pos", y="district", orientation="h", title="Top districts by positive cases", color_discrete_sequence=[TEAL])
    f3.update_layout(height=340, margin=dict(t=50), yaxis=dict(autorange="reversed"), xaxis_title="positive cases")
    st.plotly_chart(f3, use_container_width=True)

    st.subheader("Hotspot alerts")
    st.caption("Districts with 30+ screenings and positivity of 10%+ (Watch), 15%+ (High), 20%+ (Critical). Adjust the rule in hotspots().")
    st.dataframe(hot, hide_index=True, use_container_width=True)


def screen_tab(model):
    st.subheader("Analyze a blood smear image")
    if model is None:
        st.warning("DEMO MODE: no .pt model found in the models folder. Results below are random.")
    left, right = st.columns(2)
    with left:
        label = st.selectbox("District", [f"{d} ({r})" for d, r, _ in DISTRICTS])
        quality = st.selectbox("Image quality", ["Good", "Fair", "Poor"])
        mic = st.selectbox("Microscopist result (if already known)", ["Not yet read", "Positive", "Negative"])
        f = st.file_uploader("Smear image (thick smear photo)", type=["jpg", "jpeg", "png", "tif", "tiff"])
        with st.expander("Detection settings (provisional, to be set from validation results)"):
            conf = st.slider("Minimum confidence for a detection", 0.10, 0.90, 0.40, 0.05)
            min_par = st.number_input("Parasites needed to call the slide positive", 1, 50, 3)
    with right:
        if f:
            img = Image.open(f)
            img.load()
            if not st.button("Analyze and save", type="primary"):
                st.image(img, use_container_width=True)
                return
            with st.spinner("Analyzing... this can take 10-40 seconds on a laptop"):
                res = analyze(model, img, conf, min_par)
            name = label.rsplit(" (", 1)[0]
            region = next(r for d, r, _ in DISTRICTS if d == name)
            now = datetime.utcnow()
            path = UPLOADS / f"{now:%Y%m%d%H%M%S%f}.jpg"
            img.convert("RGB").save(path, quality=95)
            rid = execute("INSERT INTO screenings(created_at,district,region,result,confidence,parasites,wbc,species,microscopist,quality,image_path,source) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                          (now.isoformat(), name, region, res["result"], res["mean_conf"], res["parasites"], res["wbc"], None,
                           None if mic == "Not yet read" else mic, quality, path.name, "live-demo" if res["demo"] else "live"))
            (st.error if res["result"] == "Positive" else st.success)(
                f"**{res['result']}**  |  parasites found: {res['parasites']}  |  white blood cells: {res['wbc']}  |  saved as record #{rid}")
            st.image(res["image"], caption="red = parasite, green = white blood cell", use_container_width=True)
            st.caption("AI-assisted screening only. A trained microscopist must confirm every result.")


def validation_tab():
    st.subheader("Validation: AI result vs microscopist")
    st.caption("Enter the microscopist's reading for each screened slide. Sensitivity and specificity update automatically.")
    pend = read("SELECT id,created_at,district,result FROM screenings WHERE microscopist IS NULL AND source!='seed' ORDER BY id DESC LIMIT 50")
    if pend.empty:
        st.info("No live screenings are waiting for a microscopist reading.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        rid = c1.selectbox("Record", pend.id, format_func=lambda i: f"#{i}  {pend[pend.id == i].district.iloc[0]}  (AI: {pend[pend.id == i].result.iloc[0]})")
        val = c2.radio("Microscopist result", ["Positive", "Negative"], horizontal=True)
        if c3.button("Save reading"):
            execute("UPDATE screenings SET microscopist=? WHERE id=?", (val, int(rid)))
            st.rerun()
    demo = st.checkbox("Include demo data in the numbers below", value=read("SELECT COUNT(*) c FROM screenings WHERE source!='seed'").c[0] == 0)
    v = read("SELECT result,microscopist FROM screenings WHERE microscopist IN ('Positive','Negative')" + ("" if demo else " AND source!='seed'"))
    if v.empty:
        st.info("No slides with a microscopist reading yet.")
        return
    tp = int(((v.result == "Positive") & (v.microscopist == "Positive")).sum())
    fn = int(((v.result == "Negative") & (v.microscopist == "Positive")).sum())
    tn = int(((v.result == "Negative") & (v.microscopist == "Negative")).sum())
    fp = int(((v.result == "Positive") & (v.microscopist == "Negative")).sum())
    m = st.columns(3)
    m[0].metric("Sensitivity", fmt(tp, tp + fn).split("  ")[0])
    m[1].metric("Specificity", fmt(tn, tn + fp).split("  ")[0])
    m[2].metric("Slides compared", len(v))
    st.write("Sensitivity:", fmt(tp, tp + fn), "  \nSpecificity:", fmt(tn, tn + fp))
    st.dataframe(pd.DataFrame([[tp, fn], [fp, tn]], index=["Microscopist: Positive", "Microscopist: Negative"],
                              columns=["AI: Positive", "AI: Negative"]), use_container_width=True)
    if len(v) < 100:
        st.warning("Fewer than 100 compared slides: the confidence intervals are wide, so treat these numbers as preliminary.")


def main():
    st.set_page_config(page_title="MalariAI", page_icon="🦟", layout="wide")
    init_db()
    st.markdown("<div style='background:linear-gradient(120deg,#0b5f56,#0f8b7d 60%,#2cc4b0);padding:18px 22px;border-radius:14px;color:white'>"
                "<h2 style='margin:0;color:white'>🦟 MalariAI · National Surveillance</h2>"
                "<span style='opacity:.85'>AI-assisted blood smear screening. Screening support only, not a diagnosis.</span></div>", unsafe_allow_html=True)
    with st.sidebar:
        st.header("Filters")
        weeks = {"Last 4 weeks": 4, "Last 12 weeks": 12, "Last 26 weeks": 26}[st.selectbox("Period", ["Last 4 weeks", "Last 12 weeks", "Last 26 weeks"], index=1)]
        region = st.selectbox("Region", ["All regions"] + sorted({r for _, r, _ in DISTRICTS}))
        st.divider()
        if st.button("Load demo data"):
            seed_demo()
            st.rerun()
        if st.button("Remove demo data"):
            execute("DELETE FROM screenings WHERE source='seed'")
            st.rerun()
    model = load_model()
    now = datetime.utcnow() + timedelta(minutes=1)
    reg = None if region == "All regions" else region
    df = load_df(now - timedelta(weeks=weeks), now, reg)
    prev = load_df(now - timedelta(weeks=2 * weeks), now - timedelta(weeks=weeks), reg)
    t1, t2, t3 = st.tabs(["📊 Overview", "🔬 Screen a smear", "✅ Validation"])
    with t1:
        overview(df, prev)
    with t2:
        screen_tab(model)
    with t3:
        validation_tab()


if __name__ == "__main__":
    main()
