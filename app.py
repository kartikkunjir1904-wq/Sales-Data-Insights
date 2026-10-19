import re
import warnings
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# HELPERS  (data logic, no UI)
# ============================================================
NULL_WORDS = {"", "nan", "null", "none", "n/a", "na", "-", "--", "?", "<na>", "nat"}

SALES_EXACT = ["sales", "sale", "revenue", "total sales", "total amount", "total price",
               "selling price", "sellingprice", "sale price", "amount", "money", "price",
               "total", "sales amount", "net sales"]
SALES_WORDS = ["sales", "revenue", "amount", "price", "money", "total", "value", "cost", "mmr"]
PRODUCT_EXACT = ["product", "product name", "item", "item name", "coffee name", "category",
                 "product category", "item category", "type", "product type",
                 "make", "brand", "body", "segment", "description", "model"]
QTY_EXACT = ["quantity", "qty", "units sold", "units", "order quantity"]
ID_WORDS = {"id", "vin", "zip", "phone", "index", "year", "code", "no", "number", "card",
            "customer", "odometer", "mileage", "lat", "lon", "latitude", "longitude"}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(s).strip().lower()))


def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def looks_like_id(col):
    words = set(norm(col).split())
    return bool(words & ID_WORDS) or norm(col).endswith("id")


def to_number(series):
    """'$1,200.50' -> 1200.5 . Leaves numeric columns untouched."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def number_rate(series):
    """Share of non-null values that can be read as numbers."""
    s = series.dropna()
    if s.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(s):
        return 1.0
    return to_number(s.head(500)).notna().mean()


def parse_dates(series, dayfirst=False):
    """Parse almost any date format -> naive datetimes (NaT where impossible)."""
    s = series.astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
    warnings.simplefilter("ignore")
    try:
        out = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=dayfirst)
        if out.isna().mean() > 0.2:  # mixed formats -> slower but flexible parser
            out = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=dayfirst, format="mixed")
    except (TypeError, ValueError):
        out = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=dayfirst)
    return out.dt.tz_convert(None)


def detect_date_col(df, dayfirst=False):
    best, best_score = None, 0
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            return c
        if pd.api.types.is_numeric_dtype(s):
            continue
        sample = s.dropna().astype(str).head(300)
        if sample.empty or pd.to_numeric(sample, errors="coerce").notna().mean() > 0.8:
            continue
        parsed = parse_dates(sample, dayfirst)
        ok = parsed.dt.year.between(1980, 2100)
        rate = ok.mean()
        if rate < 0.8:
            continue
        score = rate + (1 if any(w in norm(c) for w in ("date", "time")) else 0)
        if score > best_score:
            best, best_score = c, score
    return best


def detect_sales_col(df, exclude=()):
    cols = [c for c in df.columns if c not in exclude]
    by_norm = {norm(c): c for c in cols}
    for name in SALES_EXACT:
        if name in by_norm and number_rate(df[by_norm[name]]) >= 0.8:
            return by_norm[name]
    for word in SALES_WORDS:
        for c in cols:
            if word in norm(c) and not looks_like_id(c) and number_rate(df[c]) >= 0.8:
                return c
    nums = [c for c in cols if not looks_like_id(c) and number_rate(df[c]) >= 0.8
            and df[c].nunique() > 10]
    if nums:
        return max(nums, key=lambda c: to_number(df[c]).abs().mean())
    return None


def detect_product_col(df, exclude=()):
    cols = [c for c in df.columns if c not in exclude]
    n = max(len(df), 1)

    def usable(c):
        u = df[c].nunique()
        return 2 <= u <= min(500, 0.5 * n) and not pd.api.types.is_float_dtype(df[c])

    by_norm = {norm(c): c for c in cols}
    for name in PRODUCT_EXACT:
        if name in by_norm and usable(by_norm[name]):
            return by_norm[name]
    for word in ("product", "category", "item", "type", "brand", "make", "name"):
        for c in cols:
            if word in norm(c) and not looks_like_id(c) and is_text(df[c]) and usable(c):
                return c
    cands = [c for c in cols if not looks_like_id(c) and is_text(df[c]) and usable(c)]
    return cands[0] if cands else None


def detect_qty_col(df, exclude=()):
    by_norm = {norm(c): c for c in df.columns if c not in exclude}
    for name in QTY_EXACT:
        if name in by_norm and number_rate(df[by_norm[name]]) >= 0.8:
            return by_norm[name]
    return None


def clean_data(df, sales_col, date_col, product_col, qty_col, dayfirst, drop_any):
    """Returns (cleaned_df, report_dict)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    report = {"original": len(out)}

    out = out.dropna(how="all")
    report["empty"] = report["original"] - len(out)

    # tidy text + turn 'N/A', 'null', '' etc. into real missing values
    for c in out.columns:
        if is_text(out[c]):
            s = out[c].astype(str).str.strip()
            out[c] = s.mask(s.str.lower().isin(NULL_WORDS))

    before = len(out)
    out = out.drop_duplicates()
    report["duplicates"] = before - len(out)

    if sales_col:
        out[sales_col] = to_number(out[sales_col])
    if qty_col:
        out[qty_col] = to_number(out[qty_col])
    if date_col:
        out[date_col] = parse_dates(out[date_col], dayfirst)

    before = len(out)
    if drop_any:
        out = out.dropna()
    else:
        key = [c for c in (sales_col, date_col, product_col, qty_col) if c]
        if key:
            out = out.dropna(subset=key)
    report["missing"] = before - len(out)

    before = len(out)
    if sales_col:
        out = out[out[sales_col] > 0]
    report["invalid"] = before - len(out)

    report["final"] = len(out)
    return out.reset_index(drop=True), report


def to_excel(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Cleaned Data")
    return buf.getvalue()


def period_start(dates, gran):
    if gran == "Day":
        return dates.dt.normalize()
    if gran == "Week":
        d = dates.dt.normalize()
        return d - pd.to_timedelta(d.dt.dayofweek, unit="D")
    if gran == "Month":
        return dates.dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime(dates.dt.year.astype(str) + "-01-01")


FREQ = {"Day": "D", "Week": "7D", "Month": "MS", "Year": "YS"}


def time_series(df, date_col, value_col, gran):
    """Total per period, with empty periods filled with 0 so the axis covers the full range."""
    start = period_start(df[date_col], gran)
    if value_col is None:
        grouped = df.groupby(start).size()
    else:
        grouped = df.groupby(start)[value_col].sum()
    full = pd.date_range(grouped.index.min(), grouped.index.max(), freq=FREQ[gran])
    grouped = grouped.reindex(full, fill_value=0)
    return grouped.rename_axis("Period").reset_index(name="Value")


def auto_granularity(dmin, dmax):
    days = (dmax - dmin).days
    if days <= 45:
        return "Day"
    if days <= 200:
        return "Week"
    if days <= 1460:
        return "Month"
    return "Year"


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Sales Data Cleaner & Insights", page_icon="🛒", layout="wide")
st.title("🛒 Sales Data Cleaner & Insights")
st.caption("Upload any sales CSV → cleaned data → 4 key insights.")


@st.cache_data(show_spinner="Reading file...")
def load_file(name, content):
    bio = BytesIO(content)
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(bio, low_memory=False)
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, encoding="latin-1", low_memory=False)
    return pd.read_excel(bio)


@st.cache_data(show_spinner="Detecting columns...")
def detect_all(df, dayfirst):
    date = detect_date_col(df, dayfirst)
    sales = detect_sales_col(df, exclude=[date])
    qty = detect_qty_col(df, exclude=[date, sales])
    prod = detect_product_col(df, exclude=[date, sales, qty])
    return sales, date, prod, qty


@st.cache_data(show_spinner="Cleaning data...")
def cached_clean(df, sales, date, prod, qty, dayfirst, drop_any):
    return clean_data(df, sales, date, prod, qty, dayfirst, drop_any)


# ---------------- 1. UPLOAD ----------------
file = st.sidebar.file_uploader("1️⃣ Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if not file:
    st.info("👈 Upload a CSV or Excel file to begin.")
    st.stop()

try:
    raw = load_file(file.name, file.getvalue())
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if raw.empty:
    st.error("The file has no data.")
    st.stop()

raw.columns = [str(c).strip() for c in raw.columns]

# ---------------- column choices ----------------
st.sidebar.header("⚙️ Settings")
dayfirst = st.sidebar.checkbox("Dates are day-first (31/12/2024)", value=False)
drop_any = st.sidebar.checkbox("Remove rows with ANY missing value", value=False,
                               help="Off = only rows missing sales/date/product are removed.")

g_sales, g_date, g_prod, g_qty = detect_all(raw, dayfirst)
NONE = "— none —"
options = [NONE] + list(raw.columns)


def pick(label, guess):
    idx = options.index(guess) if guess in options else 0
    choice = st.sidebar.selectbox(label, options, index=idx)
    return None if choice == NONE else choice


st.sidebar.subheader("Columns (auto-detected)")
sales_col = pick("Sales / price (money)", g_sales)
date_col = pick("Date", g_date)
prod_col = pick("Product / category", g_prod)
qty_col = pick("Quantity (optional)", g_qty)

# ---------------- 2. CLEAN ----------------
df, rep = cached_clean(raw, sales_col, date_col, prod_col, qty_col, dayfirst, drop_any)

st.header("🧹 Step 1: Cleaned Data")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Original rows", f"{rep['original']:,}")
c2.metric("Empty rows removed", f"{rep['empty']:,}")
c3.metric("Duplicates removed", f"{rep['duplicates']:,}")
c4.metric("Missing/invalid removed", f"{rep['missing'] + rep['invalid']:,}")
c5.metric("Final rows", f"{rep['final']:,}")

if df.empty:
    st.error("All rows were removed. Try a different column selection in the sidebar, "
             "or untick 'Remove rows with ANY missing value'.")
    st.stop()

st.dataframe(df.head(20), use_container_width=True)

# ---------------- 3. DOWNLOAD ----------------
d1, d2, _ = st.columns([1, 1, 4])
d1.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "cleaned_data.csv", "text/csv")
d2.download_button("⬇️ Download cleaned Excel", to_excel(df), "cleaned_data.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- 4. INSIGHTS ----------------
st.header("📈 Step 2: Insights")

used = {"Money": sales_col, "Date": date_col, "Product": prod_col, "Quantity": qty_col}
st.caption("Using → " + " | ".join(f"**{k}:** {v or 'not found'}" for k, v in used.items())
           + "  (change in sidebar if wrong)")

if sales_col:
    k1, k2, k3 = st.columns(3)
    k1.metric("Total revenue", f"{df[sales_col].sum():,.0f}")
    k2.metric("Average sale", f"{df[sales_col].mean():,.0f}")
    k3.metric("Number of sales", f"{len(df):,}")

count_label = "Units sold" if qty_col else "Number of sales"
count_value = qty_col  # None -> count rows

# ----- time-based insights (1 & 2) -----
if date_col and df[date_col].notna().any():
    dmin, dmax = df[date_col].min(), df[date_col].max()
    st.info(f"📅 Data covers **{dmin:%d %b %Y}** to **{dmax:%d %b %Y}** — "
            "the charts below use exactly this range.")

    auto = auto_granularity(dmin, dmax)
    grans = ["Day", "Week", "Month", "Year"]
    gran = st.radio("Group time by", grans, index=grans.index(auto), horizontal=True)

    st.subheader(f"1️⃣ {count_label} over time")
    ts = time_series(df, date_col, count_value, gran)
    fig = px.line(ts, x="Period", y="Value", markers=(len(ts) <= 60),
                  labels={"Value": count_label, "Period": gran})
    fig.update_xaxes(range=[dmin, dmax])
    st.plotly_chart(fig, use_container_width=True)
    peak = ts.loc[ts["Value"].idxmax()]
    st.success(f"Busiest {gran.lower()}: **{peak['Period']:%d %b %Y}** ({peak['Value']:,.0f}).")

    st.subheader("2️⃣ Revenue over time")
    if sales_col:
        rs = time_series(df, date_col, sales_col, gran)
        fig = px.bar(rs, x="Period", y="Value", labels={"Value": "Revenue", "Period": gran})
        fig.update_xaxes(range=[dmin, dmax])
        st.plotly_chart(fig, use_container_width=True)
        peak = rs.loc[rs["Value"].idxmax()]
        st.success(f"Highest revenue {gran.lower()}: **{peak['Period']:%d %b %Y}** "
                   f"({peak['Value']:,.0f}).")
    else:
        st.warning("No money column found — pick one in the sidebar to see revenue.")
else:
    st.subheader(f"1️⃣ {count_label} over time")
    st.warning("No date column found — pick one in the sidebar to see time-based insights.")
    st.subheader("2️⃣ Revenue over time")
    st.warning("No date column found — pick one in the sidebar.")

# ----- product insight (3) -----
st.subheader("3️⃣ Top selling products")
if prod_col:
    mode_opts = ["By number sold"] + (["By revenue"] if sales_col else [])
    mode = st.radio("Rank products", mode_opts, horizontal=True)
    if mode == "By revenue":
        top = df.groupby(prod_col)[sales_col].sum()
        label = "Revenue"
    elif qty_col:
        top = df.groupby(prod_col)[qty_col].sum()
        label = "Units sold"
    else:
        top = df.groupby(prod_col).size()
        label = "Number of sales"
    top = top.sort_values(ascending=False).head(10).reset_index()
    top.columns = [prod_col, label]
    fig = px.bar(top.iloc[::-1], x=label, y=prod_col, orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Top product: **{top.iloc[0][prod_col]}** ({top.iloc[0][label]:,.0f}).")
else:
    st.warning("No product/category column found — pick one in the sidebar.")

# ----- price insight (4) -----
st.subheader("4️⃣ Price range of sales")
if sales_col:
    p = df[sales_col]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lowest", f"{p.min():,.0f}")
    m2.metric("Median", f"{p.median():,.0f}")
    m3.metric("Average", f"{p.mean():,.0f}")
    m4.metric("Highest", f"{p.max():,.0f}")
    shown = p[p <= p.quantile(0.99)]  # hide extreme outliers so the chart is readable
    fig = px.histogram(shown, x=sales_col, nbins=30,
                       labels={sales_col: "Sale value", "count": "Number of sales"})
    fig.update_layout(yaxis_title="Number of sales", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top 1% of values are hidden in the chart to keep it readable "
               "(they are still counted in the numbers above).")
else:
    st.warning("No money column found — pick one in the sidebar.")
