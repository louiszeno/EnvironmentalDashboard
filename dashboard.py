import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from datetime import datetime

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="UK Inclusive Wealth Dashboard",
    page_icon="🇬🇧",
    layout="wide"
)

# ----------------- GLOBAL CONFIG -----------------
COUNTRY = "GBR"
MIN_YEAR = 2000  # base year for analysis

# Capital → indicator specs.
# Each indicator has its OWN unit.
CAPITALS = {
    "Natural": {
        "capital": "Natural",
        "primary": {
            "code": "NW.NCA.TO",
            "name": "Natural Capital (const US$)",
            "source": "World Bank – Wealth Accounts",
            "unit": "US$ (constant, trillions)"
        },
        "support": {
            "code": "AG.LND.FRST.ZS",
            "name": "Forest area (% of land area)",
            "source": "World Bank – WDI",
            "unit": "% of land area"
        },
        "support2": {
            "code": "EN.ATM.CO2E.PC",
            "name": "CO₂ emissions (metric tons per capita)",
            "source": "World Bank – WDI",
            "unit": "Metric tons per capita"
        },
    },
    "Human": {
        "capital": "Human",
        "primary": {
            "code": "NW.HCA.TO",
            "name": "Human Capital (const US$)",
            "source": "World Bank – Wealth Accounts",
            "unit": "US$ (constant, trillions)"
        },
        "support": {
            "code": "SP.DYN.LE00.IN",
            "name": "Life expectancy at birth (years)",
            "source": "World Bank – WDI",
            "unit": "Years"
        },
        "support2": {
            "code": "SE.XPD.TOTL.GD.ZS",
            "name": "Education expenditure (% of GDP)",
            "source": "World Bank – WDI",
            "unit": "% of GDP"
        },
    },
    "Manufactured": {
        "capital": "Manufactured",
        "primary": {
            "code": "NW.PCA.TO",
            "name": "Produced Capital (const US$)",
            "source": "World Bank – Wealth Accounts",
            "unit": "US$ (constant, trillions)"
        },
        "support": {
            "code": "NE.GDI.FTOT.ZS",
            "name": "Gross fixed capital formation (% of GDP)",
            "source": "World Bank – WDI",
            "unit": "% of GDP"
        },
        # ONS Net Capital Stock added separately
    },
    "Knowledge": {
        "capital": "Knowledge",
        "primary": {
            "code": "GB.XPD.RSDV.GD.ZS",
            "name": "R&D Expenditure (% of GDP)",
            "source": "World Bank – WDI",
            "unit": "% of GDP"
        },
        "support": {
            "code": "IP.PAT.RESD",
            "name": "Patent Applications (resident)",
            "source": "World Bank – WDI",
            "unit": "Number of applications"
        },
    },
    "Social": {
        "capital": "Social",
        # Voice & Accountability comes from CSV
        "primary": {
            "code": "VA.EST",
            "name": "Voice & Accountability (estimate)",
            "source": "Worldwide Governance Indicators",
            "unit": "Index (−2.5 to 2.5)"
        },
        "support": {
            "code": "GE.EST",
            "name": "Government Effectiveness (estimate)",
            "source": "Worldwide Governance Indicators",
            "unit": "Index (−2.5 to 2.5)"
        },
    }
}

VA_INDICATOR_NAME = CAPITALS["Social"]["primary"]["name"]


# ----------------- HELPERS -----------------


def wb_get_series(country, indicator, start, end, retries=3, pause=0.8):
    """Fetch a single WB indicator for given country and date range."""
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/"
        f"{indicator}?date={start}:{end}&format=json&per_page=2000"
    )
    for i in range(retries):
        try:
            r = requests.get(url, timeout=15)
        except Exception:
            time.sleep(pause * (i + 1))
            continue

        if r.status_code == 200:
            try:
                js = r.json()
            except Exception:
                time.sleep(pause * (i + 1))
                continue

            if isinstance(js, list) and len(js) > 1 and isinstance(js[1], list):
                rows = []
                for d in js[1]:
                    if d.get("date") and d.get("value") is not None:
                        rows.append(
                            {"Year": int(d["date"]), "Value": float(d["value"])}
                        )
                return pd.DataFrame(rows)
        time.sleep(pause * (i + 1))
    return pd.DataFrame(columns=["Year", "Value"])


@st.cache_data(show_spinner=True)
def load_all_data():
    """
    Pulls WB data for Natural, Human, Manufactured, Knowledge, Social (GE).
    Loads Voice & Accountability from local CSV (voice_accountability_uk.csv).
    Adds ONS Net Capital Stock.
    Returns: data_raw (long), meta (indicator metadata).
    """
    start = MIN_YEAR
    end = datetime.now().year

    frames = []
    meta_rows = []

    # --- World Bank indicators for all capitals except VA from CSV ---
    for cap, spec in CAPITALS.items():
        for role in ["primary", "support", "support2"]:
            if role not in spec:
                continue

            ind = spec[role]

            # Skip VA.EST here; we'll load from CSV instead
            if cap == "Social" and ind["code"] == "VA.EST":
                continue

            df = wb_get_series(COUNTRY, ind["code"], start, end)
            if df.empty:
                continue

            df["Capital"] = cap
            df["Role"] = role
            df["Indicator"] = ind["name"]
            df["Code"] = ind["code"]
            frames.append(df)

            meta_rows.append(
                {
                    "Capital": cap,
                    "Indicator": ind["name"],
                    "Code": ind["code"],
                    "Source": ind["source"],
                    "Unit": ind.get("unit", ""),
                }
            )

    # --- SOCIAL CAPITAL: Voice & Accountability from CSV ---
    try:
        va = pd.read_csv("voice_accountability_uk.csv")
    except FileNotFoundError:
        st.error(
            "voice_accountability_uk.csv not found. "
            "Put it in the same directory as this script."
        )
        return pd.DataFrame(), pd.DataFrame()

    if not {"Year", "Value"}.issubset(va.columns):
        st.error(
            "voice_accountability_uk.csv must have columns: Year, Value "
            "(see instructions)."
        )
        return pd.DataFrame(), pd.DataFrame()

    va["Capital"] = "Social"
    va["Role"] = "primary"
    va["Indicator"] = VA_INDICATOR_NAME
    va["Code"] = CAPITALS["Social"]["primary"]["code"]
    frames.append(va)

    meta_rows.append(
        {
            "Capital": "Social",
            "Indicator": VA_INDICATOR_NAME,
            "Code": CAPITALS["Social"]["primary"]["code"],
            "Source": CAPITALS["Social"]["primary"]["source"],
            "Unit": CAPITALS["Social"]["primary"]["unit"],
        }
    )

    # --- ONS NET CAPITAL STOCK (Manufactured) ---
    try:
        ons_url = "https://api.ons.gov.uk/timeseries/MJU5/dataset/CAPSTK/data"
        resp = requests.get(
            ons_url, headers={"User-Agent": "UK-Inclusive-Wealth-Dashboard"}, timeout=15
        )
        data = resp.json()
        years = []
        values = []
        for entry in data["years"]:
            y = int(entry["year"])
            if y >= start:
                years.append(y)
                values.append(float(entry["value"]))
        df_ons = pd.DataFrame({"Year": years, "Value": values})
        df_ons["Capital"] = "Manufactured"
        df_ons["Role"] = "support_ons"
        df_ons["Indicator"] = "Net Capital Stock (ONS – £m)"
        df_ons["Code"] = "ONS:MJU5"
        frames.append(df_ons)

        meta_rows.append(
            {
                "Capital": "Manufactured",
                "Indicator": "Net Capital Stock (ONS – £m)",
                "Code": "ONS:MJU5",
                "Source": "ONS Capital Stock dataset (series MJU5)",
                "Unit": "£ million",
            }
        )
    except Exception:
        # If ONS fails, we just skip it; dashboard still works.
        pass

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    data_raw = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(meta_rows).drop_duplicates()

    return data_raw, meta


@st.cache_data(show_spinner=True)
def load_gdp_per_capita():
    """Load GDP per capita (constant) for decoupling plot."""
    start = MIN_YEAR
    end = datetime.now().year
    df = wb_get_series(COUNTRY, "NY.GDP.PCAP.KD", start, end)
    if df.empty:
        return pd.DataFrame(columns=["Year", "GDP_Index"])
    df = df.sort_values("Year")
    mask = df["Value"].notna()
    if not mask.any():
        return pd.DataFrame(columns=["Year", "GDP_Index"])
    base = df.loc[mask, "Value"].iloc[0]
    df["GDP_Index"] = (df["Value"] / base) * 100
    return df[["Year", "GDP_Index"]]


def index_series(df):
    """
    Reindex each indicator to annual frequency, interpolate gaps,
    compute index (first valid year = 100), and flag imputed years.
    Robust to missing years and always keeps 'Observed' boolean.
    """
    if df.empty:
        return df

    out = []

    for ind, g in df.groupby("Indicator"):
        g = g.sort_values("Year").copy()

        # Mark which observations are actually in the raw data
        g["Observed"] = g["Value"].notna()

        # Build a complete annual grid for this indicator
        year_min = int(g["Year"].min())
        year_max = int(g["Year"].max())
        full = pd.DataFrame({"Year": range(year_min, year_max + 1)})

        g = full.merge(
            g[["Year", "Value", "Observed", "Indicator", "Capital", "Role", "Code"]],
            on="Year",
            how="left",
        )

        # Fill metadata down/up
        for col in ["Indicator", "Capital", "Role", "Code"]:
            g[col] = g[col].ffill().bfill()

        # After merge, Observed may be NaN floats → force to boolean
        g["Observed"] = g["Observed"].fillna(False).astype(bool)

        # Interpolate values across missing years
        g["Value"] = g["Value"].interpolate(method="linear", limit_direction="both")

        # Compute index (first valid value in the series = 100)
        valid_mask = g["Value"].notna()
        if not valid_mask.any():
            continue

        base_val = g.loc[valid_mask, "Value"].iloc[0]
        if base_val == 0:
            continue

        g["Index"] = (g["Value"] / base_val) * 100

        # Imputed = not originally observed
        g["Imputed"] = ~g["Observed"]

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def compute_trends(df_window):
    """
    For filtered data: compute absolute change, CAGR, index slope,
    max drawdown and threat flag per indicator.
    """
    rows = []
    for ind, g in df_window.groupby("Indicator"):
        g = g.sort_values("Year")
        if g.empty:
            continue

        v0 = g["Value"].iloc[0]
        v1 = g["Value"].iloc[-1]
        y0 = g["Year"].iloc[0]
        y1 = g["Year"].iloc[-1]

        yrs = max(1, y1 - y0)

        # Absolute % change
        abs_change = ((v1 - v0) / v0) * 100 if v0 else float("nan")

        # CAGR
        if v0 > 0 and v1 > 0:
            try:
                cagr = ((v1 / v0) ** (1 / yrs) - 1) * 100
            except Exception:
                cagr = float("nan")
        else:
            cagr = float("nan")

        # slope of index
        slope = (g["Index"].iloc[-1] - g["Index"].iloc[0]) / yrs

        # max drawdown on index
        run_max = g["Index"].cummax()
        drawdown = (g["Index"] / run_max - 1).min() * 100

        threat = (slope < 0) and (drawdown < -15)

        rows.append(
            {
                "Indicator": ind,
                "Start": y0,
                "End": y1,
                "Abs_Change_%": abs_change,
                "CAGR_%": cagr,
                "Index_Slope_per_year": slope,
                "Max_Drawdown_%": drawdown,
                "Threat_Flag": threat,
            }
        )

    return pd.DataFrame(rows)


# ----------------- LOAD DATA -----------------
data_raw, meta = load_all_data()
gdp_df = load_gdp_per_capita()

if data_raw.empty:
    st.error("No data available. Fix the data sources and reload.")
    st.stop()

data_idx = index_series(data_raw)
if data_idx.empty:
    st.error("Indexing failed. Check your data for missing values / years.")
    st.stop()

year_min = int(data_idx["Year"].min())
year_max = int(data_idx["Year"].max())

# ----------------- UI -----------------
st.title("🇬🇧 UK Inclusive Wealth Dashboard")
st.caption(
    "Data: World Bank Wealth Accounts & World Development Indicators; "
    "Worldwide Governance Indicators; ONS Capital Stock series."
)

# --------- EXPLAINERS ---------
with st.expander("About this framework"):
    st.markdown(
        """
This dashboard applies the **Inclusive Wealth / Five Capitals framework** used in the
Inclusive Wealth Reports and in Matson et al. (2016).  
It interprets sustainable development as the maintenance or growth of the economy’s
**productive base of well-being** across **five asset classes**:

---

### **1. Manufactured (Produced) Capital**
Physical, human-made assets used to produce goods and services.  
Includes infrastructure, machinery, buildings, equipment.

**Indicators used:**
- *Produced Capital (constant US$)* — Wealth Accounts **(stock)**
- *Net Capital Stock (ONS)* — national estimate **(stock)**
- *Gross Fixed Capital Formation (% of GDP)* — investment **(flow)**

---

### **2. Human Capital**
The present value of the skills, health, and productivity embodied in individuals.

**Indicators used:**
- *Human Capital (constant US$)* — Wealth Accounts **(stock)**
- *Life expectancy at birth* — population health **(outcome)**
- *Education expenditure (% of GDP)* — human-capital investment **(flow)**

---

### **3. Natural Capital**
Environmental assets that support ecosystem services and resource use.

**Indicators used:**
- *Natural Capital (constant US$)* — Wealth Accounts **(stock)**
- *Forest area (% of land area)* — ecosystem extent **(stock-proxy)**
- *CO₂ emissions (tons per capita)* — environmental pressure **(flow/pressure)**

---

### **4. Knowledge Capital**
Codified and tacit knowledge that drives innovation and productivity.

**Indicators used:**
- *R&D expenditure (% of GDP)* — innovation input **(flow)**
- *Patent applications (resident)* — innovation output **(stock of codified knowledge)**

---

### **5. Social / Institutional Capital**
Institutions, norms, and governance quality that enable cooperation, compliance,
and long-term investment.

**Indicators used:**
- *Voice & Accountability (WGI)* — civic freedoms and democratic participation  
- *Government Effectiveness (WGI)* — state capacity and policy quality  

Both measured on **–2.5 to +2.5** governance scales.

---

### Why these indicators?

- Wealth Accounts provide internationally consistent **capital stocks**.  
- Supporting indicators capture **flows**, **pressures**, and **institutional quality**.  
- All indicators are available as **long-run time series** from the World Bank or ONS.  
- Together, they approximate the UK’s **inclusive wealth** and reveal whether
  long-run well-being is being supported or undermined.
        """
    )

with st.expander("What does 'Voice & Accountability' measure?"):
    st.markdown(
        """
**Voice & Accountability (Worldwide Governance Indicators)** is a measure of
**institutional social capital**. It captures:

- Freedom of expression and media  
- Freedom of association and civil liberties  
- Citizen participation in selecting government  
- Electoral integrity and political rights  
- Perceived government responsiveness  

The scale runs from **−2.5 (weak governance)** to **+2.5 (strong governance)**.

It does *not* measure interpersonal trust or community cohesion; rather, it reflects the
institutional environment that shapes long-run economic coordination, investment
decisions and the ability to undertake collective action.
        """
    )

with st.expander("Data & methods notes"):
    st.markdown(
        """
### **Data Sources**
- **World Bank Wealth Accounts (CWON)** — produced, human, natural capital stocks  
- **World Development Indicators (WDI)** — emissions, forests, R&D, education, patents, GFCF, life expectancy  
- **Worldwide Governance Indicators (WGI)** — Voice & Accountability, Government Effectiveness  
- **Office for National Statistics (ONS)** — Net Capital Stock (series MJU5)

---

### **Indicator Types**
- **Stocks**: Wealth-account measures of produced, human, natural capital  
- **Flows**: Education expenditure, R&D, GFCF  
- **Pressure indicators**: CO₂ emissions per capita  
- **Ecosystem extent**: Forest area (% of land)  
- **Institutional indicators**: WGI governance scores  

Each capital therefore includes a mix of:
- **Stock indicators** (long-lived assets)  
- **Flow indicators** (investment into assets)  
- **Pressure or condition indicators** (natural capital degradation risk)  
- **Institutional indicators** (for social capital)

---

### **Interpolation**
Wealth Account values are observed **every 5 years**.  
Intermediate years are **linearly interpolated**, with dashed-line segments marking
imputed values in the charts.

---

### **Indexing**
- Index = 100 at the first observed year in the selected range  
- Allows comparison of **growth/decline** across indicators with different units  

---

### **Units**
- Wealth stocks shown in **trillions of constant US$** in KPI and absolute views  
- Other indicators shown in their natural units (% GDP, years, index, etc.)  
- Because units differ, use **Indexed mode** for comparisons and **Absolute mode** for magnitude.

---

### **Limitations**
- Capital stocks are not adjusted for population (per-capita wealth would be ideal).  
- Social capital is proxied via institutional quality (not interpersonal trust).  
- Natural capital is missing some ecosystem services not monetised in CWON.  
        """
    )

# --------- SIDEBAR CONTROLS ---------

st.sidebar.header("Controls")
yr_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(max(year_min, MIN_YEAR), year_max),
)
capitals_selected = st.sidebar.multiselect(
    "Capitals to show",
    options=list(CAPITALS.keys()),
    default=list(CAPITALS.keys()),
)
include_support = st.sidebar.toggle(
    "Include supporting indicators (e.g. patents, emissions)", value=True
)
highlight_imputed = st.sidebar.toggle("Highlight imputed years", value=True)

# filter
view = data_idx.query(
    "Year >= @yr_range[0] and Year <= @yr_range[1] and Capital in @capitals_selected"
).copy()
if not include_support:
    view = view[view["Role"] == "primary"]

if view.empty:
    st.warning("No data in the selected window with the current filters.")
    st.stop()

# ---- Display mode toggle ----
view_mode = st.radio(
    "Display mode",
    ["Indexed (relative to first valid year)", "Absolute values"],
    index=0,
    horizontal=True,
    help=(
        "Indexed mode sets the first observed value in the selected window to 100 "
        "for each indicator. Absolute mode shows raw values (monetary series in trillions US$, "
        "others in their native units)."
    ),
)

# ----------------- CLEAN KPI SNAPSHOT (PRIMARY INDICATORS ONLY) -----------------
st.subheader("Latest snapshot – primary indicators (one per capital)")

# Primary indicator names for each capital
PRIMARY_MAP = {
    "Manufactured": "Produced Capital (const US$)",
    "Human": "Human Capital (const US$)",
    "Natural": "Natural Capital (const US$)",
    "Knowledge": "R&D Expenditure (% of GDP)",
    "Social": VA_INDICATOR_NAME,
}

kpi_primary = []

for cap, ind_name in PRIMARY_MAP.items():
    subset = view[view["Indicator"] == ind_name]
    if subset.empty:
        continue
    row = subset.sort_values("Year").iloc[-1]

    # look up unit from metadata (safe)
    unit_series = meta.loc[meta["Indicator"] == ind_name, "Unit"]
    unit = unit_series.iloc[0] if not unit_series.empty else ""

    val = row["Value"]
    if isinstance(unit, str) and "trillions" in unit:
        val = val / 1e12

    kpi_primary.append((cap, ind_name, val, unit))

# Fixed 5-column layout (or fewer if some missing)
kpi_cols = st.columns(max(1, len(kpi_primary)))
for col, item in zip(kpi_cols, kpi_primary):
    cap, name, value, unit = item
    help_text = f"{cap} capital — {unit}"
    if name == VA_INDICATOR_NAME:
        help_text += (
            " | WGI Voice & Accountability: civic freedoms, political participation, "
            "and institutional quality (−2.5 = weak, +2.5 = strong)."
        )

    col.metric(
        label=name,
        value=f"{value:,.2f}",
        help=help_text,
    )

# Optional: show supporting indicators' latest values
with st.expander("See supporting indicators (secondary metrics)"):
    latest_all = (
        view.sort_values(["Indicator", "Year"])
        .groupby("Indicator")
        .tail(1)
        .reset_index(drop=True)
    )
    support_latest = latest_all[~latest_all["Indicator"].isin(PRIMARY_MAP.values())][
        ["Indicator", "Capital", "Year", "Value"]
    ]
    st.dataframe(support_latest, use_container_width=True)

# ----------------- WEALTH COMPOSITION (PIE + BAR) -----------------
st.subheader("Wealth composition (latest year)")

# Only wealth stocks from Wealth Accounts, not ONS or non-monetary indicators
wealth_stock_indicators = [
    "Produced Capital (const US$)",
    "Human Capital (const US$)",
    "Natural Capital (const US$)",
]

latest_wealth = (
    view[view["Indicator"].isin(wealth_stock_indicators)]
    .sort_values(["Indicator", "Year"])
    .groupby("Indicator")
    .tail(1)
)

comp = latest_wealth.copy()
if not comp.empty:
    comp["Value_trillions"] = comp["Value"] / 1e12

    fig_comp_pie = px.pie(
        comp,
        values="Value_trillions",
        names="Capital",
        title="Composition of manufactured, human and natural capital (trillions US$)",
    )
    st.plotly_chart(fig_comp_pie, use_container_width=True)

    fig_comp_bar = px.bar(
        comp,
        x="Capital",
        y="Value_trillions",
        title="Capital stocks compared (trillions US$)",
        text_auto=".2f",
    )
    fig_comp_bar.update_layout(yaxis_title="Trillions of US$ (constant)")
    st.plotly_chart(fig_comp_bar, use_container_width=True)
else:
    st.info("No Wealth Accounts capital-stock data available for composition overview.")

# ----------------- TRAJECTORIES -----------------

# For the chart we need units to scale absolute values correctly
chart_df = view.merge(meta[["Indicator", "Unit"]], on="Indicator", how="left")

def _scale_for_display(row):
    unit = row["Unit"] or ""
    val = row["Value"]
    # scale only monetary Wealth Accounts stocks
    if isinstance(unit, str) and "trillions" in unit:
        return val / 1e12
    return val

chart_df["Value_display"] = chart_df.apply(_scale_for_display, axis=1)

if view_mode.startswith("Indexed"):
    st.subheader("trajectories (normalised to 100)")
    y_col = "Index"
    y_title = "Index (first valid year in window = 100)"
else:
    st.subheader("Absolute trajectories (mixed units)")
    y_col = "Value_display"
    y_title = "Absolute value (trillions US$, %, counts, index, etc.)"

line_kwargs = {
    "x": "Year",
    "y": y_col,
    "color": "Indicator",
    "template": "plotly_white",
}

if highlight_imputed:
    line_kwargs["line_dash"] = "Imputed"

fig = px.line(chart_df, **line_kwargs)
fig.update_layout(
    legend_title_text="Indicator",
    yaxis_title=y_title,
)
st.plotly_chart(fig, use_container_width=True)

# Info specifically when Social capital is selected
if "Social" in capitals_selected:
    st.info(
        "**Voice & Accountability (WGI)** and **Government Effectiveness (WGI)** "
        "are governance indices measuring civic freedoms, democratic participation, "
        "institutional quality and state capacity on a −2.5 to +2.5 scale. "
        "Higher scores indicate stronger institutional social capital."
    )

if view_mode.startswith("Indexed"):
    st.markdown(
        """
**Interpretation note:** Each series is indexed so that the first *observed* value
in the selected window equals 100. Values above 100 indicate growth relative to that
starting point; values below 100 indicate decline.  
Dashed lines show years where values have been **interpolated** between
irregular observations (e.g. Wealth Accounts vintages).
        """
    )
else:
    st.markdown(
        """
**Interpretation note (absolute mode):** This chart shows raw values, not indices.
Monetary series are displayed in **trillions of constant US dollars**; other indicators
are shown in their native units (% of GDP, counts, years, governance index, etc.).  
Units are therefore **not directly comparable across lines** in this view; use the indexed
view to compare growth/decline across capitals.
        """
    )

# ----------------- CAPITAL PROFILES (SMALL MULTIPLES) -----------------
st.subheader("Capital profiles (indexed trends by capital)")

cap_names = ["Manufactured", "Human", "Natural", "Knowledge", "Social"]
cols_profiles = st.columns(3)

for i, cap in enumerate(cap_names):
    with cols_profiles[i % 3]:
        subset = chart_df[chart_df["Capital"] == cap]
        if subset.empty:
            continue
        st.markdown(f"#### {cap} capital")
        fig_cap = px.line(
            subset,
            x="Year",
            y="Index",
            color="Indicator",
            template="plotly_white",
        )
        fig_cap.update_layout(
            showlegend=(i == 0),
            yaxis_title="Index (first valid year = 100)",
        )
        st.plotly_chart(fig_cap, use_container_width=True)

# ----------------- DECOUPLING PLOT: GDP PER CAPITA VS NATURAL CAPITAL -----------------
st.subheader("Decoupling view: GDP per capita vs natural capital")

if not gdp_df.empty:
    nat = data_idx[
        (data_idx["Capital"] == "Natural")
        & (data_idx["Indicator"] == "Natural Capital (const US$)")
    ][["Year", "Index"]].rename(columns={"Index": "Natural_Index"})
    dec = pd.merge(gdp_df, nat, on="Year", how="inner")
    dec = dec[(dec["Year"] >= yr_range[0]) & (dec["Year"] <= yr_range[1])]

    if not dec.empty:
        dec_melt = dec.melt(
            id_vars="Year",
            value_vars=["GDP_Index", "Natural_Index"],
            var_name="Series",
            value_name="Index",
        )
        dec_melt["Series"] = dec_melt["Series"].map(
            {"GDP_Index": "GDP per capita (real)", "Natural_Index": "Natural capital stock"}
        )
        fig_dec = px.line(
            dec_melt,
            x="Year",
            y="Index",
            color="Series",
            template="plotly_white",
        )
        fig_dec.update_layout(
            yaxis_title="Index (first available year = 100)",
        )
        st.plotly_chart(fig_dec, use_container_width=True)
        st.markdown(
            """
This chart compares **GDP per capita** (real, constant prices) with **natural capital**
for the UK. Rising GDP per capita alongside flat or declining natural capital suggests
that economic growth has not been fully decoupled from the use of environmental assets.
            """
        )
    else:
        st.info("Not enough overlapping data to plot GDP per capita vs natural capital for this window.")
else:
    st.info("GDP per capita data not available from the World Bank API.")

# ----------------- TRENDS & THREATS -----------------
st.subheader("Trends & potential threats")

trends = compute_trends(view)
if not trends.empty:
    # sort with threats first
    trends = trends.sort_values(["Threat_Flag", "Index_Slope_per_year"], ascending=[False, True])
    st.dataframe(trends, use_container_width=True)

    threats = trends[trends["Threat_Flag"]]
    if not threats.empty:
        st.warning(
            "⚠️ Potential sustainability threats (declining stocks with large drawdowns): "
            + ", ".join(threats["Indicator"].tolist())
        )
else:
    st.info("Not enough data in the selected window to compute trends.")

# ----------------- HOW TO READ -----------------
st.markdown(
    """
### How to read these results

The dashboard shows whether each of the **five capitals** is **accumulating**,
**stagnating**, or **depleting** over time.

#### **Use Indexed Mode (recommended) to compare trends**
- **> 100** → capital increasing relative to baseline  
- **< 100** → capital decreasing  
- Dashed lines indicate interpolated years (mainly Wealth Accounts)

#### **Interpretation Guide**
- **Manufactured capital increasing** but **natural capital flat/declining**  
  → classic **weak sustainability** pattern.  
- **Human capital** rising slowly with **life expectancy increasing**  
  → improvements in population health & skills.  
- **Knowledge capital** mixed: R&D ↑ but patents ↓  
  → innovation input/output divergence.  
- **Social capital** volatile: VA and GE fluctuate around political events  
  → institutional uncertainty & governance stress.  
- **Natural pressures** rising (CO₂ per capita)  
  → suggests environmental constraints despite other capital gains.

#### **Decoupling Chart**
Shows whether the UK economy can grow (**GDP per capita**) without degrading
**natural capital**.  
If GDP rises while natural capital stagnates or falls, growth is only **weakly** sustainable.

#### **Threat Table**
Flags indicators showing:
- Persistent decline  
- Large drawdowns  
- Negative slopes over long periods  

These represent deterioration in the **productive base of well-being**.
    """
)

# ----------------- METADATA -----------------
with st.expander("Indicator definitions and sources"):
    st.dataframe(meta, use_container_width=True)

# ----------------- DOWNLOAD FILTERED DATA -----------------
merged_for_download = view.merge(
    meta[["Indicator", "Unit", "Source"]], on="Indicator", how="left"
)
csv_data = merged_for_download.to_csv(index=False)
st.download_button(
    "Download filtered data (CSV)",
    data=csv_data,
    file_name="uk_inclusive_wealth_filtered.csv",
    mime="text/csv",
)
