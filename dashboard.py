"""
UK Inclusive Wealth Dashboard
-----------------------------

This Streamlit app implements a five-capitals, inclusive-wealth style dashboard
for the United Kingdom, using:

- World Bank Wealth Accounts (produced, human, natural capital)
- World Development Indicators (R&D, education, emissions, forest area, etc.)
- Worldwide Governance Indicators (Voice & Accountability, Government Effectiveness)
- ONS Capital Stock series (more detailed manufactured capital)

The dashboard is structured to match an academic assignment brief:
- 5 capitals
- At least one primary indicator per capital
- Supporting indicators to characterise each capital
- Indexed vs absolute views
- A decoupling view (where data allow)
- Identification of threats via trend analysis

This version is intentionally explicit and commented for assessment readability.
"""

# =============================================================================
# 1. IMPORTS AND BASIC CONFIGURATION
# =============================================================================

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

# ----------------- GLOBAL CONSTANTS -----------------
COUNTRY = "GBR"          # ISO3 code for the United Kingdom
MIN_YEAR = 2000          # Base year for analysis and indexing
PLOTLY_TEMPLATE = "plotly_dark"  # Keep chart styling aligned with dark theme


# =============================================================================
# 2. GLOBAL HELPER: NUMBER FORMATTER
# =============================================================================

def fmt(x):
    """
    Format numeric values with thousands separators and 2 decimal places.

    This is used to ensure visual consistency across KPIs, tables and charts.
    """
    if isinstance(x, (int, float)):
        return f"{x:,.2f}"
    return x


# =============================================================================
# 3. CAPITAL DEFINITIONS AND INDICATOR METADATA
# =============================================================================

"""
We define the indicator set for each of the five capitals in a dictionary.
Each entry specifies:

- the World Bank / WGI indicator code
- a human-readable name
- a unit (for display and scaling)
- a source description (for metadata tables and documentation)
"""

CAPITALS = {
    "Natural": {
        "primary": {
            "code": "NW.NCA.TO",
            "name": "Natural Capital (const US$)",
            "unit": "US$ (constant, trillions)",
            "source": "World Bank – Wealth Accounts",
        },
        "support": {
            "code": "AG.LND.FRST.ZS",
            "name": "Forest area (% of land area)",
            "unit": "% of land area",
            "source": "World Bank – WDI",
        },
        "support2": {
            "code": "EN.ATM.CO2E.PC",
            "name": "CO₂ emissions (metric tons per capita)",
            "unit": "Metric tons per capita",
            "source": "World Bank – WDI",
        },
    },
    "Human": {
        "primary": {
            "code": "NW.HCA.TO",
            "name": "Human Capital (const US$)",
            "unit": "US$ (constant, trillions)",
            "source": "World Bank – Wealth Accounts",
        },
        "support": {
            "code": "SP.DYN.LE00.IN",
            "name": "Life expectancy at birth (years)",
            "unit": "Years",
            "source": "World Bank – WDI",
        },
        "support2": {
            "code": "SE.XPD.TOTL.GD.ZS",
            "name": "Education expenditure (% of GDP)",
            "unit": "% of GDP",
            "source": "World Bank – WDI",
        },
    },
    "Manufactured": {
        "primary": {
            "code": "NW.PCA.TO",
            "name": "Produced Capital (const US$)",
            "unit": "US$ (constant, trillions)",
            "source": "World Bank – Wealth Accounts",
        },
        "support": {
            "code": "NE.GDI.FTOT.ZS",
            "name": "Gross fixed capital formation (% of GDP)",
            "unit": "% of GDP",
            "source": "World Bank – WDI",
        },
        # Additional manufactured capital indicator from ONS added separately
    },
    "Knowledge": {
        "primary": {
            "code": "GB.XPD.RSDV.GD.ZS",
            "name": "R&D Expenditure (% of GDP)",
            "unit": "% of GDP",
            "source": "World Bank – WDI",
        },
        "support": {
            "code": "IP.PAT.RESD",
            "name": "Patent Applications (resident)",
            "unit": "Number of applications",
            "source": "World Bank – WDI",
        },
    },
    "Social": {
        "primary": {
            "code": "VA.EST",
            "name": "Voice & Accountability (estimate)",
            "unit": "Index (−2.5 to 2.5)",
            "source": "Worldwide Governance Indicators",
        },
        "support": {
            "code": "GE.EST",
            "name": "Government Effectiveness (estimate)",
            "unit": "Index (−2.5 to 2.5)",
            "source": "Worldwide Governance Indicators",
        },
    },
}

VA_INDICATOR_NAME = CAPITALS["Social"]["primary"]["name"]


# =============================================================================
# 4. API HELPERS TO LOAD DATA
# =============================================================================

def wb_get_series(country: str, indicator: str, start: int, end: int,
                  retries: int = 3, pause: float = 0.8) -> pd.DataFrame:
    """
    Fetch a single World Bank indicator for a given country and date range.

    Parameters
    ----------
    country : str
        ISO3 country code.
    indicator : str
        World Bank indicator code.
    start : int
        Start year.
    end : int
        End year.
    retries : int
        Number of retries if API call fails.
    pause : float
        Delay between retries.

    Returns
    -------
    DataFrame with columns ['Year', 'Value'] or empty DataFrame if no data.
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/"
        f"{indicator}?date={start}:{end}&format=json&per_page=2000"
    )

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            js = r.json()
        except Exception:
            time.sleep(pause * (attempt + 1))
            continue

        if isinstance(js, list) and len(js) > 1 and isinstance(js[1], list):
            rows = []
            for d in js[1]:
                if d.get("value") is not None:
                    rows.append(
                        {"Year": int(d["date"]), "Value": float(d["value"])}
                    )
            return pd.DataFrame(rows)

        time.sleep(pause * (attempt + 1))

    return pd.DataFrame(columns=["Year", "Value"])


@st.cache_data(show_spinner=True)
def load_all_data():
    """
    Load and assemble all indicator series used in the dashboard.

    - Wealth Accounts & WDI via World Bank API
    - Voice & Accountability via local CSV
    - Net Capital Stock via ONS API

    Returns
    -------
    data_raw : DataFrame (long format)
        Columns: Year, Value, Capital, Role, Indicator, Code
    meta : DataFrame
        Indicator metadata (Indicator, Capital, Code, Unit, Source)
    """
    start = MIN_YEAR
    end = datetime.now().year

    frames = []
    meta_rows = []

    # A. World Bank indicators (except Voice & Accountability)
    for cap, spec in CAPITALS.items():
        for role in ["primary", "support", "support2"]:
            if role not in spec:
                continue

            ind = spec[role]

            # Exclude VA here; it is loaded from CSV below
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
                    "Indicator": ind["name"],
                    "Capital": cap,
                    "Code": ind["code"],
                    "Unit": ind["unit"],
                    "Source": ind["source"],
                }
            )

    # B. Voice & Accountability (Social capital primary) from CSV
    try:
        va = pd.read_csv("voice_accountability_uk.csv")
    except FileNotFoundError:
        st.error(
            "voice_accountability_uk.csv not found. "
            "Place it in the same directory as this script. "
            "It must contain at least columns: Year, Value."
        )
        return pd.DataFrame(), pd.DataFrame()

    if not {"Year", "Value"}.issubset(va.columns):
        st.error(
            "voice_accountability_uk.csv must contain columns: Year, Value."
        )
        return pd.DataFrame(), pd.DataFrame()

    va["Capital"] = "Social"
    va["Role"] = "primary"
    va["Indicator"] = VA_INDICATOR_NAME
    va["Code"] = CAPITALS["Social"]["primary"]["code"]
    frames.append(va)

    meta_rows.append(
        {
            "Indicator": VA_INDICATOR_NAME,
            "Capital": "Social",
            "Code": CAPITALS["Social"]["primary"]["code"],
            "Unit": CAPITALS["Social"]["primary"]["unit"],
            "Source": CAPITALS["Social"]["primary"]["source"],
        }
    )

    # C. ONS Net Capital Stock for manufactured capital
    try:
        ons_url = "https://api.ons.gov.uk/timeseries/MJU5/dataset/CAPSTK/data"
        resp = requests.get(
            ons_url,
            headers={"User-Agent": "UK-Inclusive-Wealth-Dashboard"},
            timeout=15,
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
                "Indicator": "Net Capital Stock (ONS – £m)",
                "Capital": "Manufactured",
                "Code": "ONS:MJU5",
                "Unit": "£ million",
                "Source": "ONS Capital Stock dataset (series MJU5)",
            }
        )
    except Exception:
        # This failure is non-fatal; we can still run without the ONS series.
        pass

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    data_raw = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(meta_rows).drop_duplicates()

    return data_raw, meta


@st.cache_data(show_spinner=True)
def load_gdp_per_capita():
    """
    Load GDP per capita (constant prices) for decoupling analysis.

    This is indexed to 100 at the first available year.
    """
    df = wb_get_series(COUNTRY, "NY.GDP.PCAP.KD", MIN_YEAR, datetime.now().year)
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("Year")
    base = df["Value"].iloc[0]
    df["GDP_Index"] = (df["Value"] / base) * 100
    return df


# =============================================================================
# 5. INDEXING AND TREND ANALYSIS HELPERS
# =============================================================================

def index_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each indicator series to a continuous annual series:

    - Create a complete annual index between min and max year for each indicator.
    - Interpolate missing values linearly.
    - Compute an index where the first non-missing value in the series = 100.
    - Flag 'Observed' vs 'Imputed' years using boolean columns (no NA).

    Returns
    -------
    DataFrame with additional columns:
    - Observed (bool)
    - Imputed (bool)
    - Index (float)
    """
    if df.empty:
        return df

    out = []

    for ind, g in df.groupby("Indicator"):
        g = g.sort_values("Year").copy()

        # Flag original observations
        g["Observed"] = g["Value"].notna()

        # Create full annual grid
        year_min = int(g["Year"].min())
        year_max = int(g["Year"].max())
        full = pd.DataFrame({"Year": range(year_min, year_max + 1)})

        # Merge onto full grid
        g = full.merge(
            g[["Year", "Value", "Observed", "Indicator", "Capital", "Role", "Code"]],
            on="Year",
            how="left",
        )

        # Fill metadata
        for col in ["Indicator", "Capital", "Role", "Code"]:
            g[col] = g[col].ffill().bfill()

        # Ensure Observed is plain bool with no NA
        g["Observed"] = g["Observed"].fillna(False).astype(bool)

        # Interpolate values
        g["Value"] = g["Value"].interpolate(method="linear", limit_direction="both")

        # Compute index (first valid value = 100)
        valid_mask = g["Value"].notna()
        if not valid_mask.any():
            continue

        base_val = g.loc[valid_mask, "Value"].iloc[0]
        if base_val == 0:
            continue

        g["Index"] = (g["Value"] / base_val) * 100

        # Imputed = not originally observed, also plain bool
        g["Imputed"] = (~g["Observed"]).astype(bool)

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def compute_trends(df_window: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple trend diagnostics for each indicator in the current window.

    Metrics:
    - Absolute change in level (%)
    - Compound Annual Growth Rate (%)
    - Slope of index per year
    - Maximum drawdown in index (%)
    - Threat flag: True if slope < 0 AND drawdown < -15%

    Returns
    -------
    DataFrame with user-friendly column names ready for display.
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

        abs_change = ((v1 - v0) / v0) * 100 if v0 else None
        cagr = ((v1 / v0) ** (1 / yrs) - 1) * 100 if v0 > 0 and v1 > 0 else None
        slope = (g["Index"].iloc[-1] - g["Index"].iloc[0]) / yrs
        drawdown = (g["Index"] / g["Index"].cummax() - 1).min() * 100
        threat = bool(slope < 0 and drawdown < -15)

        rows.append(
            {
                "Indicator": ind,
                "Start year": y0,
                "End year": y1,
                "Absolute change (%)": abs_change,
                "CAGR (%)": cagr,
                "Index slope (per year)": slope,
                "Max drawdown (%)": drawdown,
                "Threat flag": threat,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 6. LOAD DATA AND PREPARE MASTER DATAFRAMES
# =============================================================================

data_raw, meta = load_all_data()
gdp_df = load_gdp_per_capita()

if data_raw.empty:
    st.error("No data available. Please check the data sources and reload.")
    st.stop()

data_idx = index_series(data_raw)
if data_idx.empty:
    st.error("Indexing failed. Check for missing or invalid data.")
    st.stop()

year_min = int(data_idx["Year"].min())
year_max = int(data_idx["Year"].max())

# Determine whether monetised natural capital stock exists
nat_inds = data_idx[data_idx["Capital"] == "Natural"]["Indicator"].unique()
HAS_NAT_STOCK = "Natural Capital (const US$)" in nat_inds


# =============================================================================
# 7. UI: TITLE AND HIGH-LEVEL EXPLANATION
# =============================================================================

st.title("🇬🇧 UK Inclusive Wealth Dashboard")
st.caption(
    "An inclusive-wealth, five-capitals analysis for the United Kingdom. "
    "Data: World Bank Wealth Accounts & WDI; Worldwide Governance Indicators; ONS."
)

# ----------------- EXPLANATORY EXPANDERS -----------------

with st.expander("About this framework (five capitals & inclusive wealth)"):
    st.markdown(
        """
This dashboard applies the **Inclusive Wealth / Five Capitals framework**, inspired by
the Inclusive Wealth Reports and Matson et al. (2016).

It interprets sustainable development as the maintenance or growth of an economy’s
**productive base of well-being** across five broad asset classes:

---

### 1. Manufactured (Produced) Capital
Physical, human-made assets used to produce goods and services.  
Includes **infrastructure, machinery, buildings, equipment**.

**Indicators used:**
- **Produced Capital (constant US$)** — Wealth Accounts stock of produced assets.  
- **Net Capital Stock (ONS)** — detailed UK estimate (physical infrastructure).  
- **Gross Fixed Capital Formation (% of GDP)** — investment flow into produced assets.

---

### 2. Human Capital
The discounted value of skills, health, and productivity embodied in individuals.

**Indicators used:**
- **Human Capital (constant US$)** — Wealth Accounts stock of human capital.  
- **Life expectancy at birth** — population health outcome.  
- **Education expenditure (% of GDP)** — investment in human capital formation.

---

### 3. Natural Capital
Environmental assets that provide ecosystem services and resources.

**Indicators used:**
- **Natural Capital (constant US$)** *(not available for UK in this dataset)*  
- **Forest area (% of land area)** — ecosystem extent proxy.  
- **CO₂ emissions (tons per capita)** — environmental pressure indicator.

Because the monetised natural capital stock series is missing for the UK in the
Wealth Accounts API, **natural capital is characterised using forest extent and
emissions rather than a monetary stock**.

---

### 4. Knowledge Capital
Codified and tacit knowledge that drives innovation and productivity.

**Indicators used:**
- **R&D expenditure (% of GDP)** — innovation input (flow).  
- **Patent applications (resident)** — innovation output (stock of codified knowledge).

---

### 5. Social / Institutional Capital
Institutions, norms, and governance quality that underpin collective action.

**Indicators used:**
- **Voice & Accountability (WGI)** — civic freedoms, democratic participation, media freedom.  
- **Government Effectiveness (WGI)** — state capacity and policy implementation quality.

Both are measured on a **−2.5 to +2.5** governance scale.

---

### Why measure capitals?

- Sustainable development requires that the total **productive base of well-being**
  is **non-declining** over time.
- Focusing solely on GDP can hide **depletion of natural or social capital**.
- This dashboard allows us to see whether the UK is accumulating or eroding
  each capital, and whether growth is **weakly or strongly sustainable**.
        """
    )

with st.expander("What does 'Voice & Accountability' measure?"):
    st.markdown(
        """
**Voice & Accountability (V&A)** is one of the **Worldwide Governance Indicators**.
Here it is used as the **primary proxy for institutional social capital**.

It captures perceptions of:
- Freedom of expression, media, and association  
- Electoral integrity and political rights  
- Citizen participation and accountability of government  
- Civil liberties and democratic checks and balances  

The scale runs from **−2.5 (weak governance)** to **+2.5 (strong governance)**.

This is not a measure of interpersonal trust or community cohesion. It is an
institutional indicator of whether citizens can effectively **voice preferences**
and hold government to account — a core aspect of long-run sustainability.
        """
    )

with st.expander("Data & methods (stocks, flows, interpolation, limitations)"):
    st.markdown(
        """
### Data sources

- **World Bank Wealth Accounts (CWON)** — monetary stocks of produced, human,
  and (in principle) natural capital.  
- **World Development Indicators (WDI)** — R&D, education, forest area, CO₂,
  patents, GFCF, life expectancy, GDP per capita.  
- **Worldwide Governance Indicators (WGI)** — Voice & Accountability, Government
  Effectiveness.  
- **ONS Capital Stock (MJU5)** — detailed UK estimate of net capital stock.

---

### Indicator types

We combine:

- **Stock indicators**: e.g. produced capital, human capital, net capital stock.  
- **Flow indicators**: e.g. education expenditure, R&D, GFCF.  
- **Pressure indicators**: e.g. CO₂ emissions per capita.  
- **Ecosystem extent indicators**: e.g. forest area (% of land).  
- **Institutional indicators**: governance indices from WGI.

This mix is standard in applied inclusive-wealth analysis, where no single
stock measure perfectly captures each capital.

---

### Interpolation and indexing

- Wealth Accounts are reported at **irregular intervals** (often every 5 years).  
- To create annual series, we **linearly interpolate** between observed years.  
- The dashboard flags interpolated years using a **dashed line** when the
  "Highlight imputed years" option is enabled.  
- Each series is also converted into an **index** (first valid value = 100) to
  allow comparison of growth/decline across units.

---

### Units and comparability

- Monetary capital stocks are displayed in **trillions of constant US dollars**.  
- Other indicators retain their natural units (%, years, index, counts).  
- The **Indexed view** is therefore the main tool to compare trajectories across
  capitals; the **Absolute view** is useful for magnitude.

---

### Key limitations

- **Natural capital stock (monetised)** is missing for the UK in the Wealth
  Accounts API, so natural capital is proxied via forest area and CO₂.  
- Capital stocks are not expressed **per capita**; population growth is not
  explicitly controlled for.  
- Social capital is proxied using **institutional quality**, not social networks
  or trust directly.  
- Some important ecosystem services (e.g. cultural services, biodiversity) are
  not monetised in CWON, so natural capital is **under-represented**.
        """
    )


# =============================================================================
# 8. SIDEBAR: CONTROLS
# =============================================================================

st.sidebar.header("Controls")

# Year range slider
yr_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(max(year_min, MIN_YEAR), year_max),
    help="Filter all charts and metrics to a specific time window.",
)

# Capital selection
capitals_selected = st.sidebar.multiselect(
    "Capitals to show",
    options=list(CAPITALS.keys()),
    default=list(CAPITALS.keys()),
    help="Select which capitals to display in the charts.",
)

# Include supporting indicators?
include_support = st.sidebar.toggle(
    "Include supporting indicators (patents, emissions, etc.)",
    value=True,
)

# Highlight interpolated years?
highlight_imputed = st.sidebar.toggle(
    "Highlight imputed years (dashed lines)",
    value=True,
)

# Filter master indexed data by year and capital choice
view = data_idx.query(
    "Year >= @yr_range[0] and Year <= @yr_range[1] and Capital in @capitals_selected"
).copy()

if not include_support:
    view = view[view["Role"] == "primary"]

if view.empty:
    st.warning("No data available in the selected window with current filters.")
    st.stop()

# Display mode: indexed vs absolute
view_mode = st.radio(
    "Display mode",
    ["Indexed (relative to first valid year)", "Absolute values"],
    index=0,
    horizontal=True,
    help=(
        "Indexed mode: each series is normalised so its first valid value in the "
        "selected window equals 100. Absolute mode: raw values (monetary series in "
        "trillions US$, others in native units)."
    ),
)


# =============================================================================
# 9. KPI SNAPSHOT (ONE PRIMARY INDICATOR PER CAPITAL)
# =============================================================================

st.subheader("Latest snapshot – primary indicator for each capital")

# Map each capital to its primary indicator name
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
        # For Natural capital, this will be empty because the Wealth Accounts
        # stock series is not available for the UK.
        continue

    # Latest year in the selected window
    row = subset.sort_values("Year").iloc[-1]

    unit_series = meta.loc[meta["Indicator"] == ind_name, "Unit"]
    unit = unit_series.iloc[0] if not unit_series.empty else ""

    val = row["Value"]
    # Scale monetary series in trillions
    if isinstance(unit, str) and "trillions" in unit:
        val = val / 1e12

    kpi_primary.append((cap, ind_name, val, unit))

# Display KPIs in a single row of columns
if kpi_primary:
    kpi_cols = st.columns(len(kpi_primary))
    for col, item in zip(kpi_cols, kpi_primary):
        cap, name, value, unit = item
        help_text = f"{cap} capital — {unit}"
        if name == VA_INDICATOR_NAME:
            help_text += (
                " | WGI Voice & Accountability: civic freedoms, democratic participation, "
                "and institutional quality (−2.5 = weak, +2.5 = strong)."
            )

        col.metric(
            label=name,
            value=fmt(value),
            help=help_text,
        )
else:
    st.info(
        "No primary capital stock data available for the selected filters. "
        "This reflects missing Wealth Accounts series for this country."
    )

# Supporting indicators: latest values only
with st.expander("Supporting indicators – latest values by series"):
    latest_all = (
        view.sort_values(["Indicator", "Year"])
        .groupby("Indicator")
        .tail(1)
        .reset_index(drop=True)
    )
    support_latest = latest_all[~latest_all["Indicator"].isin(PRIMARY_MAP.values())][
        ["Indicator", "Capital", "Year", "Value"]
    ].copy()
    support_latest["Value"] = support_latest["Value"].apply(fmt)
    st.dataframe(support_latest, width="stretch")


# =============================================================================
# 10. WEALTH COMPOSITION (PIE + BAR, WHERE POSSIBLE)
# =============================================================================

st.subheader("Wealth composition (latest year – Wealth Accounts stocks)")

# Wealth stocks expected from the Wealth Accounts
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

if comp.empty:
    st.info(
        "Wealth Accounts capital stock series (produced, human, natural) are not all "
        "available for the UK / selected window. A full wealth composition chart cannot "
        "be shown. This particularly reflects the absence of a monetised natural "
        "capital stock series in the API."
    )
else:
    comp["Value_trillions"] = comp["Value"] / 1e12

    fig_comp_pie = px.pie(
        comp,
        values="Value_trillions",
        names="Capital",
        title="Composition of capital stocks (trillions US$)",
        template=PLOTLY_TEMPLATE,
    )
    fig_comp_pie.update_traces(texttemplate="%{percent:.1%}")
    st.plotly_chart(fig_comp_pie, width="stretch")

    fig_comp_bar = px.bar(
        comp,
        x="Capital",
        y="Value_trillions",
        title="Capital stock levels (trillions US$)",
        text_auto=".2f",
        template=PLOTLY_TEMPLATE,
    )
    fig_comp_bar.update_layout(yaxis_title="Trillions of US$ (constant)")
    fig_comp_bar.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig_comp_bar, width="stretch")


# =============================================================================
# 11. TRAJECTORIES: INDEXED VS ABSOLUTE
# =============================================================================

st.subheader("Trajectories across capitals")

# Attach units for scaling in absolute view
chart_df = view.merge(meta[["Indicator", "Unit"]], on="Indicator", how="left")

def _scale_for_display(row):
    """
    Scale monetary Wealth Accounts stocks into trillions, leave others unchanged.
    """
    unit = row["Unit"] or ""
    val = row["Value"]
    if isinstance(unit, str) and "trillions" in unit:
        return val / 1e12
    return val

chart_df["Value_display"] = chart_df.apply(_scale_for_display, axis=1)

# Ensure Imputed is clean bool (defensive, though index_series already did this)
if "Imputed" in chart_df.columns:
    chart_df["Imputed"] = chart_df["Imputed"].fillna(False).astype(bool)

if view_mode.startswith("Indexed"):
    y_col = "Index"
    y_title = "Index (100 = first valid year in selected window)"
    st.markdown(
        """
**Indexed mode:** each series is normalised so that its first observed value in the
selected window equals 100. Values above 100 indicate growth; values below 100
indicate decline relative to that baseline.
        """
    )
else:
    y_col = "Value_display"
    y_title = "Absolute value (trillions US$, %, counts, index, etc.)"
    st.markdown(
        """
**Absolute mode:** series are displayed in their native units. Monetary stocks are shown
in **trillions of constant US dollars**; other indicators in %, years, counts, or index
units. Units are therefore not directly comparable across lines in this view.
        """
    )

line_kwargs = {
    "x": "Year",
    "y": y_col,
    "color": "Indicator",
    "template": PLOTLY_TEMPLATE,
}

if highlight_imputed:
    line_kwargs["line_dash"] = "Imputed"

fig = px.line(chart_df, **line_kwargs)
fig.update_layout(
    legend_title_text="Indicator",
    yaxis_title=y_title,
)
fig.update_yaxes(tickformat=".2f")
st.plotly_chart(fig, width="stretch")

# Special reminder when Social capital is in view
if "Social" in capitals_selected:
    st.info(
        "Social capital here is proxied via **Voice & Accountability** and "
        "**Government Effectiveness** from the Worldwide Governance Indicators. "
        "Scores are on a −2.5 to +2.5 scale, where higher values indicate stronger "
        "institutional social capital."
    )


# =============================================================================
# 12. CAPITAL PROFILES: SMALL MULTIPLE PLOTS
# =============================================================================

st.subheader("Capital profiles – indexed trends by capital")

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
            template=PLOTLY_TEMPLATE,
        )
        fig_cap.update_layout(
            showlegend=True,
            legend_title_text="Indicator",
            yaxis_title="Index (100 = first valid year)",
        )
        fig_cap.update_yaxes(tickformat=".2f")
        st.plotly_chart(fig_cap, width="stretch")


# =============================================================================
# 13. DECOUPLING VIEW: GDP PER CAPITA VS NATURAL CAPITAL
# =============================================================================

st.subheader("Decoupling view: GDP per capita vs natural capital")

if not gdp_df.empty and HAS_NAT_STOCK:
    # Monetised natural capital stock is available (NOT the case for UK in current data,
    # but this logic is included for completeness and future generalisation).
    nat = data_idx[
        (data_idx["Capital"] == "Natural")
        & (data_idx["Indicator"] == "Natural Capital (const US$)")
    ][["Year", "Index"]].rename(columns={"Index": "Natural_Index"}).copy()

    gdp_dec = gdp_df.copy()
    gdp_dec["Year"] = gdp_dec["Year"].astype(int)
    nat["Year"] = nat["Year"].astype(int)

    overlap_years = sorted(set(gdp_dec["Year"]) & set(nat["Year"]))

    if overlap_years:
        year_min_sel, year_max_sel = yr_range
        valid_years = [y for y in overlap_years if year_min_sel <= y <= year_max_sel]

        if valid_years:
            gdp_window = gdp_dec[gdp_dec["Year"].isin(valid_years)]
            nat_window = nat[nat["Year"].isin(valid_years)]
            dec = pd.merge(gdp_window, nat_window, on="Year", how="inner")

            if not dec.empty:
                dec_melt = dec.melt(
                    id_vars="Year",
                    value_vars=["GDP_Index", "Natural_Index"],
                    var_name="Series",
                    value_name="Index",
                )
                dec_melt["Series"] = dec_melt["Series"].map(
                    {
                        "GDP_Index": "GDP per capita (real, constant prices)",
                        "Natural_Index": "Natural capital stock (Wealth Accounts)",
                    }
                )

                fig_dec = px.line(
                    dec_melt,
                    x="Year",
                    y="Index",
                    color="Series",
                    template=PLOTLY_TEMPLATE,
                )
                fig_dec.update_layout(
                    yaxis_title="Index (100 = first available year)",
                )
                fig_dec.update_yaxes(tickformat=".2f")
                st.plotly_chart(fig_dec, width="stretch")

                st.markdown(
                    """
This chart compares **GDP per capita** with **monetised natural capital stock**.
Strong decoupling would imply rising GDP per capita alongside stable or increasing
natural capital. Weak sustainability appears when GDP rises but natural capital
stagnates or declines.
                    """
                )
            else:
                st.info(
                    "No overlapping rows between GDP per capita and natural capital stock "
                    "after merging for the selected range."
                )
        else:
            st.info(
                "No overlapping years between GDP per capita and natural capital stock "
                "within the selected year range."
            )
    else:
        st.info(
            "GDP per capita and natural capital stock have disjoint coverage periods; "
            "a decoupling chart cannot be computed."
        )

elif not gdp_df.empty and not HAS_NAT_STOCK:
    # This is the actual case for the UK in the current API data.
    st.info(
        "A monetised **natural capital stock** series from the World Bank Wealth Accounts "
        "is not available for the UK in the current dataset. As a result, a formal GDP–"
        "natural capital decoupling chart cannot be shown.\n\n"
        "Instead, the **Natural capital profiles above** (forest area and CO₂ emissions "
        "per capita) should be used to infer whether economic growth appears to be "
        "associated with environmental pressure or decoupling."
    )
else:
    st.info("GDP per capita data are not available from the World Bank API.")


# =============================================================================
# 14. TRENDS & POTENTIAL THREATS TABLE
# =============================================================================

st.subheader("Trends & potential threats (by indicator)")

trends = compute_trends(view)

if not trends.empty:
    # Format numeric columns to 2 dp
    numeric_cols = [
        "Absolute change (%)",
        "CAGR (%)",
        "Index slope (per year)",
        "Max drawdown (%)",
    ]
    for col in numeric_cols:
        trends[col] = trends[col].apply(fmt)

    # Sort to show threats first, then by slope
    trends = trends.sort_values(
        ["Threat flag", "Index slope (per year)"],
        ascending=[False, True],
    )

    # Optional explicit column order for clarity
    columns_order = [
        "Indicator",
        "Start year",
        "End year",
        "Absolute change (%)",
        "CAGR (%)",
        "Index slope (per year)",
        "Max drawdown (%)",
        "Threat flag",
    ]
    trends = trends[columns_order]

    st.dataframe(trends, width="stretch", hide_index=True)

    threats = trends[trends["Threat flag"] == True]
    if not threats.empty:
        st.warning(
            "⚠️ Potential sustainability threats identified in: "
            + ", ".join(threats["Indicator"].tolist())
        )
else:
    st.info("Not enough data in the selected window to compute trends.")


# =============================================================================
# 15. INTERPRETATION GUIDE
# =============================================================================

st.markdown(
    """
### How to interpret this dashboard

1. **Use the indexed trajectories** to assess whether each capital is
   **accumulating, stagnant, or depleting** over time.
2. Compare **Manufactured vs Natural** capital profiles to assess whether the UK
   displays a classic **weak sustainability** pattern (produced capital rising
   while natural capital / ecosystem indicators flatline or decline).
3. Examine **Human capital** indicators to see whether improvements in wealth
   are underpinned by better health and education, or whether there is a
   reliance on produced capital only.
4. Study **Knowledge capital** to see whether innovation input (R&D) and output
   (patents) move together or diverge.
5. Look at **Social capital** via governance indices to understand whether
   institutional quality is stable or volatile over time.

The **Trends & potential threats** table helps identify indicators where
there are:
- Persistent negative slopes in the index, and  
- Large drawdowns from historical peaks.

These flag potential **deterioration in the productive base of well-being**,
even if GDP per capita is rising.
    """
)

# =============================================================================
# 16. METADATA TABLE AND DATA DOWNLOAD
# =============================================================================

with st.expander("Indicator definitions and data sources"):
    st.dataframe(meta, width="stretch")

merged_for_download = view.merge(
    meta[["Indicator", "Unit", "Source"]],
    on="Indicator",
    how="left",
)
csv_data = merged_for_download.to_csv(index=False)

st.download_button(
    "Download filtered data (CSV)",
    data=csv_data,
    file_name="uk_inclusive_wealth_filtered.csv",
    mime="text/csv",
)
