import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dubai Housing Market Analysis",
    page_icon="🏙️",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = r"E:\Databard\DLD\dld-clean.parquet"

@st.cache_data
def load_data():
    return pl.read_parquet(DATA_PATH)

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Dubai Residential Market Analysis")
st.markdown(
    "An analysis of **566,000 residential sales transactions** from the official "
    "Dubai Land Department (DLD) registry, covering 2021 to 2025. "
    "All price metrics use median throughout."
)
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
total_tx   = df.height
total_val  = df["actual_worth"].sum() / 1e9
median_sqm = df["meter_sale_price"].median()
median_deal = df["actual_worth"].median() / 1e6

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Total Value Transacted", f"AED {total_val:.1f}B")
col3.metric("Median Price / sqm", f"AED {median_sqm:,.0f}")
col4.metric("Median Deal Size", f"AED {median_deal:.2f}M")

st.divider()

# ── Chart 1: Heatmap ──────────────────────────────────────────────────────────
st.header("Price Heatmap by Neighborhood Over Time")
st.markdown(
    "Each row is a neighborhood, each column a year, colour represents **median price per sqm**. "
    "Warmer = more expensive. Shows 130+ neighborhoods across 5 years simultaneously."
)

with st.expander("Why this chart / What can we learn?"):
    st.markdown(
        """
- **Which areas are consistently expensive** — rows uniformly dark across all years
- **Which areas heated up fast** — rows transitioning from light to dark left-to-right
- **Market-wide trends** — if the entire heatmap darkens over time, prices rose broadly
- **Hidden gems** — areas still relatively cheap despite neighbours darkening fast
- **The two-speed market** — whether premium areas are pulling further ahead of mid-market
        """
    )

heatmap_data = (
    df.group_by(["neighborhood", "year"])
      .agg(pl.median("meter_sale_price").alias("median_sqm"))
      .sort(["neighborhood", "year"])
)
heatmap_pivot = heatmap_data.pivot(
    on="year", index="neighborhood", values="median_sqm", aggregate_function="first",
).sort("neighborhood")
years = sorted(df["year"].unique().to_list())
neighborhoods = heatmap_pivot["neighborhood"].to_list()
z_values = heatmap_pivot.select([pl.col(str(y)) for y in years]).to_numpy()

fig1 = go.Figure(data=go.Heatmap(
    z=z_values, x=[str(y) for y in years], y=neighborhoods,
    colorscale="YlOrRd", colorbar=dict(title="AED / sqm"), hoverongaps=False,
))
fig1.update_layout(
    title="Median Price per sqm by Neighborhood & Year",
    xaxis_title="Year", yaxis_title="Neighborhood",
    height=1400, yaxis=dict(tickfont=dict(size=10)),
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(
    """
**The market heated up broadly, not selectively.** Across almost every neighborhood, cells darken
noticeably from 2021 to 2025. This was not a story of one or two hot pockets pulling ahead, but a
market-wide appreciation cycle driven by post-pandemic demand, population growth, and a wave of
global capital inflows into Dubai real estate.

**"Island" stands out as the persistent price outlier.** The single darkest row belongs to "Island",
which traces back to the DLD area code Island 2, widely understood to be The World Islands, a
collection of ~300 private man-made islands off the Dubai coastline. With only 267 transactions over
5 years, this is an ultra-exclusive, ultra-illiquid market. Treat this row as directionally correct,
not statistically robust.

**Premium corridors are clearly visible.** Downtown Dubai, Dubai Marina, Palm Jumeirah, and Dubai
Creek Harbour form a consistently warm band. They were expensive in 2021 and got more expensive each year.

**The fastest heat-up story belongs to mid-market areas.** Neighborhoods like Jumeirah Village
Circle, Arjan, and Dubai South show a more dramatic colour shift than the premium tier, reflecting
the off-plan boom.

**Grey cells (missing data) are informative too.** Gaps mean not enough transactions occurred that
year to produce a reliable median. They are a signal of illiquidity, not an error.
    """
)
st.divider()

# ── Chart 2: Off-Plan vs Ready ────────────────────────────────────────────────
st.header("Off-Plan vs Ready Market Share Over Time")
st.markdown(
    "Monthly transaction volume split by sale type. Shows how the market composition "
    "has shifted from parity in 2021 to off-plan dominance by 2025."
)

with st.expander("Why this chart / What can we learn?"):
    st.markdown(
        """
- **How dominant off-plan has become** and whether that dominance is growing or reversing
- **Market confidence cycles** — off-plan surges when buyers trust future delivery
- **Developer pipeline pressure** — a sustained off-plan surge means a large supply wave incoming
- **Timing of market shifts** — pinpoint exactly when the balance tipped
        """
    )

split_trend = (
    df.group_by(["year", "month", "sale_type"])
      .agg(pl.len().alias("transactions"))
      .sort(["year", "month", "sale_type"])
      .with_columns(pl.date(pl.col("year"), pl.col("month"), pl.lit(1)).alias("date"))
)
fig2 = px.area(
    split_trend, x="date", y="transactions", color="sale_type",
    title="Off-Plan vs Ready: Monthly Market Share Over Time",
    labels={"transactions": "Share (%)", "date": "", "sale_type": "Type"},
    color_discrete_map={"Off-Plan": "#EF553B", "Ready": "#636EFA"},
    groupnorm="percent",
)
fig2.update_layout(hovermode="x unified", yaxis=dict(ticksuffix="%"))
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    """
**Off-plan started at parity and never looked back.** In early 2021, off-plan and ready transactions
were roughly equal. From that point the chart tells a single, unbroken story: off-plan's share
climbed steadily, reaching 75-80% by 2025. There was no reversal, no correction.

**The pace of the shift is the real story.** Each year, off-plan took a few more percentage points
of share. That kind of steady trend is more structurally significant than a sharp spike.

**The pipeline risk is now very real.** By 2025, roughly 75-80 out of every 100 residential
transactions were off-plan. Assuming delivery timelines of 2-3 years, an exceptionally large wave
of completed units is due between 2025 and 2028.

**Ready market held its absolute volume.** The shrinking blue band does not mean fewer ready
transactions in absolute terms. Total market volume expanded significantly, so the ready market
simply grew more slowly.
    """
)
st.divider()

# ── Chart 3: Bubble chart ─────────────────────────────────────────────────────
st.header("Neighborhood Pricing vs Volume")
st.markdown(
    "Each bubble is a neighborhood. X axis = median price per sqm, Y axis = total transactions, "
    "bubble size = total value transacted. Excludes Island, Trade Center, and Palm Jabal Ali."
)

EXCLUDE = ["Island", "Trade Center", "Palm Jabal Ali"]
bubble_data = (
    df.filter(pl.col("neighborhood").is_in(EXCLUDE).not_())
      .group_by("neighborhood")
      .agg([
          pl.len().alias("transactions"),
          pl.median("meter_sale_price").alias("median_sqm"),
          pl.sum("actual_worth").alias("total_value"),
      ])
      .with_columns((pl.col("total_value") / 1e9).alias("total_value_bn"))
      .sort("transactions", descending=True)
      .head(40)
)
fig3 = px.scatter(
    bubble_data,
    x="median_sqm", y="transactions",
    size="total_value_bn", color="median_sqm",
    text="neighborhood",
    title="Neighborhood: Median Price/sqm vs Transaction Volume (Top 40)",
    labels={
        "median_sqm": "Median AED/sqm",
        "transactions": "Total Transactions",
        "total_value_bn": "Total Value (AED bn)",
    },
    color_continuous_scale="YlOrRd",
    size_max=60,
)
fig3.update_traces(textposition="top center", textfont_size=9)
fig3.update_layout(height=700, coloraxis_showscale=False, hovermode="closest")
st.plotly_chart(fig3, use_container_width=True)
st.divider()

# ── Chart 4: Bedroom Mix ──────────────────────────────────────────────────────
st.header("Bedroom Mix by Neighborhood")
st.markdown(
    "100% stacked bar showing bedroom composition for the top 20 neighborhoods, "
    "split by Flats and Villas separately."
)

with st.expander("Why this chart / What can we learn?"):
    st.markdown(
        """
- **Investor vs end-user signal** — studio/1BR heavy = investors; 3BR+ = families
- **Developer strategy fingerprints** — off-plan areas often reflect one developer's product mix
- **Diversification of supply** — even spread across bedroom types signals resilient demand
        """
    )

BEDROOM_COLORS = {
    "Studio": "#636EFA", "1 B/R": "#00CC96", "2 B/R": "#FFA15A",
    "3 B/R": "#EF553B", "4+ B/R": "#AB63FA",
}

def bedroom_mix_fig(property_cat, top_n=20):
    top_hoods = (
        df.filter(pl.col("property_cat") == property_cat)
          .group_by("neighborhood").agg(pl.len().alias("transactions"))
          .sort("transactions", descending=True).head(top_n)
          ["neighborhood"].to_list()
    )
    mix = (
        df.filter(pl.col("property_cat") == property_cat)
          .filter(pl.col("neighborhood").is_in(top_hoods))
          .filter(pl.col("bedrooms") <= 4)
          .with_columns(
              pl.when(pl.col("bedrooms") == 0).then(pl.lit("Studio"))
                .when(pl.col("bedrooms") == 1).then(pl.lit("1 B/R"))
                .when(pl.col("bedrooms") == 2).then(pl.lit("2 B/R"))
                .when(pl.col("bedrooms") == 3).then(pl.lit("3 B/R"))
                .otherwise(pl.lit("4+ B/R")).alias("bedroom_label")
          )
          .group_by(["neighborhood", "bedroom_label"]).agg(pl.len().alias("count"))
    )
    totals = mix.group_by("neighborhood").agg(pl.col("count").sum().alias("total"))
    mix = (
        mix.join(totals, on="neighborhood")
           .with_columns((pl.col("count") / pl.col("total") * 100).round(1).alias("pct"))
    )
    hood_order = (
        df.filter(pl.col("property_cat") == property_cat)
          .filter(pl.col("neighborhood").is_in(top_hoods))
          .group_by("neighborhood").agg(pl.len().alias("total"))
          .sort("total", descending=False)["neighborhood"].to_list()
    )
    fig = px.bar(
        mix, x="pct", y="neighborhood", color="bedroom_label", orientation="h",
        title=f"Bedroom Mix — {property_cat}s (Top {top_n} by Volume)",
        labels={"pct": "Share (%)", "neighborhood": "", "bedroom_label": "Bedrooms"},
        category_orders={
            "neighborhood": hood_order,
            "bedroom_label": ["Studio", "1 B/R", "2 B/R", "3 B/R", "4+ B/R"],
        },
        color_discrete_map=BEDROOM_COLORS,
    )
    fig.update_layout(
        height=700, xaxis=dict(ticksuffix="%"), barmode="stack",
        legend=dict(traceorder="normal"), hovermode="y unified",
    )
    return fig

tab1, tab2 = st.tabs(["Flats", "Villas"])
with tab1:
    st.plotly_chart(bedroom_mix_fig("Flat"), use_container_width=True)
with tab2:
    st.plotly_chart(bedroom_mix_fig("Villa"), use_container_width=True)

st.markdown(
    """
**Flats: 1 B/R dominates almost everywhere, but the studio story is the interesting one.**
Dubai Maritime City, Jumeirah Lake Towers, and Downtown Dubai all show a meaningful studio band,
reflecting mature investor markets. Meydan stands out as almost entirely 1 B/R and 2 B/R with
virtually no studios, suggesting a more end-user oriented buyer profile.

**JVC's flat mix is surprisingly balanced.** The highest-volume area in Dubai shows a healthy spread
across Studio, 1 B/R, and 2 B/R. Developers built a diverse range of unit sizes, which is one
reason JVC attracted such broad demand year after year.

**Villas: the market splits cleanly into two camps.** Villanova and Dubai South are compact
townhouse communities (3 B/R dominant). Al Barari and Dubai Investment Park skew heavily to 4+ B/R,
confirming their positioning as large-format, high-end villa destinations.
    """
)
st.divider()

# ── Chart 5: Box plots ────────────────────────────────────────────────────────
st.header("Price Distribution by Bedroom Count — 2025")
st.markdown(
    "Box plots showing the full sale price distribution for each bedroom count in 2025, "
    "split by Flat and Villa. Restricted to 2025 to reflect current market conditions."
)

box_data = (
    df.filter(pl.col("bedrooms") <= 5)
      .filter(pl.col("year") == 2025)
      .with_columns(
          pl.when(pl.col("bedrooms") == 0).then(pl.lit("Studio"))
            .when(pl.col("bedrooms") == 1).then(pl.lit("1 B/R"))
            .when(pl.col("bedrooms") == 2).then(pl.lit("2 B/R"))
            .when(pl.col("bedrooms") == 3).then(pl.lit("3 B/R"))
            .when(pl.col("bedrooms") == 4).then(pl.lit("4 B/R"))
            .otherwise(pl.lit("5 B/R")).alias("bedroom_label")
      )
)
fig5 = px.box(
    box_data, x="bedroom_label", y="actual_worth", color="property_cat",
    title="Sale Price Distribution by Bedroom Count — 2025",
    labels={"actual_worth": "Sale Price (AED)", "bedroom_label": "Bedrooms", "property_cat": "Type"},
    category_orders={"bedroom_label": ["Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", "5 B/R"]},
    color_discrete_map={"Flat": "#636EFA", "Villa": "#EF553B"},
    points=False,
)
fig5.update_layout(height=600, yaxis=dict(tickformat=",.0f"), hovermode="x unified", boxmode="group")
st.plotly_chart(fig5, use_container_width=True)

st.markdown(
    """
**Studios and 1 B/R flats are tightly priced in 2025.** The boxes are compact and sit close to
zero on this scale, reflecting the most standardised and liquid segment of the market.

**2 B/R and 3 B/R flats show growing spread.** As bedroom count rises, the interquartile range
widens noticeably, reflecting the increasing role of location and finish quality at larger sizes.

**Villas at 4 B/R and below are more tightly distributed than flats.** Most villa buyers in 2025
transact within master-planned developments with standardised product, anchoring pricing.

**5 B/R villas are the single most volatile segment.** The whisker extends to 250 million AED while
the box sits barely above zero — ultra-luxury Palm and MBR City mansions define the ceiling.
    """
)
st.divider()

# ── Chart 6: Rolling volume ───────────────────────────────────────────────────
st.header("Residential Transaction Volume: Rolling 3-Month Average")
st.markdown(
    "Raw monthly transaction count with a 3-month rolling average overlay. "
    "The smoothed line removes seasonal noise to reveal the true market trend."
)

monthly = (
    df.group_by(["year", "month"]).agg(pl.len().alias("transactions"))
      .sort(["year", "month"])
      .with_columns(pl.date(pl.col("year"), pl.col("month"), pl.lit(1)).alias("date"))
      .with_columns(
          pl.col("transactions").rolling_mean(window_size=3, min_periods=1).alias("rolling_3m")
      )
)
fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=monthly["date"].to_list(), y=monthly["transactions"].to_list(),
    mode="lines", name="Monthly (raw)",
    line=dict(color="lightsteelblue", width=1), opacity=0.5,
))
fig6.add_trace(go.Scatter(
    x=monthly["date"].to_list(), y=monthly["rolling_3m"].to_list(),
    mode="lines", name="3-Month Rolling Avg",
    line=dict(color="#EF553B", width=2.5),
))
fig6.update_layout(
    title="Residential Transaction Volume: Rolling 3-Month Average",
    xaxis_title="", yaxis_title="Transactions", height=500,
    hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig6, use_container_width=True)

st.markdown(
    """
**The trend is almost perfectly unbroken from 2021 to 2026.** Starting from ~2,000 transactions per
month in early 2021, the rolling average climbed to a peak of ~16,500-17,000 in late 2025. Roughly
an 8x increase in five years.

**2021 to mid-2022: post-pandemic ignition.** Rising from 2k to ~4,500 per month, driven by
reopening, the 10-year Golden Visa, and Dubai's emergence as a remote-work destination.

**2022 to 2024: steady institutionalisation.** The rolling average ticked consistently higher. Not
a frenzy — a structurally growing market fed by continued population inflow and investor confidence.

**Mid-2024 to early 2025: the first serious wobble.** A visible dip from ~15,000 back toward 12,000.
A deceleration, not a reversal.

**The 2026 drop-off is a data artefact, not a crash.** Incomplete data for the final months means
the trailing edge always undercounts. Treat the 2026 right edge as noise.
    """
)
st.divider()

# ── Chart 7: YoY Growth ───────────────────────────────────────────────────────
st.header("Cumulative Price Growth 2021 to 2025 by Neighborhood")
st.markdown(
    "Horizontal bar chart ranking neighborhoods by cumulative median price per sqm growth "
    "from 2021 to 2025. Top 30 gainers and bottom 10 for context."
)

price_by_year = (
    df.filter(pl.col("neighborhood").is_in(EXCLUDE).not_())
      .filter(pl.col("year").is_in([2021, 2025]))
      .group_by(["neighborhood", "year"])
      .agg(pl.median("meter_sale_price").alias("median_sqm"))
)
both_years = (
    price_by_year.group_by("neighborhood").agg(pl.len().alias("n"))
      .filter(pl.col("n") == 2)["neighborhood"].to_list()
)
yoy = (
    price_by_year.filter(pl.col("neighborhood").is_in(both_years))
      .pivot(on="year", index="neighborhood", values="median_sqm", aggregate_function="first")
      .rename({"2021": "price_2021", "2025": "price_2025"})
      .with_columns(
          ((pl.col("price_2025") - pl.col("price_2021")) / pl.col("price_2021") * 100)
          .round(1).alias("growth_pct")
      )
      .sort("growth_pct", descending=True)
)
chart_data = pl.concat([yoy.head(30), yoy.tail(10)]).unique("neighborhood").sort("growth_pct", descending=True)
fig7 = px.bar(
    chart_data, x="growth_pct", y="neighborhood", orientation="h",
    title="Cumulative Price Growth 2021 to 2025 — Median AED/sqm (Top 30 + Bottom 10)",
    labels={"growth_pct": "Price Growth (%)", "neighborhood": ""},
    color="growth_pct", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
    text="growth_pct",
)
fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig7.update_layout(
    height=1100, xaxis=dict(ticksuffix="%"), coloraxis_showscale=False,
    yaxis=dict(categoryorder="total ascending"),
)
st.plotly_chart(fig7, use_container_width=True)

st.markdown(
    """
**Golf City and Falcon City of Wonders lead at 224% and 215%.** Mid-market villa communities on the
outer fringe that started 2021 at very low price per sqm. The enormous growth reflects a low base
effect: they had the most room to run.

**Jumeirah at 195% is the chart's most significant outlier.** An established, mature, land-scarce
beachside community that was already expensive in 2021. A near-tripling in four years is a genuine
scarcity story. There is simply no new land to add in Jumeirah.

**The mature premium core lagged on a percentage basis.** Dubai Marina grew 39.8%, Business Bay
77.8%. They were already expensive in 2021 and had less room to grow. In absolute AED per sqm terms
they may still have added more value than Golf City.

**Al Safouh at -2.7% is the only area that declined.** A statistical footnote driven by thin volume
rather than a meaningful market signal.
    """
)
st.divider()

# ── Chart 8: Sunburst ─────────────────────────────────────────────────────────
st.header("Transaction Breakdown: Neighborhood → Property Type → Sale Type")
st.markdown(
    "Three dimensions in one interactive chart. Inner ring = neighborhood, middle = property type, "
    "outer = sale type. Click any segment to zoom in."
)

TOP_N = 20
top_hoods = (
    df.group_by("neighborhood").agg(pl.len().alias("transactions"))
      .sort("transactions", descending=True).head(TOP_N)["neighborhood"].to_list()
)
sunburst_data = (
    df.filter(pl.col("neighborhood").is_in(top_hoods))
      .group_by(["neighborhood", "property_cat", "sale_type"])
      .agg(pl.len().alias("transactions"))
      .sort(["neighborhood", "property_cat", "sale_type"])
)
fig8 = px.sunburst(
    sunburst_data,
    path=["neighborhood", "property_cat", "sale_type"],
    values="transactions",
    title="Transaction Breakdown by Neighborhood, Property Type & Sale Type (Top 20)",
    color="neighborhood",
    color_discrete_sequence=px.colors.qualitative.Pastel,
)
fig8.update_layout(height=750)
fig8.update_traces(textinfo="label+percent entry")
st.plotly_chart(fig8, use_container_width=True)

st.markdown(
    """
**The market is highly concentrated.** Within the top 20, JVC, Business Bay, and MBR City command
the largest slices. A handful of master communities absorb a disproportionate share of all activity.

**JVC is almost entirely off-plan flats.** Roughly 75/25 off-plan to ready split, matching its
identity as Dubai's highest-volume affordable flat market.

**Business Bay is a flat-only market.** A dense, mixed-use urban district with no land for villa
development. The entire buyer universe is either end-user flat buyers or investor units.

**Villanova is the clearest pure-villa outlier.** The only top-20 neighborhood where villas account
for virtually all volume, and the villa portion is dominated by Ready transactions.

**The outer ring confirms market-wide off-plan dominance.** Green (Off-Plan) segments are
consistently larger than ready across nearly every neighborhood and property type combination.
    """
)
st.divider()

# ── Final Thoughts ────────────────────────────────────────────────────────────
st.header("Final Thoughts")
st.markdown(
    """
Seven charts, 566,000 transactions, five years of data. Here is what the Dubai residential market
actually looks like when you let the numbers speak.

**The macro story is simple: everything went up, and it did not stop.**
From 2021 to 2025, transaction volume grew roughly 8x and median prices per sqm roughly doubled
across the broad market. The rolling volume chart shows one of the cleanest sustained uptrends you
will find in any major real estate market over this period.

**Off-plan is not a segment of the market. It is the market.**
By 2025, roughly 75-80% of all residential transactions were off-plan. A large wave of completions
is due between 2025 and 2028. Whether the market absorbs it smoothly depends entirely on whether
the demand pipeline stays as deep as the supply pipeline.

**The fastest growth was not where you expect it.**
Golf City at 224%, Falcon City at 215%, Dubai Silicon Oasis at 173%. Mid-market and fringe
communities outpaced Palm Jumeirah and Dubai Marina by a wide margin. The real appreciation story
of this cycle belongs to the affordable outer belt.

**Jumeirah is the one exception that breaks the rule.**
The only established, land-scarce, premium area that also delivered near-200% price growth. No new
supply, genuine scarcity, ultra-high-net-worth demand. That combination is rare.

**JVC is Dubai's engine, not its showroom.**
The highest transaction volume, a balanced bedroom mix, broad buyer demographic, and sustained
off-plan demand. Any analysis of Dubai residential real estate that ignores JVC is missing the point.

**What this dataset cannot tell you** is whether current prices are sustainable. It describes what
happened, not what comes next. The pipeline risk is visible in the off-plan chart, the concentration
risk in the sunburst, and the scarcity premium in the YoY growth chart.

What the data does tell you is that between 2021 and 2025, Dubai was one of the most active and
consistently appreciating residential markets on the planet. This project documents that cycle in full.
    """
)

st.caption(
    "Data source: Dubai Land Department (DLD) official residential transaction registry. "
    "Analysis: 566,235 clean residential sales transactions, 2021-2025."
)
