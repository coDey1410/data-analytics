"""
Streamlit dashboard for Chicago Crime Data Analytics.
Provides exploratory analysis, statistical outlier detection, SQL views,
and a CRUD interface backed by the Flask API and SQLite3 database.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st

st.set_page_config(
    page_title="Chicago Crime Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom card styling for top-level summary metrics
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 12px 16px;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_DIR, "chicago_crime.db")


def determine_api_endpoint():
    configured_url = os.environ.get("FLASK_API_URL")
    if configured_url:
        return configured_url

    # Check port 5001 first (avoids macOS AirPlay on 5000), then check 5000
    for test_port in [5001, 5000]:
        candidate_url = f"http://127.0.0.1:{test_port}"
        try:
            resp = requests.get(f"{candidate_url}/", timeout=0.8)
            if resp.status_code == 200:
                return candidate_url
        except Exception:
            continue
    return "http://127.0.0.1:5001"


API_URL = determine_api_endpoint()


@st.cache_data(ttl=30)
def fetch_primary_crime_dataframe():
    # Load dataset joined with community area name from SQLite
    if not os.path.exists(DATABASE_FILE):
        return pd.DataFrame()

    conn = sqlite3.connect(DATABASE_FILE)
    query_sql = """
    SELECT 
        c.id,
        c.case_number,
        c.date,
        c.block,
        c.iucr_code,
        c.primary_type,
        c.crime_description,
        c.location_desc,
        c.arrest,
        c.domestic,
        c.beat_num,
        c.district_code,
        c.ward_no,
        c.community_code,
        c.fbi_code,
        c.year,
        c.month,
        c.day_of_week,
        c.hour,
        c.latitude,
        c.longitude,
        COALESCE(comm.community_name, 'Community ' || c.community_code) AS community_name
    FROM crimes c
    LEFT JOIN ref_communities comm ON c.community_code = comm.community_code
    ORDER BY c.date DESC;
    """
    df = pd.read_sql_query(query_sql, conn)
    conn.close()

    if not df.empty:
        df["arrest_bool"] = df["arrest"].astype(bool)
        df["domestic_bool"] = df["domestic"].astype(bool)
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["hour"] = df["hour"].astype(int)

    return df


@st.cache_data(ttl=60)
def fetch_database_summary_via_api():
    try:
        response = requests.get(f"{API_URL}/api/summary", timeout=2.5)
        if response.status_code == 200:
            return response.json().get("summary", {})
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_quality_report_via_api():
    try:
        response = requests.get(f"{API_URL}/api/data-quality", timeout=2.5)
        if response.status_code == 200:
            return response.json().get("report", {})
    except Exception:
        pass
    return None


# Sidebar controls
st.sidebar.title("Chicago Crime Analytics")
st.sidebar.caption("Python, Flask, Streamlit and SQLite3")

# Connection status indicator
backend_healthy = False
try:
    health_check = requests.get(f"{API_URL}/", timeout=1.0)
    backend_healthy = (health_check.status_code == 200)
except Exception:
    backend_healthy = False

if backend_healthy:
    st.sidebar.success(f"Flask API online at {API_URL}")
else:
    st.sidebar.warning("Flask API offline. Querying SQLite database directly.")

section_selection = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Crime Analysis",
        "Arrest Analysis",
        "Community Analysis",
        "Statistical Analysis",
        "Database Management",
        "Reports and Data Quality"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

full_crime_df = fetch_primary_crime_dataframe()

if full_crime_df.empty:
    st.error("SQLite database empty or missing. Please ensure app.py has initialized chicago_crime.db.")
    st.stop()

# Filter options
available_years = ["All"] + sorted(full_crime_df["year"].unique().tolist())
chosen_year = st.sidebar.selectbox("Year", available_years)

available_types = ["All"] + sorted(full_crime_df["primary_type"].unique().tolist())
chosen_type = st.sidebar.selectbox("Primary Type", available_types)

available_districts = ["All"] + sorted([int(d) for d in full_crime_df["district_code"].dropna().unique().tolist()])
chosen_district = st.sidebar.selectbox("District", available_districts)

community_options = ["All"] + sorted(full_crime_df["community_name"].dropna().unique().tolist())
chosen_community = st.sidebar.selectbox("Community Area", community_options)

chosen_arrest = st.sidebar.radio("Arrest Status", ["All", "Arrested Only", "No Arrest"], horizontal=True)

# Apply filters
filtered_df = full_crime_df.copy()

if chosen_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == int(chosen_year)]

if chosen_type != "All":
    filtered_df = filtered_df[filtered_df["primary_type"] == chosen_type]

if chosen_district != "All":
    filtered_df = filtered_df[filtered_df["district_code"] == int(chosen_district)]

if chosen_community != "All":
    filtered_df = filtered_df[filtered_df["community_name"] == chosen_community]

if chosen_arrest == "Arrested Only":
    filtered_df = filtered_df[filtered_df["arrest"] == 1]
elif chosen_arrest == "No Arrest":
    filtered_df = filtered_df[filtered_df["arrest"] == 0]

st.sidebar.caption(f"Showing {len(filtered_df):,} of {len(full_crime_df):,} records")


def display_kpi_card(title, value):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 1: Dashboard Overview
# ---------------------------------------------------------------------------
if section_selection == "Dashboard Overview":
    st.title("Chicago Crime Analytics Dashboard")
    st.markdown("High-level key performance indicators and primary distribution trends.")

    total_crimes = len(filtered_df)
    total_arrests = int(filtered_df["arrest"].sum()) if total_crimes > 0 else 0
    arrest_percentage = round((total_arrests * 100.0 / total_crimes), 2) if total_crimes > 0 else 0.0
    category_count = filtered_df["primary_type"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("Total Crime Records", f"{total_crimes:,}")
    with col2:
        display_kpi_card("Total Arrests", f"{total_arrests:,}")
    with col3:
        display_kpi_card("Arrest Percentage", f"{arrest_percentage}%")
    with col4:
        display_kpi_card("Crime Categories", f"{category_count}")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Yearly Crime Trend")
        yearly_tally = filtered_df.groupby("year")["id"].count().reset_index()
        yearly_tally.columns = ["Year", "Crime Count"]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.lineplot(
            data=yearly_tally,
            x="Year",
            y="Crime Count",
            marker="o",
            color="#1f77b4",
            linewidth=2.5,
            ax=ax
        )
        for _, row in yearly_tally.iterrows():
            ax.annotate(
                f"{int(row['Crime Count'])}",
                (row["Year"], row["Crime Count"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontweight="bold"
            )
        ax.set_title("Total Reported Crimes by Year", fontsize=11, pad=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Crime Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.subheader("Top 5 Crime Categories")
        top5_categories = filtered_df["primary_type"].value_counts().head(5).reset_index()
        top5_categories.columns = ["Primary Type", "Count"]

        fig, ax = plt.subplots(figsize=(7, 4.2))
        sns.barplot(
            data=top5_categories,
            y="Primary Type",
            x="Count",
            hue="Primary Type",
            legend=False,
            palette="Blues_r",
            ax=ax
        )
        for idx, row in top5_categories.iterrows():
            ax.text(row["Count"] + 1, idx, f"{int(row['Count'])}", va="center", fontweight="bold")
        ax.set_title("Top 5 Crime Categories", fontsize=11, pad=10)
        ax.set_xlabel("Incident Count")
        ax.set_ylabel("Primary Type")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info("Theft and battery represent the largest volume of offenses in this sample. The overall arrest rate is 24.90%.")


# ---------------------------------------------------------------------------
# Section 2: Crime Analysis
# ---------------------------------------------------------------------------
elif section_selection == "Crime Analysis":
    st.title("Crime Trends and Category Distribution")

    st.subheader("Crime Trend Over Years")
    yearly_counts = filtered_df["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(yearly_counts.index, yearly_counts.values, marker="s", color="#0d6efd", linewidth=2.5, markersize=7)
    for yr, cnt in yearly_counts.items():
        ax.annotate(f"{cnt:,}", (yr, cnt), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    ax.set_title("Yearly Crime Trend", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Reported Crimes")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    max_year = yearly_counts.idxmax() if not yearly_counts.empty else "N/A"
    st.info(f"Reported crimes in this sample are concentrated in {max_year} with {yearly_counts.max()} incidents. Earlier years reflect smaller sample cohorts.")

    st.markdown("---")

    st.subheader("Top 10 Crime Categories by Volume and Share")
    top10_series = filtered_df["primary_type"].value_counts().head(10)
    total_in_scope = len(filtered_df)

    category_table = pd.DataFrame({
        "Primary Type": top10_series.index,
        "Incident Count": top10_series.values,
        "Percentage (%)": np.round((top10_series.values * 100.0 / total_in_scope), 2)
    })

    col_chart, col_table = st.columns([1.3, 1.0])
    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=category_table,
            y="Primary Type",
            x="Incident Count",
            hue="Primary Type",
            legend=False,
            palette="mako",
            ax=ax
        )
        for i, val in enumerate(category_table["Incident Count"]):
            pct = category_table["Percentage (%)"].iloc[i]
            ax.text(val + 1, i, f"{val} ({pct}%)", va="center", fontsize=9, fontweight="bold")
        ax.set_title("Top 10 Crime Categories", fontsize=12)
        ax.set_xlabel("Incident Count")
        ax.set_ylabel("Primary Type")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.write("Breakdown Table:")
        st.dataframe(category_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Heatmap of Crime by Month and Day of Week")

    month_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    heat_df = filtered_df.copy()
    heat_df["Month_Name"] = heat_df["month"].map(month_names)

    pivot_matrix = heat_df.pivot_table(
        index="day_of_week",
        columns="Month_Name",
        values="id",
        aggfunc="count",
        fill_value=0
    )

    pivot_matrix = pivot_matrix.reindex(index=weekday_order, columns=[month_names[m] for m in month_order if month_names[m] in pivot_matrix.columns]).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Crime Incidents by Day of Week and Month", fontsize=12, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if not pivot_matrix.empty and pivot_matrix.values.sum() > 0:
        max_coord = np.unravel_index(np.argmax(pivot_matrix.to_numpy()), pivot_matrix.shape)
        peak_weekday = pivot_matrix.index[max_coord[0]]
        peak_month = pivot_matrix.columns[max_coord[1]]
        peak_val = pivot_matrix.iloc[max_coord[0], max_coord[1]]
        st.info(f"Highest frequency occurred on {peak_weekday} in {peak_month} with {peak_val} incidents.")


# ---------------------------------------------------------------------------
# Section 3: Arrest Analysis
# ---------------------------------------------------------------------------
elif section_selection == "Arrest Analysis":
    st.title("Arrest and Crime Outcomes")

    total_crimes = len(filtered_df)
    total_arrests = int(filtered_df["arrest"].sum()) if total_crimes > 0 else 0
    # Required calculation: arrest_rate = df['Arrest'].mean() * 100
    arrest_rate = (filtered_df["arrest"].mean() * 100) if total_crimes > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        display_kpi_card("Total Crimes Evaluated", f"{total_crimes:,}")
    with col2:
        display_kpi_card("Total Arrests Recorded", f"{total_arrests:,}")
    with col3:
        display_kpi_card("Overall Arrest Rate", f"{arrest_rate:.2f}%")

    st.markdown("---")

    col_chart_left, col_chart_right = st.columns([1.2, 1.0])

    with col_chart_left:
        st.subheader("Comparison of Arrest Rates across Crime Categories")
        cat_arrest_stats = filtered_df.groupby("primary_type").agg(
            total_cases=("id", "count"),
            arrests=("arrest", "sum")
        ).reset_index()
        cat_arrest_stats["arrest_rate"] = np.round((cat_arrest_stats["arrests"] * 100.0 / cat_arrest_stats["total_cases"]), 2)
        meaningful_cats = cat_arrest_stats[cat_arrest_stats["total_cases"] >= 5].sort_values(by="arrest_rate", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=meaningful_cats,
            y="primary_type",
            x="arrest_rate",
            hue="primary_type",
            legend=False,
            palette="crest",
            ax=ax
        )
        for i, val in enumerate(meaningful_cats["arrest_rate"]):
            total_ct = meaningful_cats["total_cases"].iloc[i]
            ax.text(val + 1, i, f"{val}% (n={total_ct})", va="center", fontsize=9, fontweight="bold")
        ax.set_title("Arrest Rate (%) by Category (Min 5 Cases)", fontsize=11)
        ax.set_xlabel("Arrest Percentage (%)")
        ax.set_ylabel("Primary Offense Category")
        ax.set_xlim(0, 115)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_chart_right:
        st.subheader("Arrest Outcome Proportion")
        arrest_split = pd.DataFrame({
            "Outcome": ["Arrest Made", "No Arrest"],
            "Count": [total_arrests, total_crimes - total_arrests]
        })

        fig, ax = plt.subplots(figsize=(5.5, 5))
        colors = ["#28a745", "#dc3545"]
        ax.pie(
            arrest_split["Count"],
            labels=arrest_split["Outcome"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Overall Arrest Ratio", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    st.subheader("Arrest Counts and Rates Over Years")
    yearly_arrest_trends = filtered_df.groupby("year").agg(
        crimes=("id", "count"),
        arrests=("arrest", "sum")
    ).reset_index()
    yearly_arrest_trends["arrest_rate"] = np.round((yearly_arrest_trends["arrests"] * 100.0 / yearly_arrest_trends["crimes"]), 2)

    col_trend1, col_trend2 = st.columns(2)
    with col_trend1:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(data=yearly_arrest_trends, x="year", y="arrests", hue="year", legend=False, palette="Blues", ax=ax)
        for i, val in enumerate(yearly_arrest_trends["arrests"]):
            ax.text(i, val + 1, str(int(val)), ha="center", fontweight="bold")
        ax.set_title("Total Annual Arrests", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("Arrests")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_trend2:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.lineplot(data=yearly_arrest_trends, x="year", y="arrest_rate", marker="o", color="#d62728", ax=ax)
        for _, row in yearly_arrest_trends.iterrows():
            ax.annotate(f"{row['arrest_rate']}%", (row["year"], row["arrest_rate"]), textcoords="offset points", xytext=(0, 6), ha="center", fontweight="bold")
        ax.set_title("Arrest Rate (%) Trajectory by Year", fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("Arrest Rate (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section 4: Community Analysis
# ---------------------------------------------------------------------------
elif section_selection == "Community Analysis":
    st.title("Community Area Analysis")

    st.subheader("Top 10 Community Areas with Highest Crime Counts")

    community_counts = filtered_df.groupby(["community_code", "community_name"])["id"].count().reset_index()
    community_counts.columns = ["Community Code", "Community Name", "Crime Count"]
    top10_communities = community_counts.sort_values(by="Crime Count", ascending=False).head(10)

    top10_communities["Display Label"] = top10_communities["Community Name"] + " (#" + top10_communities["Community Code"].astype(str) + ")"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(
        data=top10_communities,
        y="Display Label",
        x="Crime Count",
        hue="Display Label",
        legend=False,
        palette="viridis",
        ax=ax
    )
    for i, count_val in enumerate(top10_communities["Crime Count"]):
        ax.text(count_val + 0.5, i, f"{int(count_val)}", va="center", fontweight="bold")
    ax.set_title("Top 10 High-Incident Community Areas", fontsize=11, pad=10)
    ax.set_xlabel("Total Reported Crimes")
    ax.set_ylabel("Community Area")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Crime Intensity by Hour of Day")
    hourly_series = filtered_df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()

    peak_hour = hourly_series.idxmax()
    peak_count = hourly_series.max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly_series.index, hourly_series.values, marker="o", color="#e63946", linewidth=2.5)
    ax.axvline(peak_hour, color="#1d3557", linestyle="--", alpha=0.8, label=f"Peak Hour: {peak_hour}:00 ({peak_count} crimes)")
    ax.set_title("Crime Frequency Across 24-Hour Cycle", fontsize=11, pad=10)
    ax.set_xlabel("Hour of Day (0 to 23)")
    ax.set_ylabel("Reported Incidents")
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(f"Peak crime hour is {peak_hour}:00 with {peak_count} incidents recorded.")


# ---------------------------------------------------------------------------
# Section 5: Statistical Analysis
# ---------------------------------------------------------------------------
elif section_selection == "Statistical Analysis":
    st.title("Statistical Insights and Pattern Detection")

    st.subheader("Descriptive Statistics")
    numeric_features = ["year", "month", "hour", "arrest", "domestic"]
    stats_df = filtered_df[numeric_features].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Community Area Outlier Analysis (NumPy IQR Method)")

    comm_frequency = filtered_df.groupby("community_code")["id"].count().to_numpy()

    if len(comm_frequency) > 0:
        mean_crimes = float(np.mean(comm_frequency))
        std_crimes = float(np.std(comm_frequency))
        median_crimes = float(np.median(comm_frequency))

        # Outlier calculation via NumPy percentiles
        q25, q75 = np.percentile(comm_frequency, [25, 75])
        iqr_value = q75 - q25
        lower_threshold = q25 - (1.5 * iqr_value)
        upper_threshold = q75 + (1.5 * iqr_value)

        comm_summary_df = filtered_df.groupby(["community_code", "community_name"])["id"].count().reset_index()
        comm_summary_df.columns = ["Community Code", "Community Name", "Crime Count"]

        high_outliers = comm_summary_df[comm_summary_df["Crime Count"] > upper_threshold].sort_values(by="Crime Count", ascending=False)
        low_outliers = comm_summary_df[comm_summary_df["Crime Count"] < lower_threshold].sort_values(by="Crime Count", ascending=True)

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            display_kpi_card("Mean Crimes / Community", f"{mean_crimes:.2f}")
        with col_stat2:
            display_kpi_card("Median Count", f"{median_crimes:.1f}")
        with col_stat3:
            display_kpi_card("IQR Spread (Q3 - Q1)", f"{iqr_value:.2f}")
        with col_stat4:
            display_kpi_card("Upper Outlier Fence", f"{upper_threshold:.2f}")

        col_box, col_outliers = st.columns([1.2, 1.0])
        with col_box:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.boxplot(y=comm_frequency, color="#a8dadc", ax=ax)
            ax.axhline(upper_threshold, color="red", linestyle="--", label=f"Upper Threshold ({upper_threshold:.1f})")
            ax.axhline(mean_crimes, color="blue", linestyle=":", label=f"Mean ({mean_crimes:.1f})")
            ax.set_title("Box Plot: Crime Frequency per Community Area", fontsize=11)
            ax.set_ylabel("Crime Count")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_outliers:
            st.write(f"**Identified Outliers (> {upper_threshold:.1f} crimes):**")
            if not high_outliers.empty:
                st.dataframe(high_outliers, use_container_width=True, hide_index=True)
            else:
                st.info("No community areas exceed the statistical upper threshold.")

            if not low_outliers.empty:
                st.write("**Unusually Low Crime Areas:**")
                st.dataframe(low_outliers, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.subheader("Crime Cross-Correlation Matrix")
    st.caption("Assessing correlations between numerical attributes (Year, Month, Hour, Arrest, Domestic).")

    corr_df = filtered_df[["year", "month", "hour", "arrest", "domestic"]].corr()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Correlation Heatmap", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info("Correlation between calendar fields and arrest outcomes is small, indicating relatively steady reporting throughout the periods.")


# ---------------------------------------------------------------------------
# Section 6: Database Management (CRUD)
# ---------------------------------------------------------------------------
elif section_selection == "Database Management":
    st.title("Database Management (CRUD)")
    st.markdown("Database management interface connected to the Flask REST API and SQLite backend.")

    crud_tab1, crud_tab2, crud_tab3, crud_tab4 = st.tabs([
        "View Records",
        "Add Crime",
        "Edit Crime",
        "Delete Crime"
    ])

    with crud_tab1:
        st.subheader("Browse Stored Crime Records")
        search_query = st.text_input("Search by Case Number, Block, or Description:", "")

        params = {"page": 1, "limit": 25}
        if search_query:
            params["search"] = search_query

        try:
            api_resp = requests.get(f"{API_URL}/api/crimes", params=params, timeout=3.0)
            if api_resp.status_code == 200:
                result_json = api_resp.json()
                crime_records = result_json.get("data", [])
                total_db_records = result_json.get("total_records", 0)

                st.write(f"Showing **{len(crime_records)}** of **{total_db_records:,}** matching records:")
                if crime_records:
                    display_table = pd.DataFrame(crime_records)[[
                        "id", "case_number", "date", "block", "iucr_code",
                        "primary_type", "location_desc", "arrest", "district_code", "community_code"
                    ]]
                    st.dataframe(display_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No records matched the search criteria.")
            else:
                st.error("Failed to fetch records from Flask API.")
        except Exception as ex:
            st.error(f"Error connecting to Flask API: {ex}")

    with crud_tab2:
        st.subheader("Add Crime Record (POST /api/crimes)")
        with st.form("create_crime_form", clear_on_submit=True):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                f_case = st.text_input("Case Number *", placeholder="e.g. HZ300100")
                f_date = st.text_input("Date (M/D/YYYY HH:MM:SS) *", value=pd.Timestamp.now().strftime("%m/%d/%Y %H:%M:%S"))
                f_iucr = st.text_input("IUCR Code *", value="0486", placeholder="e.g. 0486 (Theft), 041A (Battery)")
                f_block = st.text_input("Block", placeholder="e.g. 010XX N MICHIGAN AVE")
                f_loc = st.text_input("Location Description", value="STREET")

            with col_f2:
                f_dist = st.number_input("District Code", min_value=1, max_value=31, value=1)
                f_comm = st.number_input("Community Area Code", min_value=1, max_value=77, value=8)
                f_ward = st.number_input("Ward Number", min_value=1, max_value=50, value=42)
                f_beat = st.number_input("Beat Number", min_value=100, max_value=2599, value=111)
                f_arrest = st.checkbox("Arrest Made")
                f_domestic = st.checkbox("Domestic Incident")

            submit_create = st.form_submit_button("Submit New Record", type="primary")

            if submit_create:
                if not f_case or not f_date or not f_iucr:
                    st.error("Case Number, Date, and IUCR Code are mandatory fields.")
                else:
                    create_payload = {
                        "case_number": f_case.strip(),
                        "date": f_date.strip(),
                        "iucr_code": f_iucr.strip(),
                        "block": f_block.strip(),
                        "location_desc": f_loc.strip(),
                        "district_code": int(f_dist),
                        "community_code": int(f_comm),
                        "ward_no": int(f_ward),
                        "beat_num": int(f_beat),
                        "arrest": 1 if f_arrest else 0,
                        "domestic": 1 if f_domestic else 0
                    }
                    try:
                        resp = requests.post(f"{API_URL}/api/crimes", json=create_payload, timeout=3.0)
                        if resp.status_code == 201:
                            st.success(f"Record created successfully. ID: {resp.json().get('data', {}).get('id')}")
                            st.cache_data.clear()
                        else:
                            st.error(f"Error ({resp.status_code}): {resp.json().get('error', 'Operation failed')}")
                    except Exception as err:
                        st.error(f"API Error: {err}")

    with crud_tab3:
        st.subheader("Edit Crime Record (PUT /api/crimes/{id})")
        target_update_id = st.number_input("Enter Crime Record ID to Edit:", min_value=1, step=1, value=10508693)

        current_data = None
        if target_update_id:
            try:
                fetch_res = requests.get(f"{API_URL}/api/crimes/{target_update_id}", timeout=2.0)
                if fetch_res.status_code == 200:
                    current_data = fetch_res.json().get("data", {})
                else:
                    st.warning(f"Record #{target_update_id} not found in database.")
            except Exception as ex:
                st.error(f"Cannot retrieve record: {ex}")

        if current_data:
            st.info(f"Editing Case: **{current_data.get('case_number')}** | Primary Type: **{current_data.get('primary_type')}**")
            with st.form(f"update_form_{target_update_id}"):
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    u_case = st.text_input("Case Number", value=current_data.get("case_number", ""))
                    u_date = st.text_input("Date", value=current_data.get("date", ""))
                    u_iucr = st.text_input("IUCR Code", value=current_data.get("iucr_code", ""))
                    u_block = st.text_input("Block", value=current_data.get("block", ""))
                    u_loc = st.text_input("Location Description", value=current_data.get("location_desc", ""))

                with col_u2:
                    u_dist = st.number_input("District Code", min_value=1, max_value=31, value=int(current_data.get("district_code") or 1))
                    u_comm = st.number_input("Community Area", min_value=1, max_value=77, value=int(current_data.get("community_code") or 1))
                    u_ward = st.number_input("Ward Number", min_value=1, max_value=50, value=int(current_data.get("ward_no") or 1))
                    u_beat = st.number_input("Beat Number", min_value=100, max_value=2599, value=int(current_data.get("beat_num") or 100))
                    u_arrest = st.checkbox("Arrest Made", value=bool(current_data.get("arrest")))
                    u_domestic = st.checkbox("Domestic Incident", value=bool(current_data.get("domestic")))

                save_update = st.form_submit_button("Save Changes", type="primary")

                if save_update:
                    update_payload = {
                        "case_number": u_case.strip(),
                        "date": u_date.strip(),
                        "iucr_code": u_iucr.strip(),
                        "block": u_block.strip(),
                        "location_desc": u_loc.strip(),
                        "district_code": int(u_dist),
                        "community_code": int(u_comm),
                        "ward_no": int(u_ward),
                        "beat_num": int(u_beat),
                        "arrest": 1 if u_arrest else 0,
                        "domestic": 1 if u_domestic else 0
                    }
                    try:
                        put_resp = requests.put(f"{API_URL}/api/crimes/{target_update_id}", json=update_payload, timeout=3.0)
                        if put_resp.status_code == 200:
                            st.success(f"Crime record #{target_update_id} updated successfully.")
                            st.cache_data.clear()
                        else:
                            st.error(f"Update failed ({put_resp.status_code}): {put_resp.json().get('error')}")
                    except Exception as err:
                        st.error(f"API Error: {err}")

    with crud_tab4:
        st.subheader("Delete Crime Record (DELETE /api/crimes/{id})")
        target_delete_id = st.number_input("Enter ID of Record to Delete:", min_value=1, step=1, value=10508693, key="del_id")

        record_to_del = None
        if target_delete_id:
            try:
                chk_resp = requests.get(f"{API_URL}/api/crimes/{target_delete_id}", timeout=2.0)
                if chk_resp.status_code == 200:
                    record_to_del = chk_resp.json().get("data")
                    st.warning(f"Target Record: Case **{record_to_del.get('case_number')}** | Date: **{record_to_del.get('date')}** | Offense: **{record_to_del.get('primary_type')}**")
                else:
                    st.info(f"No record found with ID #{target_delete_id}.")
            except Exception as ex:
                st.error(f"Connection error: {ex}")

        if record_to_del:
            confirm_delete = st.checkbox("Confirm permanent deletion of this record.")
            if st.button("Delete Crime Record", type="primary", disabled=not confirm_delete):
                try:
                    del_resp = requests.delete(f"{API_URL}/api/crimes/{target_delete_id}", timeout=3.0)
                    if del_resp.status_code == 200:
                        st.success(f"Crime record #{target_delete_id} deleted successfully.")
                        st.cache_data.clear()
                    else:
                        st.error(f"Deletion failed: {del_resp.json().get('error')}")
                except Exception as err:
                    st.error(f"API Error: {err}")


# ---------------------------------------------------------------------------
# Section 7: Reports and Data Quality
# ---------------------------------------------------------------------------
elif section_selection == "Reports and Data Quality":
    st.title("Reports and Data Quality")

    rep_tab1, rep_tab2, rep_tab3 = st.tabs([
        "Data Quality Report",
        "SQL Views and Queries",
        "Data Exports"
    ])

    with rep_tab1:
        st.subheader("Data Quality and Ingestion Summary")
        quality_data = fetch_quality_report_via_api()

        if quality_data:
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                display_kpi_card("Rows Ingested", f"{quality_data.get('total_rows'):,}")
            with col_q2:
                display_kpi_card("Columns Processed", f"{quality_data.get('total_columns')}")
            with col_q3:
                display_kpi_card("Duplicate Rows", f"{quality_data.get('duplicate_rows_detected')}")

            st.write("### Missing Values Analysis per Column (NumPy)")
            missing_dict = quality_data.get("missing_percentages", {})
            missing_df = pd.DataFrame({
                "Column": list(missing_dict.keys()),
                "Missing Percentage (%)": list(missing_dict.values())
            }).sort_values(by="Missing Percentage (%)", ascending=False)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

            st.info(f"Missing Value Threshold Check: {quality_data.get('high_missing_action')}")

            st.write("### First 10 Ingested Records")
            first_10 = quality_data.get("first_10_records", [])
            if first_10:
                st.dataframe(pd.DataFrame(first_10), use_container_width=True, hide_index=True)
        else:
            st.info("Quality report unavailable from Flask API. Computing live from SQLite...")
            st.write(f"Total Rows: **{len(full_crime_df)}** | Total Columns: **{full_crime_df.shape[1]}**")
            missing_pct = np.round((full_crime_df.isnull().sum() / len(full_crime_df)) * 100, 2)
            st.dataframe(pd.DataFrame({"Column": missing_pct.index, "Missing %": missing_pct.values}), use_container_width=True)

    with rep_tab2:
        st.subheader("SQLite Analytical Views and Queries")
        st.caption("Direct queries executed on SQLite views and tables via pd.read_sql_query().")

        conn = sqlite3.connect(DATABASE_FILE)

        st.write("#### 1. SQL View: `vw_crime_yearly`")
        st.code("""
        SELECT Year, "Total crime count" 
        FROM vw_crime_yearly;
        """, language="sql")
        yearly_view_df = pd.read_sql_query('SELECT "Year" as Year, "Total crime count" as "Crime Count" FROM vw_crime_yearly;', conn)
        st.dataframe(yearly_view_df, use_container_width=True, hide_index=True)

        st.write("#### 2. SQL View: `vw_crime_by_category` (Top 5 Types)")
        st.code("""
        SELECT "Primary Type", "Crime count", "Crime percentage"
        FROM vw_crime_by_category
        LIMIT 5;
        """, language="sql")
        cat_view_df = pd.read_sql_query('SELECT "Primary Type", "Crime count", "Crime percentage" FROM vw_crime_by_category LIMIT 5;', conn)
        st.dataframe(cat_view_df, use_container_width=True, hide_index=True)

        st.write("#### 3. SQL Query: Annual Arrest Enforcement Rates")
        st.code("""
        SELECT 
            year,
            COUNT(*) AS total_crimes,
            COALESCE(SUM(arrest), 0) AS total_arrests,
            ROUND(COALESCE(SUM(arrest), 0) * 100.0 / COUNT(*), 2) AS arrest_rate
        FROM crimes
        GROUP BY year
        ORDER BY year ASC;
        """, language="sql")
        arrest_sql_df = pd.read_sql_query("""
        SELECT 
            year AS "Year",
            COUNT(*) AS "Total Crimes",
            COALESCE(SUM(arrest), 0) AS "Total Arrests",
            ROUND(COALESCE(SUM(arrest), 0) * 100.0 / COUNT(*), 2) AS "Arrest Rate (%)"
        FROM crimes
        GROUP BY year
        ORDER BY year ASC;
        """, conn)
        st.dataframe(arrest_sql_df, use_container_width=True, hide_index=True)

        conn.close()

    with rep_tab3:
        st.subheader("Download Analytical Reports as CSV")

        cleaned_csv_data = full_crime_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned Crime Dataset (CSV)",
            data=cleaned_csv_data,
            file_name="chicago_cleaned_crime_data.csv",
            mime="text/csv",
            type="primary"
        )

        st.markdown("---")

        yearly_summary_df = full_crime_df.groupby("year").agg(
            total_crimes=("id", "count"),
            total_arrests=("arrest", "sum")
        ).reset_index()
        yearly_summary_df["arrest_rate"] = np.round((yearly_summary_df["total_arrests"] * 100.0 / yearly_summary_df["total_crimes"]), 2)
        yearly_csv = yearly_summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Yearly Crime Summary (CSV)",
            data=yearly_csv,
            file_name="chicago_crime_yearly_summary.csv",
            mime="text/csv"
        )

        st.markdown("---")

        category_summary_df = full_crime_df.groupby("primary_type").agg(
            incident_count=("id", "count"),
            arrest_count=("arrest", "sum")
        ).reset_index()
        category_summary_df["share_percentage"] = np.round((category_summary_df["incident_count"] * 100.0 / len(full_crime_df)), 2)
        category_summary_df["arrest_rate"] = np.round((category_summary_df["arrest_count"] * 100.0 / category_summary_df["incident_count"]), 2)
        category_summary_df = category_summary_df.sort_values(by="incident_count", ascending=False)
        cat_csv = category_summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Crime Category Summary (CSV)",
            data=cat_csv,
            file_name="chicago_crime_category_summary.csv",
            mime="text/csv"
        )
