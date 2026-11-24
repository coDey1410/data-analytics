# Chicago Crime Data Analytics

A data analytics web application for Chicago crime records, built with Python, SQLite3, Flask, and Streamlit.

---

## 1. Overview

This project provides an end-to-end data pipeline, analytical backend, and interactive dashboard for analyzing Chicago reported crime data.

- **Data Ingestion**: Cleans raw CSV records, standardizes text casing, extracts date features (Year, Month, Day of Week, Hour), and loads data into SQLite3 without creating duplicate entries.
- **SQLite Database & Views**: Stores records in SQLite with reference tables (police districts, wards, IUCR codes, community areas, beats) and custom views (`vw_crime_yearly`, `vw_crime_by_category`).
- **Flask REST API**: Exposes endpoints for CRUD operations (Create, Read with pagination/filters, Update, Delete) and summary statistics using parameterized SQL queries.
- **Streamlit Dashboard**: Visualizes yearly trends, category distributions, arrest outcomes, monthly heatmaps, hourly intensity, and community area outlier detection.
- **Exports**: Provides CSV downloads for cleaned data and summary tables.

---

## 2. Tech Stack

- **Backend**: Flask
- **Frontend**: Streamlit
- **Database**: SQLite3 (native `sqlite3` module)
- **Data Analysis**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **HTTP Client**: Requests

---

## 3. Project Structure

```
chicago-crime-analytics/
│
├── app.py                 # Flask backend, database setup, and CRUD API
├── streamlit_app.py       # Streamlit dashboard and visualizations
├── requirements.txt       # Dependencies
├── README.md              # Documentation
│
├── data/                  # Source CSV files
│   ├── chicago_crime_dataset.csv
│   ├── chicago_district_ps_info.csv
│   ├── chicago_ward_offices.csv
│   ├── iucr_codes.csv
│   ├── chicago_city_community.csv
│   └── chicago_police_beat_info.csv
│
└── chicago_crime.db       # SQLite database (auto-generated)
```

---

## 4. Setup and Installation

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Running the Project

Start the Flask backend in one terminal:
```bash
python app.py
```
*On initial startup, `app.py` automatically checks for `chicago_crime.db`, sets up tables, cleans the CSVs, and inserts records.*

In a second terminal, run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
*Open http://localhost:8501 in your browser.*

---

## 6. Dashboard Sections

1. **Dashboard Overview**: Key metrics (total crimes, total arrests, arrest rate, categories count), yearly trend line, and top 5 categories.
2. **Crime Analysis**: Annual volume plot, top 10 crime categories breakdown, and a monthly by weekday heatmap.
3. **Arrest Analysis**: Arrest percentage, arrest rates compared across crime types, and annual arrest counts.
4. **Community Analysis**: Top 10 high-crime community areas with neighborhood names, and a 24-hour crime intensity curve.
5. **Statistical Analysis**: Descriptive statistics table, IQR outlier detection for community areas with box plot, and numerical cross-correlation matrix.
6. **Database Management**: CRUD interface to browse records with pagination and search, insert new crimes, edit records, and delete records with confirmation.
7. **Reports and Data Quality**: Data quality report (missing percentage per column, duplicate check), SQL view query outputs, and CSV export buttons.
