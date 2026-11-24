"""
Backend Flask API and SQLite3 integration for Chicago Crime Data Analytics.
Handles CSV ingestion, data cleaning, schema setup, SQL views, and REST endpoints for CRUD operations.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_NAME = os.path.join(BASE_DIR, "chicago_crime.db")

app = Flask(__name__)

# Cache the data quality metrics so we don't have to recompute on every GET request
INGESTION_QUALITY_REPORT = {}


def get_db_connection():
    # Return sqlite connection with Row factory for column access by name
    connection = sqlite3.connect(DATABASE_NAME, timeout=15.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def repair_and_read_csv(file_path):
    # Some lines in the raw export have unclosed quotes at the end of lines
    # Read as text first, close any uneven quotes per line, then pass into pandas
    with open(file_path, "r", encoding="utf-8", errors="replace") as file_handle:
        raw_lines = file_handle.read().splitlines()

    sanitized_lines = []
    for line in raw_lines:
        if line.count('"') % 2 != 0:
            line = line + '"'
        sanitized_lines.append(line)

    from io import StringIO
    cleaned_buffer = StringIO("\n".join(sanitized_lines))
    return pd.read_csv(cleaned_buffer)


def build_database_schema(db_conn):
    # Create the primary table, reference tables, and analytical views
    cursor = db_conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crimes (
        id INTEGER PRIMARY KEY,
        case_number TEXT NOT NULL,
        date TEXT NOT NULL,
        block TEXT,
        iucr_code TEXT NOT NULL,
        primary_type TEXT NOT NULL,
        crime_description TEXT,
        location_desc TEXT,
        arrest INTEGER DEFAULT 0,
        domestic INTEGER DEFAULT 0,
        beat_num INTEGER,
        district_code INTEGER,
        ward_no INTEGER,
        community_code INTEGER,
        fbi_code TEXT,
        x_coordinate REAL,
        y_coordinate REAL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        day_of_week TEXT NOT NULL,
        hour INTEGER NOT NULL,
        date_of_update TEXT,
        latitude REAL,
        longitude REAL,
        location TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ref_iucr_codes (
        iucr TEXT PRIMARY KEY,
        primary_type TEXT NOT NULL,
        description TEXT,
        index_code TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ref_districts (
        district_code INTEGER PRIMARY KEY,
        district_name TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        zip TEXT,
        phone TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ref_wards (
        ward INTEGER PRIMARY KEY,
        alderman TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        zipcode TEXT,
        ward_phone TEXT,
        email TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ref_communities (
        community_code INTEGER PRIMARY KEY,
        community_name TEXT NOT NULL,
        population TEXT,
        area_sqmile REAL,
        area_sqkm REAL,
        density_per_sqmi TEXT,
        density_per_sqkm TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ref_beats (
        district INTEGER,
        sector INTEGER,
        beat INTEGER,
        beat_num INTEGER PRIMARY KEY
    );
    """)

    # SQL View 1: Annual crime count
    cursor.execute("DROP VIEW IF EXISTS vw_crime_yearly;")
    cursor.execute("""
    CREATE VIEW vw_crime_yearly AS
    SELECT 
        year AS "Year", 
        COUNT(*) AS "Total crime count"
    FROM crimes
    GROUP BY year
    ORDER BY year ASC;
    """)

    # SQL View 2: Crime category breakdown with percentage share
    cursor.execute("DROP VIEW IF EXISTS vw_crime_by_category;")
    cursor.execute("""
    CREATE VIEW vw_crime_by_category AS
    SELECT 
        primary_type AS "Primary Type",
        COUNT(*) AS "Crime count",
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM crimes), 2) AS "Crime percentage"
    FROM crimes
    GROUP BY primary_type
    ORDER BY "Crime count" DESC;
    """)

    db_conn.commit()


def run_data_ingestion_pipeline():
    # Load all CSV files, clean records, derive date fields, and load into SQLite
    global INGESTION_QUALITY_REPORT
    print("Reading and cleaning Chicago crime dataset...")

    crime_csv_path = os.path.join(DATA_DIR, "chicago_crime_dataset.csv")
    iucr_csv_path = os.path.join(DATA_DIR, "iucr_codes.csv")
    dist_csv_path = os.path.join(DATA_DIR, "chicago_district_ps_info.csv")
    ward_csv_path = os.path.join(DATA_DIR, "chicago_ward_offices.csv")
    comm_csv_path = os.path.join(DATA_DIR, "chicago_city_community.csv")
    beat_csv_path = os.path.join(DATA_DIR, "chicago_police_beat_info.csv")

    if not os.path.exists(crime_csv_path):
        print(f"Data file not found at {crime_csv_path}")
        return

    raw_crime_df = repair_and_read_csv(crime_csv_path)
    total_raw_rows, total_raw_cols = raw_crime_df.shape
    raw_crime_df.columns = [c.strip().lower() for c in raw_crime_df.columns]

    # Load IUCR code reference table
    raw_iucr_df = repair_and_read_csv(iucr_csv_path)
    raw_iucr_df.columns = [c.strip() for c in raw_iucr_df.columns]
    # Standardize IUCR codes to 4 characters with leading zeros
    raw_iucr_df["iucr_clean"] = raw_iucr_df["IUCR"].astype(str).str.strip().str.zfill(4)
    iucr_lookup_map = raw_iucr_df.set_index("iucr_clean")[["PRIMARY_TYPE", "DESCRIPTION"]].to_dict("index")

    # Data quality calculations using NumPy
    total_rows = len(raw_crime_df)
    missing_counts = raw_crime_df.isnull().sum()
    missing_percentages = np.round((missing_counts.to_numpy() / total_rows) * 100, 2)
    missing_report = dict(zip(raw_crime_df.columns, missing_percentages))

    # Check for columns missing over 50% data
    high_missing_cols = [c for c, pct in missing_report.items() if pct > 50.0]
    duplicate_records_count = int(raw_crime_df.duplicated(subset=["id"]).sum())

    cleaned_df = raw_crime_df.copy()

    # Clean text values
    cleaned_df["block"] = cleaned_df["block"].fillna("UNKNOWN BLOCK").astype(str).str.strip().str.upper()
    cleaned_df["location_desc"] = cleaned_df["location_desc"].fillna("OTHER").astype(str).str.strip().str.upper()

    # Map boolean columns to 1 and 0 integers
    cleaned_df["arrest"] = cleaned_df["arrest"].astype(str).str.upper().map(
        {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
    ).fillna(0).astype(int)
    cleaned_df["domestic"] = cleaned_df["domestic"].astype(str).str.upper().map(
        {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
    ).fillna(0).astype(int)

    # Parse dates and extract calendar features
    parsed_datetimes = pd.to_datetime(cleaned_df["date"], errors="coerce")
    cleaned_df["year"] = parsed_datetimes.dt.year.fillna(cleaned_df["year"]).astype(int)
    cleaned_df["month"] = parsed_datetimes.dt.month.fillna(1).astype(int)
    cleaned_df["day_of_week"] = parsed_datetimes.dt.day_name().fillna("Unknown")
    cleaned_df["hour"] = parsed_datetimes.dt.hour.fillna(0).astype(int)

    # Match IUCR code to get primary crime type and description
    cleaned_df["iucr_code_clean"] = cleaned_df["iucr_code"].astype(str).str.strip().str.zfill(4)
    primary_types = []
    descriptions = []

    for code in cleaned_df["iucr_code_clean"]:
        match_record = iucr_lookup_map.get(code)
        if match_record:
            primary_types.append(str(match_record.get("PRIMARY_TYPE", "OTHER OFFENSE")).strip().upper())
            descriptions.append(str(match_record.get("DESCRIPTION", "UNKNOWN")).strip())
        else:
            primary_types.append("OTHER OFFENSE")
            descriptions.append("UNKNOWN")

    cleaned_df["primary_type"] = primary_types
    cleaned_df["crime_description"] = descriptions

    INGESTION_QUALITY_REPORT = {
        "total_rows": int(total_raw_rows),
        "total_columns": int(total_raw_cols),
        "column_names": list(raw_crime_df.columns),
        "column_dtypes": {c: str(raw_crime_df[c].dtype) for c in raw_crime_df.columns},
        "missing_counts": missing_counts.to_dict(),
        "missing_percentages": missing_report,
        "high_missing_columns": high_missing_cols,
        "high_missing_action": (
            "No columns exceeded the 50% missing threshold. All original columns preserved."
            if not high_missing_cols else f"Columns with >50% missing values: {high_missing_cols}"
        ),
        "duplicate_rows_detected": duplicate_records_count,
        "first_10_records": cleaned_df.head(10).to_dict(orient="records")
    }

    conn = get_db_connection()
    build_database_schema(conn)
    cursor = conn.cursor()

    # Load IUCR reference table
    for _, row in raw_iucr_df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO ref_iucr_codes (iucr, primary_type, description, index_code)
        VALUES (?, ?, ?, ?)
        """, (
            str(row["IUCR"]).strip().zfill(4),
            str(row.get("PRIMARY_TYPE", "")).strip().upper(),
            str(row.get("DESCRIPTION", "")).strip(),
            str(row.get("INDEX CODE", "")).strip()
        ))

    # Load district details
    if os.path.exists(dist_csv_path):
        dist_df = repair_and_read_csv(dist_csv_path)
        dist_df.columns = [c.strip().upper() for c in dist_df.columns]
        for _, row in dist_df.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO ref_districts (district_code, district_name, address, city, state, zip, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row["DISTRICT_CODE"]),
                str(row.get("DISTRICT_NAME", "")).strip(),
                str(row.get("ADDRESS", "")).strip(),
                str(row.get("CITY", "")).strip(),
                str(row.get("STATE", "")).strip(),
                str(row.get("ZIP", "")).strip(),
                str(row.get("PHONE", "")).strip()
            ))

    # Load ward offices
    if os.path.exists(ward_csv_path):
        ward_df = repair_and_read_csv(ward_csv_path)
        ward_df.columns = [c.strip().lower() for c in ward_df.columns]
        for _, row in ward_df.iterrows():
            try:
                w_num = int(row["ward"])
            except (ValueError, TypeError):
                continue
            cursor.execute("""
            INSERT OR REPLACE INTO ref_wards (ward, alderman, address, city, state, zipcode, ward_phone, email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                w_num,
                str(row.get("alderman", "")).strip(),
                str(row.get("address", "")).strip(),
                str(row.get("city", "")).strip(),
                str(row.get("state", "")).strip(),
                str(row.get("zipcode", "")).strip(),
                str(row.get("ward_phone", "")).strip(),
                str(row.get("email", "")).strip()
            ))

    # Load community names and areas
    if os.path.exists(comm_csv_path):
        comm_df = repair_and_read_csv(comm_csv_path)
        comm_df.columns = [c.strip().lower() for c in comm_df.columns]
        for _, row in comm_df.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO ref_communities (
                community_code, community_name, population, area_sqmile, area_sqkm, density_per_sqmi, density_per_sqkm
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row["community_code"]),
                str(row["community_name"]).strip(),
                str(row.get("population", "")).strip(),
                float(row.get("area_sqmile", 0) or 0),
                float(row.get("area_sqkm", 0) or 0),
                str(row.get("density_per_sqmi", "")).strip(),
                str(row.get("density_per_sqkm", "")).strip()
            ))

    # Load police beats
    if os.path.exists(beat_csv_path):
        beat_df = repair_and_read_csv(beat_csv_path)
        beat_df.columns = [c.strip().upper() for c in beat_df.columns]
        for _, row in beat_df.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO ref_beats (district, sector, beat, beat_num)
            VALUES (?, ?, ?, ?)
            """, (
                int(row["DISTRICT"]),
                int(row["SECTOR"]),
                int(row["BEAT"]),
                int(row["BEAT_NUM"])
            ))

    # Insert crime records into SQLite; INSERT OR IGNORE ensures no duplicate IDs on repeated runs
    insert_records_payload = []
    for _, row in cleaned_df.iterrows():
        insert_records_payload.append((
            int(row["id"]),
            str(row["case_number"]).strip(),
            str(row["date"]).strip(),
            str(row["block"]),
            str(row["iucr_code"]).strip(),
            str(row["primary_type"]),
            str(row["crime_description"]),
            str(row["location_desc"]),
            int(row["arrest"]),
            int(row["domestic"]),
            int(row["beat_num"]) if pd.notnull(row["beat_num"]) else None,
            int(row["district_code"]) if pd.notnull(row["district_code"]) else None,
            int(row["ward_no"]) if pd.notnull(row["ward_no"]) else None,
            int(row["community_code"]) if pd.notnull(row["community_code"]) else None,
            str(row.get("fbi_code", "")).strip(),
            float(row["x_coordinate"]) if pd.notnull(row.get("x_coordinate")) else None,
            float(row["y_coordinate"]) if pd.notnull(row.get("y_coordinate")) else None,
            int(row["year"]),
            int(row["month"]),
            str(row["day_of_week"]),
            int(row["hour"]),
            str(row.get("date_of_update", "")).strip(),
            float(row["latitude"]) if pd.notnull(row.get("latitude")) else None,
            float(row["longitude"]) if pd.notnull(row.get("longitude")) else None,
            str(row.get("location", "")).strip()
        ))

    cursor.executemany("""
    INSERT OR IGNORE INTO crimes (
        id, case_number, date, block, iucr_code, primary_type, crime_description,
        location_desc, arrest, domestic, beat_num, district_code, ward_no,
        community_code, fbi_code, x_coordinate, y_coordinate, year, month,
        day_of_week, hour, date_of_update, latitude, longitude, location
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    );
    """, insert_records_payload)

    conn.commit()
    conn.close()
    print(f"Ingested {len(insert_records_payload)} crime records into SQLite database.")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index_status():
    return jsonify({
        "project": "Chicago Crime Data Analytics Capstone",
        "backend": "Flask",
        "database": "SQLite3",
        "status": "online",
        "endpoints": [
            "GET /api/summary",
            "GET /api/data-quality",
            "GET /api/crimes",
            "GET /api/crimes/<id>",
            "POST /api/crimes",
            "PUT /api/crimes/<id>",
            "DELETE /api/crimes/<id>",
            "GET /api/views/yearly",
            "GET /api/views/category",
            "GET /api/views/arrests-yearly"
        ]
    }), 200


@app.route("/api/data-quality", methods=["GET"])
def get_data_quality_report():
    if not INGESTION_QUALITY_REPORT:
        run_data_ingestion_pipeline()
    return jsonify({"success": True, "report": INGESTION_QUALITY_REPORT}), 200


@app.route("/api/summary", methods=["GET"])
def get_database_summary():
    # Return KPI stats and summary tables from database
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS total_records, COALESCE(SUM(arrest), 0) AS total_arrests FROM crimes;")
    kpi_row = cursor.fetchone()
    total_crimes = kpi_row["total_records"] if kpi_row else 0
    total_arrests = kpi_row["total_arrests"] if kpi_row else 0
    overall_arrest_rate = round((total_arrests * 100.0 / total_crimes), 2) if total_crimes > 0 else 0.0

    cursor.execute("SELECT COUNT(DISTINCT primary_type) AS category_count FROM crimes;")
    distinct_types = cursor.fetchone()["category_count"]

    cursor.execute('SELECT "Year" as year, "Total crime count" as count FROM vw_crime_yearly ORDER BY "Year" ASC;')
    yearly_trend = [dict(row) for row in cursor.fetchall()]

    cursor.execute("""
    SELECT 
        "Primary Type" as primary_type, 
        "Crime count" as crime_count, 
        "Crime percentage" as crime_percentage 
    FROM vw_crime_by_category 
    LIMIT 10;
    """)
    top_categories = [dict(row) for row in cursor.fetchall()]

    cursor.execute("""
    SELECT 
        primary_type,
        COUNT(*) AS total_incidents,
        SUM(arrest) AS arrest_count,
        ROUND(SUM(arrest) * 100.0 / COUNT(*), 2) AS arrest_rate
    FROM crimes
    GROUP BY primary_type
    ORDER BY total_incidents DESC;
    """)
    arrests_by_type = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return jsonify({
        "success": True,
        "summary": {
            "total_records": total_crimes,
            "total_arrests": total_arrests,
            "arrest_percentage": overall_arrest_rate,
            "distinct_categories": distinct_types,
            "yearly_trend": yearly_trend,
            "top_categories": top_categories,
            "arrests_by_category": arrests_by_type
        }
    }), 200


@app.route("/api/crimes", methods=["GET"])
def list_crimes():
    # Return paginated list of crime records with optional filter parameters
    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
        offset = (page - 1) * limit

        filter_clauses = []
        filter_params = []

        year = request.args.get("year")
        if year and year != "All":
            filter_clauses.append("year = ?")
            filter_params.append(int(year))

        primary_type = request.args.get("primary_type")
        if primary_type and primary_type != "All":
            filter_clauses.append("primary_type = ?")
            filter_params.append(primary_type.strip().upper())

        district = request.args.get("district")
        if district and district != "All":
            filter_clauses.append("district_code = ?")
            filter_params.append(int(district))

        community = request.args.get("community")
        if community and community != "All":
            filter_clauses.append("community_code = ?")
            filter_params.append(int(community))

        arrest = request.args.get("arrest")
        if arrest in ["0", "1"]:
            filter_clauses.append("arrest = ?")
            filter_params.append(int(arrest))

        search_text = request.args.get("search")
        if search_text:
            filter_clauses.append("(case_number LIKE ? OR block LIKE ? OR location_desc LIKE ?)")
            pattern = f"%{search_text.strip()}%"
            filter_params.extend([pattern, pattern, pattern])

        where_sql = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

        conn = get_db_connection()
        cursor = conn.cursor()

        count_sql = f"SELECT COUNT(*) as total FROM crimes {where_sql};"
        cursor.execute(count_sql, filter_params)
        total_matching = cursor.fetchone()["total"]

        query_sql = f"""
        SELECT 
            id, case_number, date, block, iucr_code, primary_type,
            crime_description, location_desc, arrest, domestic,
            beat_num, district_code, ward_no, community_code,
            year, month, day_of_week, hour, latitude, longitude
        FROM crimes
        {where_sql}
        ORDER BY id DESC
        LIMIT ? OFFSET ?;
        """
        cursor.execute(query_sql, filter_params + [limit, offset])
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()

        total_pages = (total_matching + limit - 1) // limit if limit else 1

        return jsonify({
            "success": True,
            "page": page,
            "limit": limit,
            "total_records": total_matching,
            "total_pages": total_pages,
            "data": records
        }), 200

    except Exception:
        return jsonify({"success": False, "error": "Unable to retrieve crime records."}), 500


@app.route("/api/crimes/<int:crime_id>", methods=["GET"])
def get_single_crime(crime_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM crimes WHERE id = ?;", (crime_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False, "error": f"Crime record with ID {crime_id} not found."}), 404

    return jsonify({"success": True, "data": dict(row)}), 200


@app.route("/api/crimes", methods=["POST"])
def create_crime_record():
    # Insert new crime record with field validation and date parsing
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"success": False, "error": "Request body must be a valid JSON object."}), 400

    case_number = str(payload.get("case_number", "")).strip().upper()
    date_str = str(payload.get("date", "")).strip()
    iucr_code = str(payload.get("iucr_code", "")).strip().zfill(4)

    if not case_number or not date_str or not iucr_code:
        return jsonify({
            "success": False,
            "error": "Missing required fields: case_number, date, and iucr_code are mandatory."
        }), 400

    try:
        parsed_dt = pd.to_datetime(date_str)
        year_val = int(parsed_dt.year)
        month_val = int(parsed_dt.month)
        day_of_week_val = str(parsed_dt.day_name())
        hour_val = int(parsed_dt.hour)
    except Exception:
        return jsonify({"success": False, "error": "Invalid date format. Expected standard date or timestamp."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT primary_type, description FROM ref_iucr_codes WHERE iucr = ?;", (iucr_code,))
    iucr_match = cursor.fetchone()
    if iucr_match:
        primary_type = iucr_match["primary_type"]
        crime_desc = iucr_match["description"]
    else:
        primary_type = str(payload.get("primary_type", "OTHER OFFENSE")).strip().upper()
        crime_desc = str(payload.get("crime_description", "INCIDENT REPORT")).strip()

    record_id = payload.get("id")
    if record_id is not None:
        try:
            record_id = int(record_id)
        except ValueError:
            conn.close()
            return jsonify({"success": False, "error": "Record ID must be an integer."}), 400
    else:
        cursor.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM crimes;")
        record_id = cursor.fetchone()["next_id"]

    try:
        cursor.execute("""
        INSERT INTO crimes (
            id, case_number, date, block, iucr_code, primary_type, crime_description,
            location_desc, arrest, domestic, beat_num, district_code, ward_no,
            community_code, fbi_code, x_coordinate, y_coordinate, year, month,
            day_of_week, hour, date_of_update, latitude, longitude, location
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """, (
            record_id,
            case_number,
            date_str,
            str(payload.get("block", "UNKNOWN")).strip().upper(),
            iucr_code,
            primary_type,
            crime_desc,
            str(payload.get("location_desc", "STREET")).strip().upper(),
            1 if payload.get("arrest") in [1, True, "1", "TRUE", "true"] else 0,
            1 if payload.get("domestic") in [1, True, "1", "TRUE", "true"] else 0,
            int(payload["beat_num"]) if payload.get("beat_num") is not None else None,
            int(payload["district_code"]) if payload.get("district_code") is not None else None,
            int(payload["ward_no"]) if payload.get("ward_no") is not None else None,
            int(payload["community_code"]) if payload.get("community_code") is not None else None,
            str(payload.get("fbi_code", "")).strip(),
            float(payload["x_coordinate"]) if payload.get("x_coordinate") is not None else None,
            float(payload["y_coordinate"]) if payload.get("y_coordinate") is not None else None,
            year_val,
            month_val,
            day_of_week_val,
            hour_val,
            datetime.now().strftime("%m/%d/%Y"),
            float(payload["latitude"]) if payload.get("latitude") is not None else None,
            float(payload["longitude"]) if payload.get("longitude") is not None else None,
            str(payload.get("location", ""))
        ))
        conn.commit()

        cursor.execute("SELECT * FROM crimes WHERE id = ?;", (record_id,))
        created_row = dict(cursor.fetchone())
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Crime record {record_id} successfully created.",
            "data": created_row
        }), 201

    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"success": False, "error": f"A crime record with ID {record_id} or Case Number already exists."}), 400
    except Exception:
        conn.close()
        return jsonify({"success": False, "error": "Internal error occurred while creating crime record."}), 500


@app.route("/api/crimes/<int:crime_id>", methods=["PUT"])
def update_crime_record(crime_id):
    # Update an existing crime record by ID
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"success": False, "error": "Request body must be a valid JSON object."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM crimes WHERE id = ?;", (crime_id,))
    current_record = cursor.fetchone()

    if not current_record:
        conn.close()
        return jsonify({"success": False, "error": f"Crime record {crime_id} does not exist."}), 404

    # Recalculate date fields if date was modified
    date_val = str(payload.get("date", current_record["date"])).strip()
    try:
        parsed_dt = pd.to_datetime(date_val)
        year_val = int(parsed_dt.year)
        month_val = int(parsed_dt.month)
        day_of_week_val = str(parsed_dt.day_name())
        hour_val = int(parsed_dt.hour)
    except Exception:
        year_val = current_record["year"]
        month_val = current_record["month"]
        day_of_week_val = current_record["day_of_week"]
        hour_val = current_record["hour"]

    # Re-lookup IUCR classification if code was modified
    iucr_code = str(payload.get("iucr_code", current_record["iucr_code"])).strip().zfill(4)
    cursor.execute("SELECT primary_type, description FROM ref_iucr_codes WHERE iucr = ?;", (iucr_code,))
    iucr_match = cursor.fetchone()
    if iucr_match:
        primary_type = iucr_match["primary_type"]
        crime_desc = iucr_match["description"]
    else:
        primary_type = str(payload.get("primary_type", current_record["primary_type"])).strip().upper()
        crime_desc = str(payload.get("crime_description", current_record["crime_description"])).strip()

    try:
        cursor.execute("""
        UPDATE crimes SET
            case_number = ?,
            date = ?,
            block = ?,
            iucr_code = ?,
            primary_type = ?,
            crime_description = ?,
            location_desc = ?,
            arrest = ?,
            domestic = ?,
            beat_num = ?,
            district_code = ?,
            ward_no = ?,
            community_code = ?,
            year = ?,
            month = ?,
            day_of_week = ?,
            hour = ?,
            date_of_update = ?
        WHERE id = ?;
        """, (
            str(payload.get("case_number", current_record["case_number"])).strip().upper(),
            date_val,
            str(payload.get("block", current_record["block"])).strip().upper(),
            iucr_code,
            primary_type,
            crime_desc,
            str(payload.get("location_desc", current_record["location_desc"])).strip().upper(),
            1 if payload.get("arrest", current_record["arrest"]) in [1, True, "1", "TRUE", "true"] else 0,
            1 if payload.get("domestic", current_record["domestic"]) in [1, True, "1", "TRUE", "true"] else 0,
            int(payload.get("beat_num", current_record["beat_num"]) or current_record["beat_num"]),
            int(payload.get("district_code", current_record["district_code"]) or current_record["district_code"]),
            int(payload.get("ward_no", current_record["ward_no"]) or current_record["ward_no"]),
            int(payload.get("community_code", current_record["community_code"]) or current_record["community_code"]),
            year_val,
            month_val,
            day_of_week_val,
            hour_val,
            datetime.now().strftime("%m/%d/%Y"),
            crime_id
        ))
        conn.commit()

        cursor.execute("SELECT * FROM crimes WHERE id = ?;", (crime_id,))
        updated_row = dict(cursor.fetchone())
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Crime record {crime_id} successfully updated.",
            "data": updated_row
        }), 200

    except Exception:
        conn.close()
        return jsonify({"success": False, "error": f"Failed to update crime record {crime_id}."}), 500


@app.route("/api/crimes/<int:crime_id>", methods=["DELETE"])
def delete_crime_record(crime_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM crimes WHERE id = ?;", (crime_id,))
    target = cursor.fetchone()

    if not target:
        conn.close()
        return jsonify({"success": False, "error": f"Crime record {crime_id} not found."}), 404

    try:
        cursor.execute("DELETE FROM crimes WHERE id = ?;", (crime_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": f"Crime record {crime_id} successfully deleted."}), 200
    except Exception:
        conn.close()
        return jsonify({"success": False, "error": f"Unable to delete crime record {crime_id}."}), 500


# ---------------------------------------------------------------------------
# SQL View Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/views/yearly", methods=["GET"])
def get_yearly_view():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT "Year" AS year, "Total crime count" AS total_crimes FROM vw_crime_yearly;')
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"success": True, "data": rows}), 200


@app.route("/api/views/category", methods=["GET"])
def get_category_view():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT "Primary Type" AS primary_type, "Crime count" AS count, "Crime percentage" AS percentage FROM vw_crime_by_category;')
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"success": True, "data": rows}), 200


@app.route("/api/views/arrests-yearly", methods=["GET"])
def get_arrests_yearly_sql():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        year,
        COUNT(*) AS total_crimes,
        COALESCE(SUM(arrest), 0) AS arrest_count,
        ROUND(COALESCE(SUM(arrest), 0) * 100.0 / COUNT(*), 2) AS arrest_rate
    FROM crimes
    GROUP BY year
    ORDER BY year ASC;
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({"success": True, "data": rows}), 200


@app.route("/api/reference/categories", methods=["GET"])
def get_reference_categories():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT primary_type FROM crimes ORDER BY primary_type ASC;")
    categories = [row["primary_type"] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"success": True, "categories": categories}), 200


@app.route("/api/reference/communities", methods=["GET"])
def get_reference_communities():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT community_code, community_name FROM ref_communities ORDER BY community_name ASC;")
    communities = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({"success": True, "communities": communities}), 200


# Initial check on startup
if not os.path.exists(DATABASE_NAME):
    run_data_ingestion_pipeline()
else:
    try:
        check_conn = get_db_connection()
        check_cursor = check_conn.cursor()
        check_cursor.execute("SELECT COUNT(*) as count FROM crimes;")
        record_count = check_cursor.fetchone()["count"]
        check_conn.close()
        if record_count == 0:
            run_data_ingestion_pipeline()
        else:
            if not INGESTION_QUALITY_REPORT:
                run_data_ingestion_pipeline()
    except Exception:
        run_data_ingestion_pipeline()


if __name__ == "__main__":
    import socket

    def check_port_free(port_num):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port_num))
            sock.close()
            return True
        except OSError:
            sock.close()
            return False

    designated_port = int(os.environ.get("FLASK_PORT", 5000 if check_port_free(5000) else 5001))
    print(f"Starting Flask server on port {designated_port}...")
    app.run(host="0.0.0.0", port=designated_port, debug=False)
