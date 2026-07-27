import sqlite3
import pandas as pd

def build_database():
    conn = sqlite3.connect("data/hospital_assistant.db")

    medicine_df = pd.read_csv("data/Medicine_Details.csv")
    medicine_df.to_sql("medicines", conn, if_exists="replace", index=False)

    hospital_df = pd.read_csv("data/HospInfo.csv")
    hospital_df.to_sql("hospitals", conn, if_exists="replace", index=False)

    # Optional: an index for faster lookups, and something concrete to point to
    conn.execute("CREATE INDEX IF NOT EXISTS idx_medicine_name ON medicines(`Medicine Name`)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hospital_state ON hospitals(State)")
    conn.commit()

    print("Database built: data/hospital_assistant.db")
    print("Tables:", conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    conn.close()

build_database()