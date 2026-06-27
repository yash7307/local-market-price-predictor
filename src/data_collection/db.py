import sqlite3
from pathlib import Path
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parents[2] / "data" / "pricing_logs.db"

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS optimization_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        category TEXT,
        current_price REAL,
        cost_price REAL,
        competitor_price REAL,
        inventory INTEGER,
        optimal_price REAL,
        expected_demand REAL,
        expected_profit REAL,
        context_json TEXT,
        action_taken TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

def log_optimization(
    category: str,
    current_price: float,
    cost_price: float,
    competitor_price: float,
    inventory: int,
    optimal_price: float,
    expected_demand: float,
    expected_profit: float,
    context: dict,
    action_taken: str = "RECOMMENDED"
) -> int:
    """Log an optimization run to the database."""
    init_db()  # Ensure DB exists
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO optimization_logs (
        timestamp, category, current_price, cost_price, competitor_price, 
        inventory, optimal_price, expected_demand, expected_profit, 
        context_json, action_taken
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        category,
        current_price,
        cost_price,
        competitor_price,
        inventory,
        optimal_price,
        expected_demand,
        expected_profit,
        json.dumps(context),
        action_taken
    ))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def get_recent_logs(limit: int = 50) -> list[dict]:
    """Retrieve recent optimization logs."""
    init_db()
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM optimization_logs 
    ORDER BY timestamp DESC 
    LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()
    print("Database ready.")
