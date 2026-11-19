import hashlib
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trustbank.db")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            cnic TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'CUSTOMER' CHECK (role IN ('CUSTOMER', 'ADMIN')),
            city TEXT NOT NULL,
            monthly_income REAL NOT NULL CHECK (monthly_income >= 0),
            employment_status TEXT NOT NULL CHECK (employment_status IN ('Salaried', 'Self-Employed', 'Business', 'Student')),
            credit_score INTEGER DEFAULT 720,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            account_number TEXT UNIQUE NOT NULL,
            iban TEXT UNIQUE NOT NULL,
            card_number TEXT UNIQUE NOT NULL,
            card_cvv TEXT NOT NULL,
            card_expiry TEXT NOT NULL,
            account_type TEXT DEFAULT 'Current' CHECK (account_type IN ('Current', 'Savings', 'Asaan')),
            balance REAL NOT NULL DEFAULT 0.0 CHECK (balance >= 0),
            status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'FROZEN', 'CLOSED')),
            opened_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_reference TEXT UNIQUE NOT NULL,
            sender_account_id INTEGER NOT NULL,
            receiver_account_id INTEGER,
            receiver_label TEXT NOT NULL,
            amount REAL NOT NULL CHECK (amount > 0),
            txn_city TEXT NOT NULL,
            txn_hour INTEGER NOT NULL CHECK (txn_hour BETWEEN 0 AND 23),
            fraud_score REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL CHECK (status IN ('COMPLETED', 'BLOCKED_FRAUD')),
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (sender_account_id) REFERENCES accounts(account_id),
            FOREIGN KEY (receiver_account_id) REFERENCES accounts(account_id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id INTEGER NOT NULL,
            account_id INTEGER NOT NULL,
            fraud_score REAL NOT NULL,
            risk_factors TEXT NOT NULL,
            resolution_status TEXT DEFAULT 'OPEN' CHECK (resolution_status IN ('OPEN', 'RESOLVED', 'FALSE_POSITIVE')),
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (txn_id) REFERENCES transactions(txn_id) ON DELETE CASCADE,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS loan_applications (
            loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            requested_amount REAL NOT NULL CHECK (requested_amount > 0),
            tenure_months INTEGER NOT NULL CHECK (tenure_months > 0),
            existing_debt REAL NOT NULL DEFAULT 0.0,
            purpose TEXT NOT NULL,
            default_probability REAL NOT NULL,
            recommended_limit REAL NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('APPROVED', 'REJECTED', 'MANUAL_REVIEW')),
            ai_reasoning TEXT NOT NULL,
            applied_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            engine_source TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            confidence_score REAL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_sender ON transactions(sender_account_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_status ON transactions(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_account ON fraud_alerts(account_id);")

        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_auto_freeze_on_fraud
        AFTER INSERT ON fraud_alerts
        WHEN NEW.fraud_score >= 75.0
        BEGIN
            UPDATE accounts
            SET status = 'FROZEN'
            WHERE account_id = NEW.account_id;

            INSERT INTO ai_audit_logs (entity_type, entity_id, engine_source, action_taken, confidence_score)
            VALUES (
                'ACCOUNT',
                NEW.account_id,
                'DB_TRIGGER_SHIELD',
                'Account automatically FROZEN due to critical fraud alert: ' || NEW.risk_factors,
                NEW.fraud_score
            );
        END;
        """)

        # Clean Database: Only Compliance Officer system account is initialized
        existing = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing == 0:
            admin_pw = hash_password("admin123")
            cursor.execute(
                "INSERT INTO customers (full_name, cnic, email, password_hash, role, city, monthly_income, employment_status, credit_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("Compliance Officer", "35200-0000000-0", "admin@trustbank.pk", admin_pw, "ADMIN", "Lahore", 350000, "Salaried", 850),
            )
            cursor.execute(
                "INSERT INTO accounts (customer_id, account_number, iban, card_number, card_cvv, card_expiry, account_type, balance, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1, "TB-9999-0001", "PK36TRST00000099990001", "4532 9999 0000 0001", "999", "12/30", "Current", 5000000.00, "ACTIVE"),
            )