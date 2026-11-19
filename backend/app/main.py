import json
import os
import random
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .database import get_db, hash_password, init_database
from .ml_pipeline import ARTIFACTS_DIR, train_and_export_all
from .schemas import (
    DepositRequest,
    LoanEvaluationRequest,
    LoginRequest,
    RegisterAccountRequest,
    TransferRequest,
    UnfreezeRequest,
)

app = FastAPI(
    title="TrustBank AI — Enterprise Core Banking & Dual-ML Engine",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fraud_model = None
loan_model = None


@app.on_event("startup")
def startup_event():
    global fraud_model, loan_model
    init_database()
    fraud_path = os.path.join(ARTIFACTS_DIR, "fraud_pipeline.pkl")
    loan_path = os.path.join(ARTIFACTS_DIR, "loan_pipeline.pkl")
    if not (os.path.exists(fraud_path) and os.path.exists(loan_path)):
        train_and_export_all()
    fraud_model = joblib.load(fraud_path)
    loan_model = joblib.load(loan_path)


@app.post("/api/auth/login")
def login_user(payload: LoginRequest):
    with get_db() as conn:
        cur = conn.cursor()
        user = cur.execute("""
            SELECT c.customer_id, c.full_name, c.cnic, c.email, c.role, c.city,
                   c.monthly_income, c.employment_status, c.credit_score,
                   a.account_id, a.account_number, a.iban, a.card_number, a.card_cvv,
                   a.card_expiry, a.account_type, a.balance, a.status
            FROM customers c
            JOIN accounts a ON c.customer_id = a.customer_id
            WHERE (LOWER(c.email) = LOWER(?) OR LOWER(a.account_number) = LOWER(?))
              AND c.password_hash = ?
        """, (payload.email.strip(), payload.email.strip(), hash_password(payload.password))).fetchone()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid email/account number or password.")

        return {"user": dict(user)}


@app.post("/api/auth/register")
def register_new_account(payload: RegisterAccountRequest):
    with get_db() as conn:
        cur = conn.cursor()
        exists = cur.execute(
            "SELECT 1 FROM customers WHERE LOWER(email) = LOWER(?) OR cnic = ?",
            (payload.email.strip(), payload.cnic.strip()),
        ).fetchone()
        if exists:
            raise HTTPException(status_code=400, detail="An account with this Email or CNIC already exists.")

        credit_score = 740 if payload.monthly_income >= 120000 else 680
        cur.execute("""
            INSERT INTO customers (full_name, cnic, email, password_hash, role, city, monthly_income, employment_status, credit_score)
            VALUES (?, ?, ?, ?, 'CUSTOMER', ?, ?, ?, ?)
        """, (
            payload.full_name.strip(),
            payload.cnic.strip(),
            payload.email.strip().lower(),
            hash_password(payload.password),
            payload.city.strip(),
            payload.monthly_income,
            payload.employment_status,
            credit_score,
        ))
        customer_id = cur.lastrowid

        rand_suffix = random.randint(1000, 9999)
        acc_num = f"TB-{1000 + customer_id}-{rand_suffix}"
        iban = f"PK36TRST000000{1000 + customer_id}{rand_suffix}"
        card_num = f"4532 {random.randint(1000,9999)} {random.randint(1000,9999)} {rand_suffix}"
        cvv = str(random.randint(100, 999))

        cur.execute("""
            INSERT INTO accounts (customer_id, account_number, iban, card_number, card_cvv, card_expiry, account_type, balance, status)
            VALUES (?, ?, ?, ?, ?, '08/31', ?, ?, 'ACTIVE')
        """, (customer_id, acc_num, iban, card_num, cvv, payload.account_type, payload.initial_deposit))

        user = cur.execute("""
            SELECT c.customer_id, c.full_name, c.cnic, c.email, c.role, c.city,
                   c.monthly_income, c.employment_status, c.credit_score,
                   a.account_id, a.account_number, a.iban, a.card_number, a.card_cvv,
                   a.card_expiry, a.account_type, a.balance, a.status
            FROM customers c
            JOIN accounts a ON c.customer_id = a.customer_id
            WHERE c.customer_id = ?
        """, (customer_id,)).fetchone()

        return {"user": dict(user)}


@app.get("/api/accounts/lookup/{identifier}")
def lookup_account_title(identifier: str):
    clean = identifier.strip().upper()
    with get_db() as conn:
        cur = conn.cursor()
        acc = cur.execute("""
            SELECT a.account_id, a.account_number, a.iban, c.full_name, c.city
            FROM accounts a
            JOIN customers c ON a.customer_id = c.customer_id
            WHERE UPPER(a.account_number) = ? OR UPPER(a.iban) = ?
        """, (clean, clean)).fetchone()

        if not acc:
            raise HTTPException(status_code=404, detail="No TrustBank account found with this number/IBAN.")
        return dict(acc)


@app.post("/api/accounts/deposit")
def deposit_funds(payload: DepositRequest):
    with get_db() as conn:
        cur = conn.cursor()
        acc = cur.execute("""
            SELECT a.*, c.full_name, c.city
            FROM accounts a JOIN customers c ON a.customer_id = c.customer_id
            WHERE a.account_id = ?
        """, (payload.account_id,)).fetchone()

        if not acc:
            raise HTTPException(status_code=404, detail="Account not found.")
        if acc["status"] == "FROZEN":
            raise HTTPException(status_code=403, detail="Account is FROZEN. Unfreeze before depositing.")

        cur.execute("UPDATE accounts SET balance = balance + ? WHERE account_id = ?", (payload.amount, payload.account_id))
        txn_ref = f"DEP-{uuid.uuid4().hex[:8].upper()}"

        cur.execute("""
            INSERT INTO transactions (txn_reference, sender_account_id, receiver_account_id, receiver_label, amount, txn_city, txn_hour, fraud_score, status)
            VALUES (?, ?, ?, ?, ?, ?, 12, 0.0, 'COMPLETED')
        """, (txn_ref, payload.account_id, payload.account_id, f"Self Deposit ({payload.deposit_source})", payload.amount, acc["city"]))

        return {
            "status": "COMPLETED",
            "txn_reference": txn_ref,
            "message": f"PKR {payload.amount:,.0f} deposited into {acc['account_number']} successfully!",
        }


@app.get("/api/dashboard")
def get_dashboard_overview():
    with get_db() as conn:
        cur = conn.cursor()
        accounts = [dict(row) for row in cur.execute("""
            SELECT a.*, c.full_name, c.cnic, c.email, c.role, c.city, c.monthly_income, c.employment_status, c.credit_score
            FROM accounts a
            JOIN customers c ON a.customer_id = c.customer_id
            ORDER BY a.account_id
        """).fetchall()]

        transactions = [dict(row) for row in cur.execute("""
            SELECT t.*, a.account_number AS sender_account_number, c.full_name AS sender_name
            FROM transactions t
            JOIN accounts a ON t.sender_account_id = a.account_id
            JOIN customers c ON a.customer_id = c.customer_id
            ORDER BY t.txn_id DESC LIMIT 40
        """).fetchall()]

        alerts = [dict(row) for row in cur.execute("""
            SELECT f.*, a.account_number, c.full_name
            FROM fraud_alerts f
            JOIN accounts a ON f.account_id = a.account_id
            JOIN customers c ON a.customer_id = c.customer_id
            ORDER BY f.alert_id DESC LIMIT 20
        """).fetchall()]

        loans = [dict(row) for row in cur.execute("""
            SELECT l.*, c.full_name, c.monthly_income, c.city
            FROM loan_applications l
            JOIN customers c ON l.customer_id = c.customer_id
            ORDER BY l.loan_id DESC LIMIT 20
        """).fetchall()]

        audit_logs = [dict(row) for row in cur.execute("""
            SELECT * FROM ai_audit_logs ORDER BY log_id DESC LIMIT 25
        """).fetchall()]

        stats = {
            "total_liquidity": sum(a["balance"] for a in accounts),
            "active_accounts": sum(1 for a in accounts if a["status"] == "ACTIVE"),
            "frozen_accounts": sum(1 for a in accounts if a["status"] == "FROZEN"),
            "total_transactions": cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0],
            "blocked_frauds": cur.execute("SELECT COUNT(*) FROM transactions WHERE status = 'BLOCKED_FRAUD'").fetchone()[0],
            "fraud_amount_prevented": cur.execute("SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE status = 'BLOCKED_FRAUD'").fetchone()[0],
            "approved_loans": cur.execute("SELECT COUNT(*) FROM loan_applications WHERE decision = 'APPROVED'").fetchone()[0],
        }

        return {
            "stats": stats,
            "accounts": accounts,
            "transactions": transactions,
            "alerts": alerts,
            "loans": loans,
            "audit_logs": audit_logs,
        }


@app.post("/api/transfer")
def execute_fund_transfer(payload: TransferRequest):
    with get_db() as conn:
        cur = conn.cursor()
        sender = cur.execute("""
            SELECT a.*, c.city AS home_city, c.full_name
            FROM accounts a JOIN customers c ON a.customer_id = c.customer_id
            WHERE a.account_id = ?
        """, (payload.sender_account_id,)).fetchone()

        if not sender:
            raise HTTPException(status_code=404, detail="Invalid sender account.")
        if sender["status"] == "FROZEN":
            raise HTTPException(status_code=403, detail=f"Account {sender['account_number']} is FROZEN due to security alert. Contact Compliance Admin to unfreeze.")
        if sender["balance"] < payload.amount:
            raise HTTPException(status_code=400, detail="Insufficient account balance for this transfer.")

        clean_acc = payload.beneficiary_account_number.strip().upper()
        is_internal = payload.beneficiary_bank.strip().lower() in ["trustbank ai", "trustbank"]
        receiver = None

        if is_internal:
            receiver = cur.execute("""
                SELECT a.*, c.full_name
                FROM accounts a JOIN customers c ON a.customer_id = c.customer_id
                WHERE UPPER(a.account_number) = ? OR UPPER(a.iban) = ?
            """, (clean_acc, clean_acc)).fetchone()

            if not receiver:
                raise HTTPException(status_code=404, detail=f"TrustBank account '{payload.beneficiary_account_number}' does not exist.")
            if receiver["account_id"] == sender["account_id"]:
                raise HTTPException(status_code=400, detail="You cannot transfer money to your own account. Use 'Add Cash / Deposit' instead.")

            receiver_id = receiver["account_id"]
            receiver_label = f"{receiver['full_name']} • TrustBank ({receiver['account_number']})"
        else:
            receiver_id = None
            title = payload.beneficiary_title.strip() if payload.beneficiary_title and payload.beneficiary_title.strip() else "External Beneficiary"
            receiver_label = f"{title} • {payload.beneficiary_bank} ({payload.beneficiary_account_number.strip()})"

        is_city_changed = 1 if payload.txn_city.strip().lower() != sender["home_city"].strip().lower() else 0
        balance_drain_ratio = min(payload.amount / max(sender["balance"], 1.0), 1.0)
        recent_txns = cur.execute("""
            SELECT COUNT(*) FROM transactions
            WHERE sender_account_id = ? AND created_at >= datetime('now', 'localtime', '-1 hour')
        """, (sender["account_id"],)).fetchone()[0]

        features_df = pd.DataFrame([{
            "amount": payload.amount,
            "txn_hour": payload.txn_hour,
            "is_city_changed": is_city_changed,
            "balance_drain_ratio": balance_drain_ratio,
            "velocity_1h": recent_txns + 1,
        }])

        fraud_prob = float(fraud_model.predict_proba(features_df)[0][1]) * 100
        fraud_score = round(fraud_prob, 2)
        txn_ref = f"TXN-{uuid.uuid4().hex[:8].upper()}"

        reasons = []
        if payload.amount > 150000:
            reasons.append(f"High Transfer Amount (PKR {payload.amount:,.0f})")
        if 0 <= payload.txn_hour <= 5:
            reasons.append(f"Odd-Hour Activity ({payload.txn_hour}:00 hrs)")
        if is_city_changed:
            reasons.append(f"Geo-Location Shift ({sender['home_city']} -> {payload.txn_city})")
        if balance_drain_ratio > 0.70:
            reasons.append(f"High Balance Drain ({balance_drain_ratio*100:.0f}%)")
        risk_summary = " + ".join(reasons) if reasons else "Anomalous behavioral pattern detected by ML"

        if fraud_score >= 65.0:
            cur.execute("""
                INSERT INTO transactions (txn_reference, sender_account_id, receiver_account_id, receiver_label, amount, txn_city, txn_hour, fraud_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'BLOCKED_FRAUD')
            """, (txn_ref, sender["account_id"], receiver_id, receiver_label, payload.amount, payload.txn_city, payload.txn_hour, fraud_score))
            txn_id = cur.lastrowid

            cur.execute("""
                INSERT INTO fraud_alerts (txn_id, account_id, fraud_score, risk_factors)
                VALUES (?, ?, ?, ?)
            """, (txn_id, sender["account_id"], fraud_score, risk_summary))

            cur.execute("""
                INSERT INTO ai_audit_logs (entity_type, entity_id, engine_source, action_taken, confidence_score)
                VALUES ('TRANSACTION', ?, 'ML_FRAUD_SHIELD', ?, ?)
            """, (txn_id, f"Blocked IBFT {txn_ref} to {receiver_label}: {risk_summary}", fraud_score))

            updated_status = cur.execute("SELECT status FROM accounts WHERE account_id = ?", (sender["account_id"],)).fetchone()["status"]

            return {
                "status": "BLOCKED_FRAUD",
                "txn_reference": txn_ref,
                "fraud_score": fraud_score,
                "account_status": updated_status,
                "message": f"FRAUD BLOCKED! Risk Score: {fraud_score}%. Reason: {risk_summary}.",
            }

        cur.execute("UPDATE accounts SET balance = balance - ? WHERE account_id = ?", (payload.amount, sender["account_id"]))
        if is_internal and receiver_id:
            cur.execute("UPDATE accounts SET balance = balance + ? WHERE account_id = ?", (payload.amount, receiver_id))

        cur.execute("""
            INSERT INTO transactions (txn_reference, sender_account_id, receiver_account_id, receiver_label, amount, txn_city, txn_hour, fraud_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED')
        """, (txn_ref, sender["account_id"], receiver_id, receiver_label, payload.amount, payload.txn_city, payload.txn_hour, fraud_score))
        txn_id = cur.lastrowid

        cur.execute("""
            INSERT INTO ai_audit_logs (entity_type, entity_id, engine_source, action_taken, confidence_score)
            VALUES ('TRANSACTION', ?, 'ML_FRAUD_SHIELD', ?, ?)
        """, (txn_id, f"IBFT Settled {txn_ref} -> {receiver_label} (PKR {payload.amount:,.0f})", fraud_score))

        return {
            "status": "COMPLETED",
            "txn_reference": txn_ref,
            "fraud_score": fraud_score,
            "account_status": "ACTIVE",
            "message": f"PKR {payload.amount:,.0f} sent via IBFT to {receiver_label}.",
        }


@app.post("/api/loans/evaluate")
def evaluate_loan_application(payload: LoanEvaluationRequest):
    with get_db() as conn:
        cur = conn.cursor()
        customer = cur.execute("""
            SELECT c.*, a.balance, a.account_id
            FROM customers c JOIN accounts a ON c.customer_id = a.customer_id
            WHERE c.customer_id = ?
        """, (payload.customer_id,)).fetchone()

        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found.")

        txn_count = cur.execute("SELECT COUNT(*) FROM transactions WHERE sender_account_id = ?", (customer["account_id"],)).fetchone()[0]

        features_df = pd.DataFrame([{
            "monthly_income": customer["monthly_income"],
            "account_balance": customer["balance"],
            "requested_amount": payload.requested_amount,
            "existing_debt": payload.existing_debt,
            "tenure_months": payload.tenure_months,
            "avg_monthly_txns": max(txn_count * 4, 6),
        }])

        default_prob = round(float(loan_model.predict_proba(features_df)[0][1]) * 100, 2)
        monthly_emi = (payload.requested_amount + payload.existing_debt) / payload.tenure_months
        dti_ratio = round((monthly_emi / max(customer["monthly_income"], 1.0)) * 100, 1)

        safe_monthly_capacity = max((customer["monthly_income"] * 0.40) - (payload.existing_debt / payload.tenure_months), 0)
        recommended_limit = round(safe_monthly_capacity * payload.tenure_months, -3)

        if default_prob < 35.0 and dti_ratio <= 45.0:
            decision = "APPROVED"
            reasoning = f"Strong repayment capacity (DTI: {dti_ratio}%, Default Risk: {default_prob}%)."
        elif default_prob < 65.0:
            decision = "MANUAL_REVIEW"
            reasoning = f"Moderate risk profile (DTI: {dti_ratio}%). Recommend capping loan at PKR {recommended_limit:,.0f}."
        else:
            decision = "REJECTED"
            reasoning = f"High default risk ({default_prob}%) with elevated Debt-to-Income ratio ({dti_ratio}%)."

        cur.execute("""
            INSERT INTO loan_applications (customer_id, requested_amount, tenure_months, existing_debt, purpose, default_probability, recommended_limit, decision, ai_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (payload.customer_id, payload.requested_amount, payload.tenure_months, payload.existing_debt, payload.purpose, default_prob, recommended_limit, decision, reasoning))
        loan_id = cur.lastrowid

        cur.execute("""
            INSERT INTO ai_audit_logs (entity_type, entity_id, engine_source, action_taken, confidence_score)
            VALUES ('LOAN_APPLICATION', ?, 'ML_LOAN_ENGINE', ?, ?)
        """, (loan_id, f"Loan #{loan_id} -> {decision}: {reasoning}", default_prob))

        return {
            "loan_id": loan_id,
            "customer_name": customer["full_name"],
            "decision": decision,
            "default_probability": default_prob,
            "dti_ratio": dti_ratio,
            "recommended_limit": recommended_limit,
            "ai_reasoning": reasoning,
        }


@app.post("/api/admin/unfreeze")
def unfreeze_account(payload: UnfreezeRequest):
    with get_db() as conn:
        cur = conn.cursor()
        acc = cur.execute("SELECT * FROM accounts WHERE account_id = ?", (payload.account_id,)).fetchone()
        if not acc:
            raise HTTPException(status_code=404, detail="Account not found.")

        cur.execute("UPDATE accounts SET status = 'ACTIVE' WHERE account_id = ?", (payload.account_id,))
        cur.execute("UPDATE fraud_alerts SET resolution_status = 'RESOLVED' WHERE account_id = ? AND resolution_status = 'OPEN'", (payload.account_id,))
        cur.execute("""
            INSERT INTO ai_audit_logs (entity_type, entity_id, engine_source, action_taken, confidence_score)
            VALUES ('ACCOUNT', ?, 'ADMIN_OVERRIDE', ?, 100.0)
        """, (payload.account_id, f"Account {acc['account_number']} unfrozen: {payload.admin_note}"))

        return {"status": "ACTIVE", "message": f"Account {acc['account_number']} is now ACTIVE."}


@app.get("/api/metrics")
def get_model_metrics():
    metrics_path = os.path.join(ARTIFACTS_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        train_and_export_all()
    with open(metrics_path, "r") as f:
        return json.load(f)