"""
Generate synthetic fintech data for the Transaction Analytics & Anomaly Detection
Pipeline: customers, transactions, and support tickets.

Everything is seeded for reproducibility. Three realistic anomaly patterns are
deliberately baked into the transaction stream so that the downstream anomaly
detection step (python/anomaly_detection/detect_anomalies.py) has genuine
statistical signal to find -- the detector itself has no knowledge of these
constants, it discovers them from the aggregated time series.

Injected events (see README "Key Findings" for what the detector recovers):
  1. Fraud spike   - West region, 2025-09-08 to 2025-09-14 (elevated volume,
                      late-night timestamps, high decline rate)
  2. Processing outage - 2025-11-12 (system-wide transaction volume collapses
                      to ~15% of baseline for the day)
  3. Holiday surge - 2025-11-28 to 2025-12-24 (volume + spend climb ~1.6-2x,
                      skewed toward retail/electronics)

Run:
    python python/etl/generate_synthetic_data.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# --------------------------------------------------------------------------
# Config / reproducibility
# --------------------------------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

DATA_DIR = "data"

N_CUSTOMERS = 5_000
N_TICKETS_TARGET = 3_000

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 6, 30)
ALL_DATES = pd.date_range(START_DATE, END_DATE, freq="D")
TOTAL_DAYS = len(ALL_DATES)

# Target average daily transaction volume to land near ~250,000 total.
BASE_DAILY_RATE = 250_000 / TOTAL_DAYS  # ~458/day before seasonality

# --------------------------------------------------------------------------
# Reference domains
# --------------------------------------------------------------------------

REGIONS = ["Northeast", "Midwest", "South", "West", "International"]
REGION_WEIGHTS = [0.27, 0.22, 0.24, 0.20, 0.07]

ACCOUNT_TYPES = ["Basic", "Premium", "Business"]
ACCOUNT_WEIGHTS = [0.55, 0.32, 0.13]
# Relative transaction activity multiplier by account type (Business banks
# transact far more often than a Basic personal account).
ACCOUNT_ACTIVITY = {"Basic": 1.0, "Premium": 1.8, "Business": 3.0}

RISK_SEGMENTS = ["Low", "Medium", "High"]
RISK_WEIGHTS_DEFAULT = [0.70, 0.22, 0.08]
# West region skews slightly riskier, which is part of why the fraud ring
# targets it during the injected spike week.
RISK_WEIGHTS_WEST = [0.55, 0.30, 0.15]

CHANNELS = ["mobile_app", "web", "pos", "atm", "phone"]
CHANNEL_WEIGHTS_BY_ACCOUNT = {
    "Basic":    [0.45, 0.20, 0.25, 0.08, 0.02],
    "Premium":  [0.35, 0.35, 0.15, 0.05, 0.10],
    "Business": [0.15, 0.40, 0.20, 0.05, 0.20],
}

STATUSES = ["completed", "declined", "pending", "reversed"]
STATUS_WEIGHTS_DEFAULT = [0.90, 0.06, 0.02, 0.02]

MERCHANT_CATEGORIES = [
    "groceries", "dining", "travel", "electronics", "utilities",
    "entertainment", "healthcare", "retail", "fuel", "subscription",
    "transfer", "other",
]
CATEGORY_WEIGHTS_DEFAULT = [0.17, 0.14, 0.05, 0.09, 0.09, 0.08, 0.06, 0.13, 0.08, 0.06, 0.03, 0.02]

# lognormal (mu, sigma) per category, mu expressed as ln(target median $)
CATEGORY_AMOUNT_PARAMS = {
    "groceries":      (np.log(45), 0.40),
    "dining":         (np.log(32), 0.50),
    "travel":         (np.log(420), 0.60),
    "electronics":    (np.log(260), 0.70),
    "utilities":      (np.log(110), 0.30),
    "entertainment":  (np.log(55), 0.50),
    "healthcare":     (np.log(140), 0.60),
    "retail":         (np.log(85), 0.60),
    "fuel":           (np.log(50), 0.30),
    "subscription":   (np.log(14), 0.30),
    "transfer":       (np.log(180), 0.90),
    "other":          (np.log(40), 0.60),
}

# --------------------------------------------------------------------------
# Injected anomaly windows
# --------------------------------------------------------------------------

FRAUD_START, FRAUD_END = datetime(2025, 9, 8), datetime(2025, 9, 14)
OUTAGE_DATE = datetime(2025, 11, 12)
HOLIDAY_START, HOLIDAY_END = datetime(2025, 11, 28), datetime(2025, 12, 24)


def in_range(d: datetime, start: datetime, end: datetime) -> bool:
    return start <= d <= end


# --------------------------------------------------------------------------
# Step 1: customers.csv
# --------------------------------------------------------------------------

def generate_customers() -> pd.DataFrame:
    customer_ids = [f"CUST{i:06d}" for i in range(1, N_CUSTOMERS + 1)]
    regions = rng.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)
    account_types = rng.choice(ACCOUNT_TYPES, size=N_CUSTOMERS, p=ACCOUNT_WEIGHTS)

    risk_segments = np.empty(N_CUSTOMERS, dtype=object)
    for i, region in enumerate(regions):
        weights = RISK_WEIGHTS_WEST if region == "West" else RISK_WEIGHTS_DEFAULT
        risk_segments[i] = rng.choice(RISK_SEGMENTS, p=weights)

    # 70% "legacy" customers signed up before the analysis window; 30% sign
    # up organically during the 18-month window (drives natural growth in
    # the daily transaction count over time).
    is_legacy = rng.random(N_CUSTOMERS) < 0.70
    legacy_start = datetime(2018, 1, 1)
    signup_dates = []
    for legacy in is_legacy:
        if legacy:
            delta_days = (START_DATE - legacy_start).days
            offset = int(rng.integers(0, max(delta_days, 1)))
            signup_dates.append(legacy_start + timedelta(days=offset))
        else:
            delta_days = (END_DATE - START_DATE).days
            offset = int(rng.integers(0, delta_days))
            signup_dates.append(START_DATE + timedelta(days=offset))

    # Per-customer activity multiplier adds heterogeneity: some customers
    # are simply heavier users than others, independent of account type.
    activity_multiplier = rng.lognormal(mean=0.0, sigma=0.6, size=N_CUSTOMERS)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": [d.date().isoformat() for d in signup_dates],
        "region": regions,
        "account_type": account_types,
        "risk_segment": risk_segments,
        "_activity_weight": [
            ACCOUNT_ACTIVITY[a] * m for a, m in zip(account_types, activity_multiplier)
        ],
    })
    return df


# --------------------------------------------------------------------------
# Step 2: transactions.csv
# --------------------------------------------------------------------------

def daily_seasonality_multiplier(date: datetime) -> float:
    """Mild weekly seasonality: modest lift on weekends (consumer retail)."""
    return 1.10 if date.weekday() >= 5 else 1.00


def category_weights_for(date: datetime) -> list[float]:
    if in_range(date, HOLIDAY_START, HOLIDAY_END):
        # Holiday surge skews spend toward retail/electronics/entertainment.
        w = np.array([0.12, 0.11, 0.05, 0.16, 0.06, 0.10, 0.04, 0.20, 0.06, 0.05, 0.03, 0.02])
        return list(w / w.sum())
    return CATEGORY_WEIGHTS_DEFAULT


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    customers_sorted = customers.sort_values("signup_date").reset_index(drop=True)
    signup_dates_sorted = pd.to_datetime(customers_sorted["signup_date"]).values.astype("datetime64[D]")
    cust_ids_sorted = customers_sorted["customer_id"].to_numpy()
    weights_sorted = customers_sorted["_activity_weight"].to_numpy()
    west_mask_sorted = (customers_sorted["region"] == "West").to_numpy()

    acct_lookup = customers_sorted.set_index("customer_id")["account_type"]

    daily_frames = []
    txn_counter = 1

    for date in ALL_DATES:
        date_np = np.datetime64(date.date(), "D")
        eligible_n = int(np.searchsorted(signup_dates_sorted, date_np, side="right"))
        if eligible_n == 0:
            continue

        pool_ids = cust_ids_sorted[:eligible_n]
        pool_weights = weights_sorted[:eligible_n]
        pool_west = west_mask_sorted[:eligible_n]

        # ---- baseline daily rate with seasonality + injected events ----
        rate = BASE_DAILY_RATE * daily_seasonality_multiplier(date)

        if in_range(date, HOLIDAY_START, HOLIDAY_END):
            # Ramp up toward Black Friday / Christmas, not a flat block.
            peak_days = {datetime(2025, 11, 28), datetime(2025, 11, 29), datetime(2025, 12, 23), datetime(2025, 12, 24)}
            rate *= 1.9 if date in peak_days else 1.5

        if date == OUTAGE_DATE:
            # System-wide processing outage: volume collapses for the day.
            rate *= 0.15

        n_today = int(rng.poisson(rate))

        # ---- sample base transactions for the day ----
        p = pool_weights / pool_weights.sum()
        sampled_customers = rng.choice(pool_ids, size=n_today, p=p)

        cat_weights = category_weights_for(date)
        categories = rng.choice(MERCHANT_CATEGORIES, size=n_today, p=cat_weights)

        amounts = np.empty(n_today)
        for cat in MERCHANT_CATEGORIES:
            mask = categories == cat
            n_cat = mask.sum()
            if n_cat == 0:
                continue
            mu, sigma = CATEGORY_AMOUNT_PARAMS[cat]
            amounts[mask] = rng.lognormal(mean=mu, sigma=sigma, size=n_cat)
        amounts = np.round(amounts, 2)

        # channel: depends on the customer's account type
        cust_accounts = acct_lookup.loc[sampled_customers].to_numpy()
        channels = np.empty(n_today, dtype=object)
        for acct in ACCOUNT_TYPES:
            mask = cust_accounts == acct
            n_acct = mask.sum()
            if n_acct == 0:
                continue
            channels[mask] = rng.choice(CHANNELS, size=n_acct, p=CHANNEL_WEIGHTS_BY_ACCOUNT[acct])

        statuses = rng.choice(STATUSES, size=n_today, p=STATUS_WEIGHTS_DEFAULT)

        # random time-of-day, weighted toward business hours
        seconds_in_day = rng.normal(loc=14 * 3600, scale=5 * 3600, size=n_today)
        seconds_in_day = np.clip(seconds_in_day, 0, 86_399).astype(int)
        timestamps = [date + timedelta(seconds=int(s)) for s in seconds_in_day]

        day_df = pd.DataFrame({
            "customer_id": sampled_customers,
            "timestamp": timestamps,
            "amount": amounts,
            "merchant_category": categories,
            "channel": channels,
            "status": statuses,
        })

        # ---- injected fraud spike: West region, 2025-09-08 to 2025-09-14 ----
        if in_range(date, FRAUD_START, FRAUD_END) and pool_west.sum() > 0:
            n_fraud = int(rng.integers(140, 190))
            fraud_pool = pool_ids[pool_west]
            fraud_customers = rng.choice(fraud_pool, size=n_fraud)
            fraud_categories = rng.choice(["electronics", "retail", "transfer"], size=n_fraud, p=[0.45, 0.35, 0.20])
            fraud_amounts = np.round(rng.uniform(50, 900, size=n_fraud), 2)
            fraud_channels = np.full(n_fraud, "web", dtype=object)
            # heavy decline rate as fraud attempts get blocked
            fraud_statuses = rng.choice(STATUSES, size=n_fraud, p=[0.35, 0.50, 0.10, 0.05])
            # concentrated in the 1am-4am window
            fraud_seconds = rng.integers(1 * 3600, 4 * 3600, size=n_fraud)
            fraud_timestamps = [date + timedelta(seconds=int(s)) for s in fraud_seconds]

            fraud_df = pd.DataFrame({
                "customer_id": fraud_customers,
                "timestamp": fraud_timestamps,
                "amount": fraud_amounts,
                "merchant_category": fraud_categories,
                "channel": fraud_channels,
                "status": fraud_statuses,
            })
            day_df = pd.concat([day_df, fraud_df], ignore_index=True)

        daily_frames.append(day_df)

    txns = pd.concat(daily_frames, ignore_index=True)
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    txns.insert(0, "transaction_id", [f"TXN{i:08d}" for i in range(1, len(txns) + 1)])
    txns["timestamp"] = txns["timestamp"].apply(lambda t: t.isoformat(sep=" "))
    return txns


# --------------------------------------------------------------------------
# Step 3: support_tickets.csv
# --------------------------------------------------------------------------

TICKET_CATEGORIES = [
    "failed_transaction", "login_issue", "fee_complaint", "slow_support",
    "fraud_report", "account_access", "positive_feedback", "general_inquiry",
]
TICKET_CATEGORY_WEIGHTS = [0.18, 0.15, 0.14, 0.12, 0.08, 0.10, 0.15, 0.08]

SUBJECT_TEMPLATES = {
    "failed_transaction": [
        "Payment declined for no reason",
        "Transaction failed but money was taken",
        "Charge did not go through at {merchant}",
        "My {category} payment keeps failing",
        "Failed transaction, need urgent help",
    ],
    "login_issue": [
        "Can't log into my account",
        "App keeps logging me out",
        "Password reset link not working",
        "Two-factor code never arrives",
        "Locked out of online banking",
    ],
    "fee_complaint": [
        "Unexpected fee on my statement",
        "Why was I charged an overdraft fee",
        "Monthly fee increased without notice",
        "Dispute a service charge",
        "Too many hidden fees",
    ],
    "slow_support": [
        "Still waiting on a reply from support",
        "Been on hold for over an hour",
        "No response to my ticket in a week",
        "Support is way too slow",
        "Following up again on an open case",
    ],
    "fraud_report": [
        "Unauthorized transaction on my account",
        "I think my card was compromised",
        "Suspicious charges I did not make",
        "Report possible fraud on my account",
        "Someone used my account without permission",
    ],
    "account_access": [
        "Account frozen without explanation",
        "Need to verify my identity again",
        "Can't access funds in my account",
        "Account under review, need status update",
        "Request to unlock my account",
    ],
    "positive_feedback": [
        "Great experience with your app",
        "Thank you to your support team",
        "Really happy with the new features",
        "Excellent service this week",
        "Just wanted to say thanks",
    ],
    "general_inquiry": [
        "Question about account types",
        "How do I update my address",
        "Asking about interest rates",
        "Need details on a recent statement",
        "General question about your services",
    ],
}

BODY_PHRASES = {
    "failed_transaction": [
        "I tried to pay {merchant} for {amount} and the transaction was declined.",
        "The payment failed but the funds still show as pending on my account.",
        "This is the second time a {category} purchase has failed this month.",
        "I need this resolved quickly because the merchant is threatening to cancel my order.",
        "Please refund the pending hold from the failed charge.",
        "My card was declined even though I have sufficient balance.",
        "The transaction error code didn't explain what actually went wrong.",
    ],
    "login_issue": [
        "I've reset my password three times and still can't log in.",
        "The mobile app crashes every time I try to sign in.",
        "I never receive the verification code by text or email.",
        "It says my credentials are invalid even though I just reset them.",
        "This has been happening for {days} days now and it's frustrating.",
        "Can someone manually reset my login so I can access my account?",
    ],
    "fee_complaint": [
        "I was charged a {amount} fee I don't understand.",
        "Nobody told me the monthly maintenance fee was going up.",
        "This overdraft fee seems unfair given my account history.",
        "I want this fee reversed or a clear explanation of why it was charged.",
        "Your fee schedule isn't clearly disclosed anywhere in the app.",
        "This is the third fee I've been charged this year that surprised me.",
    ],
    "slow_support": [
        "I opened this ticket {days} days ago and still have no response.",
        "I've called twice and been on hold for over 40 minutes each time.",
        "Your chat support disconnected me before resolving my issue.",
        "I just want an update, even if the answer is 'still working on it'.",
        "This is taking far too long to resolve for a simple issue.",
        "Please escalate this, I've already followed up twice.",
    ],
    "fraud_report": [
        "There's a charge of {amount} at {merchant} that I never made.",
        "My card must have been compromised, please freeze my account immediately.",
        "I noticed several small unauthorized charges over the past {days} days.",
        "Please investigate this and issue a new card right away.",
        "I did not authorize this transaction and want it reversed.",
        "This looks like classic card-testing fraud, several tiny charges in a row.",
    ],
    "account_access": [
        "My account has been frozen and nobody has explained why.",
        "I submitted my ID for verification {days} days ago and heard nothing back.",
        "I can't withdraw or transfer any funds right now.",
        "This review process is taking way too long for a routine check.",
        "Please tell me what documents you still need from me.",
        "I rely on this account for daily expenses and need access restored.",
    ],
    "positive_feedback": [
        "The new budgeting feature in the app is genuinely useful.",
        "Your support agent resolved my issue in minutes, really appreciated it.",
        "Switching to this bank was one of the best financial decisions I've made.",
        "The app redesign is clean and much easier to navigate.",
        "Thank you for being transparent about the recent fee changes.",
        "Just wanted to recognize your team for great service this week.",
    ],
    "general_inquiry": [
        "Can you explain the difference between the Basic and Premium account tiers?",
        "How do I update my mailing address on file?",
        "What's the current interest rate on savings accounts?",
        "Where can I download a statement from last {days} days?",
        "Do you offer joint accounts for {category} purposes?",
        "Just have a quick question about how transfers are processed.",
    ],
}


def make_body(category: str) -> str:
    n_sentences = random.randint(2, 4)
    sentences = random.sample(BODY_PHRASES[category], k=min(n_sentences, len(BODY_PHRASES[category])))
    filled = []
    for s in sentences:
        filled.append(s.format(
            merchant=fake.company(),
            amount=f"${round(random.uniform(8, 650), 2)}",
            category=random.choice(MERCHANT_CATEGORIES),
            days=random.randint(2, 14),
        ))
    return " ".join(filled)


def generate_tickets(customers: pd.DataFrame) -> pd.DataFrame:
    customers_sorted = customers.sort_values("signup_date").reset_index(drop=True)
    signup_dates_sorted = pd.to_datetime(customers_sorted["signup_date"]).values.astype("datetime64[D]")
    cust_ids_sorted = customers_sorted["customer_id"].to_numpy()
    west_mask_sorted = (customers_sorted["region"] == "West").to_numpy()

    rows = []
    ticket_counter = 1

    def eligible_pool(date: datetime, west_only: bool = False):
        date_np = np.datetime64(date.date(), "D")
        eligible_n = int(np.searchsorted(signup_dates_sorted, date_np, side="right"))
        ids = cust_ids_sorted[:eligible_n]
        if west_only:
            ids = ids[west_mask_sorted[:eligible_n]]
        return ids

    # Baseline tickets spread across the whole window.
    n_baseline = N_TICKETS_TARGET - 220  # leave room for anomaly-correlated spikes below
    for _ in range(n_baseline):
        offset = int(rng.integers(0, TOTAL_DAYS))
        created = START_DATE + timedelta(days=offset, seconds=int(rng.integers(0, 86_399)))
        pool = eligible_pool(created)
        if len(pool) == 0:
            continue
        category = rng.choice(TICKET_CATEGORIES, p=TICKET_CATEGORY_WEIGHTS)
        customer_id = rng.choice(pool)
        subject = random.choice(SUBJECT_TEMPLATES[category]).format(
            merchant=fake.company(), category=random.choice(MERCHANT_CATEGORIES)
        )
        body = make_body(category)
        rows.append((customer_id, created, subject, body, category))

    # Extra fraud_report / failed_transaction tickets during the fraud week,
    # correlated with the injected West-region fraud spike in transactions.
    fraud_days = pd.date_range(FRAUD_START, FRAUD_END, freq="D")
    for day in fraud_days:
        pool = eligible_pool(day, west_only=True)
        if len(pool) == 0:
            continue
        n = int(rng.integers(8, 14))
        for _ in range(n):
            customer_id = rng.choice(pool)
            created = day + timedelta(seconds=int(rng.integers(0, 86_399)))
            category = rng.choice(["fraud_report", "failed_transaction"], p=[0.65, 0.35])
            subject = random.choice(SUBJECT_TEMPLATES[category]).format(
                merchant=fake.company(), category=random.choice(MERCHANT_CATEGORIES)
            )
            body = make_body(category)
            rows.append((customer_id, created, subject, body, category))

    # Extra failed_transaction / slow_support tickets around the outage day.
    outage_window = pd.date_range(OUTAGE_DATE, OUTAGE_DATE + timedelta(days=2), freq="D")
    for day in outage_window:
        pool = eligible_pool(day)
        if len(pool) == 0:
            continue
        n = int(rng.integers(15, 25))
        for _ in range(n):
            customer_id = rng.choice(pool)
            created = day + timedelta(seconds=int(rng.integers(0, 86_399)))
            category = rng.choice(["failed_transaction", "slow_support"], p=[0.7, 0.3])
            subject = random.choice(SUBJECT_TEMPLATES[category]).format(
                merchant=fake.company(), category=random.choice(MERCHANT_CATEGORIES)
            )
            body = make_body(category)
            rows.append((customer_id, created, subject, body, category))

    rows.sort(key=lambda r: r[1])
    ticket_ids = [f"TCKT{i:05d}" for i in range(1, len(rows) + 1)]
    df = pd.DataFrame(rows, columns=["customer_id", "created_at", "subject", "body_text", "category"])
    df.insert(0, "ticket_id", ticket_ids)
    df["created_at"] = df["created_at"].apply(lambda t: t.isoformat(sep=" "))
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"Generating synthetic data with seed={SEED} ...")

    customers = generate_customers()
    customers_out = customers.drop(columns=["_activity_weight"])
    customers_out.to_csv(f"{DATA_DIR}/customers.csv", index=False)
    print(f"  wrote {len(customers_out):,} rows -> {DATA_DIR}/customers.csv")

    transactions = generate_transactions(customers)
    transactions.to_csv(f"{DATA_DIR}/transactions.csv", index=False)
    print(f"  wrote {len(transactions):,} rows -> {DATA_DIR}/transactions.csv")

    tickets = generate_tickets(customers)
    tickets.to_csv(f"{DATA_DIR}/support_tickets.csv", index=False)
    print(f"  wrote {len(tickets):,} rows -> {DATA_DIR}/support_tickets.csv")

    print("\nInjected anomaly windows (for reference / README):")
    print(f"  fraud spike   : {FRAUD_START.date()} to {FRAUD_END.date()} (West region)")
    print(f"  outage        : {OUTAGE_DATE.date()} (system-wide volume drop)")
    print(f"  holiday surge : {HOLIDAY_START.date()} to {HOLIDAY_END.date()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
