-- Daily transaction KPIs by region and channel. This is the primary input
-- to the Python anomaly detection step (python/anomaly_detection/detect_anomalies.py),
-- which reads this mart straight out of DuckDB.

with fct as (
    select * from {{ ref('fct_transactions') }}
),

daily as (
    select
        transaction_date,
        region,
        channel,
        count(*) as txn_count,
        sum(amount) as total_amount,
        sum(amount) filter (where is_completed) as completed_amount,
        count(*) filter (where is_completed) as completed_count,
        count(*) filter (where is_declined) as declined_count,
        count(distinct customer_id) as active_customers,
        avg(amount) as avg_amount
    from fct
    group by transaction_date, region, channel
)

select * from daily
order by transaction_date, region, channel
