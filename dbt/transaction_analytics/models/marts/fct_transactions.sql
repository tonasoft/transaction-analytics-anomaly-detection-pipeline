-- Transaction fact table: one row per transaction, grain-preserving.
-- Denormalizes a couple of customer attributes (region, account_type) that
-- are almost always used to slice transaction volume, which keeps the daily
-- KPI aggregation below a single-table group-by instead of a join.

with transactions as (
    select * from {{ ref('stg_transactions') }}
),

customers as (
    select customer_id, region, account_type, risk_segment
    from {{ ref('stg_customers') }}
),

final as (
    select
        t.transaction_id,
        t.customer_id,
        c.region,
        c.account_type,
        c.risk_segment,
        t.transaction_ts,
        t.transaction_date,
        t.amount,
        t.merchant_category,
        t.channel,
        t.status,
        t.is_completed,
        t.is_declined
    from transactions t
    left join customers c on t.customer_id = c.customer_id
)

select * from final
