-- Customer dimension: one row per customer, enriched with lifetime
-- transaction activity so BI tools can slice by cohort/behavior without
-- re-joining the fact table every time.

with customers as (
    select * from {{ ref('stg_customers') }}
),

txn_stats as (
    select
        customer_id,
        count(*) as lifetime_txn_count,
        sum(amount) filter (where is_completed) as lifetime_completed_amount,
        min(transaction_date) as first_txn_date,
        max(transaction_date) as last_txn_date
    from {{ ref('stg_transactions') }}
    group by customer_id
),

ticket_stats as (
    select
        customer_id,
        count(*) as lifetime_ticket_count
    from {{ ref('stg_support_tickets') }}
    group by customer_id
),

final as (
    select
        c.customer_id,
        c.signup_date,
        c.region,
        c.account_type,
        c.risk_segment,
        date_diff('day', c.signup_date, current_date) as tenure_days,
        coalesce(t.lifetime_txn_count, 0) as lifetime_txn_count,
        coalesce(t.lifetime_completed_amount, 0) as lifetime_completed_amount,
        t.first_txn_date,
        t.last_txn_date,
        coalesce(k.lifetime_ticket_count, 0) as lifetime_ticket_count
    from customers c
    left join txn_stats t on c.customer_id = t.customer_id
    left join ticket_stats k on c.customer_id = k.customer_id
)

select * from final
