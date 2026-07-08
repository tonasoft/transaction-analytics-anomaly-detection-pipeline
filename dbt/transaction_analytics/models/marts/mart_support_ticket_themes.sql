-- Daily support ticket volume by category (the label assigned at ticket
-- creation time). This is the SQL-modeled view of ticket themes; the deeper
-- unsupervised theme clustering + sentiment scoring lives in
-- python/nlp/analyze_tickets.py and is exported separately to
-- outputs/ticket_themes.csv since TF-IDF/KMeans isn't practical to express
-- in SQL. Both are surfaced on the dashboard.

with tickets as (
    select * from {{ ref('stg_support_tickets') }}
),

daily as (
    select
        created_date,
        category,
        count(*) as ticket_count,
        count(distinct customer_id) as distinct_customers
    from tickets
    group by created_date, category
)

select * from daily
order by created_date, category
