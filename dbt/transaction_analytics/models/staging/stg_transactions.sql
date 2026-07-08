-- Cleaned, typed view of raw transactions. One row per transaction.

with source as (
    select * from {{ source('raw', 'transactions') }}
),

renamed as (
    select
        transaction_id,
        customer_id,
        cast(timestamp as timestamp) as transaction_ts,
        cast(cast(timestamp as timestamp) as date) as transaction_date,
        cast(amount as decimal(12, 2)) as amount,
        merchant_category,
        channel,
        status,
        status = 'completed' as is_completed,
        status = 'declined' as is_declined
    from source
)

select * from renamed
