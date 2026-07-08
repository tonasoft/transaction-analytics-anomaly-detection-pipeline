-- Cleaned, typed view of raw customer records. One row per customer.

with source as (
    select * from {{ source('raw', 'customers') }}
),

renamed as (
    select
        customer_id,
        cast(signup_date as date) as signup_date,
        region,
        account_type,
        risk_segment
    from source
)

select * from renamed
