-- Cleaned, typed view of raw support tickets. One row per ticket.

with source as (
    select * from {{ source('raw', 'support_tickets') }}
),

renamed as (
    select
        ticket_id,
        customer_id,
        cast(created_at as timestamp) as created_at,
        cast(cast(created_at as timestamp) as date) as created_date,
        subject,
        body_text,
        category
    from source
)

select * from renamed
