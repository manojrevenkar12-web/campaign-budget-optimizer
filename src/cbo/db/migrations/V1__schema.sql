-- Tenancy: every advertiser's data is scoped by tenant; API keys are stored hashed.
create table tenants (
    id text primary key,
    name text not null,
    created_at timestamptz not null default now()
);

create table api_keys (
    key_hash text primary key,
    tenant_id text not null references tenants(id),
    label text not null default '',
    created_at timestamptz not null default now(),
    revoked_at timestamptz
);

create table campaigns (
    id text primary key,
    tenant_id text not null references tenants(id),
    name text not null default '',
    daily_budget numeric(12, 2) not null check (daily_budget > 0),
    currency text not null default 'EUR',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index campaigns_tenant on campaigns (tenant_id);

create table ad_sets (
    id text primary key,
    campaign_id text not null references campaigns(id) on delete cascade,
    platform text not null check (platform in ('meta', 'tiktok', 'pinterest', 'google')),
    name text not null default '',
    active boolean not null default true
);
create index ad_sets_campaign on ad_sets (campaign_id);

create table daily_stats (
    ad_set_id text not null references ad_sets(id) on delete cascade,
    day date not null,
    spend numeric(12, 2) not null check (spend >= 0),
    impressions bigint not null check (impressions >= 0),
    clicks bigint not null check (clicks >= 0 and clicks <= impressions),
    conversions integer not null check (conversions >= 0),
    ingested_at timestamptz not null default now(),
    primary key (ad_set_id, day)
);

create table creative_daily (
    creative_id text not null,
    ad_set_id text not null references ad_sets(id) on delete cascade,
    day date not null,
    creative_format text not null check (creative_format in ('static', 'carousel', 'video')),
    creative_age integer not null check (creative_age >= 0),
    frequency double precision not null check (frequency >= 0),
    impressions bigint not null check (impressions >= 0),
    clicks bigint not null check (clicks >= 0),
    primary key (creative_id, day)
);
create index creative_daily_ad_set on creative_daily (ad_set_id, day);

-- Model state written by the worker, read by the API on every request.
create table posteriors (
    ad_set_id text primary key references ad_sets(id) on delete cascade,
    state jsonb not null,
    fitted_through date,
    updated_at timestamptz not null default now()
);

create table fatigue_scores (
    creative_id text primary key,
    ad_set_id text not null references ad_sets(id) on delete cascade,
    day date not null,
    probability double precision not null check (probability between 0 and 1),
    state text not null check (state in ('at_risk_scored', 'already_fatigued')),
    model_version text not null,
    scored_at timestamptz not null default now()
);

-- Every recommendation is stored with what it was made from, so it can be explained later.
create table recommendations (
    id uuid primary key,
    campaign_id text not null references campaigns(id) on delete cascade,
    day date not null,
    policy text not null,
    budget numeric(12, 2) not null,
    allocation jsonb not null,
    model_state_as_of timestamptz,
    idempotency_key text,
    created_at timestamptz not null default now(),
    unique (campaign_id, idempotency_key)
);
create index recommendations_campaign on recommendations (campaign_id, created_at desc);

-- A small job queue: workers claim jobs with FOR UPDATE SKIP LOCKED.
create table jobs (
    id bigserial primary key,
    kind text not null check (kind in ('refresh_campaign')),
    campaign_id text not null references campaigns(id) on delete cascade,
    status text not null default 'queued' check (status in ('queued', 'running', 'done', 'dead')),
    run_after timestamptz not null default now(),
    locked_until timestamptz,
    attempts integer not null default 0,
    max_attempts integer not null default 5,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
-- at most one queued refresh per campaign: new stats just extend the pending job
create unique index jobs_one_queued on jobs (kind, campaign_id) where status = 'queued';
create index jobs_claim on jobs (status, run_after);
