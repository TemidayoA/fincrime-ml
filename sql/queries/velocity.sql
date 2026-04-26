-- =============================================================================
-- FinCrime-ML · Velocity Window Queries
-- MySQL 8.0+ · Used by transaction monitoring feature engineering
-- =============================================================================
--
-- Purpose
--   Velocity features are the primary signal for fraud detection and AML
--   structuring detection. These queries compute rolling-window aggregates
--   over configurable time horizons, mapped to the indicators in:
--
--   JMLSG Part I para 5.3.7  — Unusual transaction velocity
--   JMLSG Part I para 5.3.11 — Structuring: multiple sub-threshold transactions
--   FATF R.10                 — Customer due diligence on unusual patterns
--   FCA FCG 3.2               — Transaction monitoring: volume and frequency
--
-- Usage
--   Queries use a :account_id parameter and :as_of_datetime anchor.
--   In Python: pass via SQLAlchemy text() with bindparams.
--   Indexed on (sender_account_id, transacted_at) for sub-second response.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q1: Transaction count by window — sender velocity
--     Detects sudden spikes in outbound activity (JMLSG para 5.3.7)
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id,

    -- 1-hour window
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 1 HOUR  THEN 1 END)  AS txn_count_1h,
    -- 6-hour window
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 6 HOUR  THEN 1 END)  AS txn_count_6h,
    -- 24-hour window
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN 1 END)  AS txn_count_24h,
    -- 7-day window
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY   THEN 1 END)  AS txn_count_7d,

    -- Baseline: prior 30 days for z-score computation
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 30 DAY  THEN 1 END)  AS txn_count_30d

FROM transactions t
WHERE t.sender_account_id  = :account_id
  AND t.transacted_at      < :as_of_datetime
  AND t.transacted_at     >= :as_of_datetime - INTERVAL 30 DAY
GROUP BY t.sender_account_id;


-- ---------------------------------------------------------------------------
-- Q2: Amount aggregates by window — outbound value velocity
--     High cumulative amounts relative to account history → layering signal
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id,

    -- Rolling sum
    SUM(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 1 HOUR  THEN t.amount_gbp ELSE 0 END)  AS amount_sum_1h,
    SUM(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 6 HOUR  THEN t.amount_gbp ELSE 0 END)  AS amount_sum_6h,
    SUM(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN t.amount_gbp ELSE 0 END)  AS amount_sum_24h,
    SUM(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY   THEN t.amount_gbp ELSE 0 END)  AS amount_sum_7d,

    -- Rolling mean (for deviation scoring)
    AVG(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 30 DAY THEN t.amount_gbp END)  AS amount_mean_30d,
    STDDEV(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 30 DAY THEN t.amount_gbp END) AS amount_std_30d,

    -- Current transaction deviation from 30-day mean
    ROUND(
        (:current_amount - AVG(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 30 DAY THEN t.amount_gbp END))
        / NULLIF(STDDEV(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 30 DAY THEN t.amount_gbp END), 0),
        4
    ) AS amount_z_score_30d

FROM transactions t
WHERE t.sender_account_id  = :account_id
  AND t.transacted_at      < :as_of_datetime
  AND t.transacted_at     >= :as_of_datetime - INTERVAL 30 DAY
GROUP BY t.sender_account_id;


-- ---------------------------------------------------------------------------
-- Q3: Structuring detection — sub-threshold clustering
--     POCA 2002 s.330: multiple transactions just below GBP 10,000
--     JMLSG Part I para 5.3.11
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id,

    -- Count of transactions in the POCA structuring avoidance band
    COUNT(CASE
        WHEN t.amount_gbp BETWEEN 8500.00 AND 9950.00
         AND t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY
        THEN 1
    END) AS structuring_count_7d,

    -- Total value of sub-threshold transactions in 7 days
    SUM(CASE
        WHEN t.amount_gbp BETWEEN 8500.00 AND 9950.00
         AND t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY
        THEN t.amount_gbp ELSE 0
    END) AS structuring_amount_7d,

    -- Rolling 24h sub-threshold count (acute structuring)
    COUNT(CASE
        WHEN t.amount_gbp BETWEEN 8500.00 AND 9950.00
         AND t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR
        THEN 1
    END) AS structuring_count_24h,

    -- Flag: 2+ structuring transactions in 7 days
    CASE
        WHEN COUNT(CASE
            WHEN t.amount_gbp BETWEEN 8500.00 AND 9950.00
             AND t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY
            THEN 1 END) >= 2
        THEN 1 ELSE 0
    END AS structuring_pattern_flag

FROM transactions t
WHERE t.sender_account_id  = :account_id
  AND t.transacted_at      < :as_of_datetime
  AND t.transacted_at     >= :as_of_datetime - INTERVAL 30 DAY
GROUP BY t.sender_account_id;


-- ---------------------------------------------------------------------------
-- Q4: Rapid movement detection — receive-then-send within 2 hours
--     FATF R.10 layering indicator; funds transit an account rapidly
-- ---------------------------------------------------------------------------
SELECT
    send_side.sender_account_id          AS account_id,
    COUNT(*)                             AS rapid_movement_count,
    MIN(send_side.transacted_at)         AS earliest_send,
    MAX(recv_side.transacted_at)         AS latest_receive,
    SUM(send_side.amount_gbp)            AS total_sent_gbp,
    SUM(recv_side.amount_gbp)            AS total_received_gbp,
    ROUND(
        SUM(send_side.amount_gbp) / NULLIF(SUM(recv_side.amount_gbp), 0),
        4
    )                                    AS pass_through_ratio
FROM transactions send_side
JOIN transactions recv_side
    ON  recv_side.receiver_account_id = send_side.sender_account_id
    AND recv_side.transacted_at       BETWEEN send_side.transacted_at - INTERVAL 2 HOUR
                                          AND send_side.transacted_at
WHERE send_side.sender_account_id = :account_id
  AND send_side.transacted_at    >= :as_of_datetime - INTERVAL 7 DAY
  AND send_side.transacted_at     < :as_of_datetime
GROUP BY send_side.sender_account_id
HAVING rapid_movement_count >= 1;


-- ---------------------------------------------------------------------------
-- Q5: Cross-border velocity — unusual international transaction frequency
--     FATF R.16: cross-border wire transfer monitoring
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id,
    COUNT(*)                                                        AS cross_border_count_7d,
    COUNT(DISTINCT t.country_destination)                           AS distinct_dest_countries_7d,
    SUM(t.amount_gbp)                                               AS cross_border_amount_7d,
    GROUP_CONCAT(DISTINCT t.country_destination ORDER BY t.country_destination SEPARATOR ',')
                                                                    AS destination_countries
FROM transactions t
WHERE t.sender_account_id = :account_id
  AND t.is_cross_border   = 1
  AND t.transacted_at    >= :as_of_datetime - INTERVAL 7 DAY
  AND t.transacted_at     < :as_of_datetime
GROUP BY t.sender_account_id;


-- ---------------------------------------------------------------------------
-- Q6: Unique counterparty count — fan-out / fan-in detection
--     Accounts transacting with many unique receivers = potential money mule hub
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id,

    -- Unique receivers in each window
    COUNT(DISTINCT CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN t.receiver_account_id END) AS unique_receivers_24h,
    COUNT(DISTINCT CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY   THEN t.receiver_account_id END) AS unique_receivers_7d,
    COUNT(DISTINCT CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 30 DAY  THEN t.receiver_account_id END) AS unique_receivers_30d,

    -- Fan-out flag: >5 unique receivers in 24 hours
    CASE WHEN COUNT(DISTINCT CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN t.receiver_account_id END) > 5
         THEN 1 ELSE 0 END                                              AS fan_out_flag_24h

FROM transactions t
WHERE t.sender_account_id = :account_id
  AND t.transacted_at    >= :as_of_datetime - INTERVAL 30 DAY
  AND t.transacted_at     < :as_of_datetime
GROUP BY t.sender_account_id;


-- ---------------------------------------------------------------------------
-- Q7: Composite velocity feature vector — all windows in one pass
--     Use for batch feature extraction; the Python pipeline calls this with
--     a derived table or CTE for a cohort of accounts.
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id                                             AS account_id,

    -- Count velocity
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 1 HOUR  THEN 1 END)  AS txn_count_1h,
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN 1 END)  AS txn_count_24h,
    COUNT(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY   THEN 1 END)  AS txn_count_7d,

    -- Amount velocity
    COALESCE(SUM(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN t.amount_gbp END), 0) AS amount_sum_24h,
    COALESCE(SUM(CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY   THEN t.amount_gbp END), 0) AS amount_sum_7d,
    COALESCE(AVG(t.amount_gbp), 0)                                  AS amount_mean_30d,

    -- Counterparty diversity
    COUNT(DISTINCT CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 24 HOUR THEN t.receiver_account_id END) AS unique_receivers_24h,
    COUNT(DISTINCT CASE WHEN t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY   THEN t.receiver_account_id END) AS unique_receivers_7d,

    -- Structuring signal
    SUM(CASE WHEN t.amount_gbp BETWEEN 8500.00 AND 9950.00
             AND t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY
         THEN 1 ELSE 0 END)                                         AS structuring_count_7d,

    -- Mule-linked transaction counts
    SUM(CASE WHEN t.is_mule_receiver = 1 THEN 1 ELSE 0 END)        AS mule_receiver_count_30d,
    SUM(CASE WHEN t.is_mule_sender   = 1 THEN 1 ELSE 0 END)        AS mule_sender_count_30d,

    -- Cross-border
    SUM(CASE WHEN t.is_cross_border = 1
             AND t.transacted_at >= :as_of_datetime - INTERVAL 7 DAY
         THEN 1 ELSE 0 END)                                         AS cross_border_count_7d

FROM transactions t
WHERE t.sender_account_id = :account_id
  AND t.transacted_at    >= :as_of_datetime - INTERVAL 30 DAY
  AND t.transacted_at     < :as_of_datetime
GROUP BY t.sender_account_id;
