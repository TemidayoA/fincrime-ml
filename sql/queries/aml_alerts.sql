-- =============================================================================
-- FinCrime-ML · AML Alert Queue Queries
-- MySQL 8.0+ · MLRO operational and MI reporting queries
-- =============================================================================
--
-- Purpose
--   Operational queries for the MLRO alert work queue, SAR filing workflow,
--   alert fatigue monitoring, typology analysis, and management information.
--
-- Regulatory alignment
--   POCA 2002 s.330  — SAR filing; mandatory disclosure obligation
--   JMLSG Part I Ch.5— Transaction monitoring indicators and typology breakdown
--   FCA SYSC 6.3     — Alert queue management and documented review process
--   MLR 2017 Reg 19  — Staff oversight: alert volume vs. review capacity
--   PRA SS1/23       — Model performance monitoring; champion/challenger
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q1: MLRO priority work queue
--     All open alerts sorted by priority then risk score.
--     This is the primary view an MLRO loads at the start of each day.
-- ---------------------------------------------------------------------------
SELECT
    a.alert_id,
    a.transaction_id,
    a.priority,
    a.risk_score,
    a.risk_tier,
    a.n_triggers,
    a.trigger_reasons,
    a.sar_recommended,
    a.amount_gbp,
    a.typology,
    a.scored_at,
    TIMESTAMPDIFF(HOUR, a.scored_at, NOW())  AS hours_open,
    t.sender_account_id,
    t.receiver_account_id,
    t.channel,
    t.country_origin,
    t.country_destination
FROM aml_alerts   a
JOIN transactions t ON t.transaction_id = a.transaction_id
WHERE a.status = 'OPEN'
ORDER BY
    a.priority    ASC,
    a.risk_score  DESC,
    a.scored_at   ASC;


-- ---------------------------------------------------------------------------
-- Q2: Priority 1 alerts — immediate MLRO escalation (POCA 2002 s.330)
--     Must be reviewed within 24 hours per firm policy aligned to JMLSG.
--     Includes regulatory references for each alert.
-- ---------------------------------------------------------------------------
SELECT
    a.alert_id,
    a.transaction_id,
    a.risk_score,
    a.risk_tier,
    a.trigger_reasons,
    a.sar_recommended,
    a.regulatory_refs,
    a.mlro_summary,
    a.amount_gbp,
    a.typology,
    a.scored_at,
    TIMESTAMPDIFF(HOUR, a.scored_at, NOW())  AS hours_open,
    -- Flag alerts that have breached the 24-hour SLA
    CASE
        WHEN TIMESTAMPDIFF(HOUR, a.scored_at, NOW()) > 24
        THEN 1 ELSE 0
    END                                      AS sla_breached
FROM aml_alerts a
WHERE a.priority = 1
  AND a.status   = 'OPEN'
ORDER BY a.scored_at ASC;


-- ---------------------------------------------------------------------------
-- Q3: SAR filing queue — recommended alerts not yet referred to NCA
--     Supports the s.330 disclosure obligation workflow.
-- ---------------------------------------------------------------------------
SELECT
    a.alert_id,
    a.transaction_id,
    a.risk_score,
    a.risk_tier,
    a.n_triggers,
    a.trigger_reasons,
    a.regulatory_refs,
    a.mlro_summary,
    a.amount_gbp,
    a.typology,
    a.scored_at,
    TIMESTAMPDIFF(HOUR, a.scored_at, NOW())  AS hours_since_scoring,
    t.sender_account_id,
    t.receiver_account_id,
    t.channel,
    t.country_origin,
    t.country_destination,
    acc_s.risk_segment                       AS sender_risk_segment,
    acc_r.risk_segment                       AS receiver_risk_segment
FROM aml_alerts  a
JOIN transactions t  ON  t.transaction_id  = a.transaction_id
JOIN accounts    acc_s ON acc_s.account_id = t.sender_account_id
JOIN accounts    acc_r ON acc_r.account_id = t.receiver_account_id
LEFT JOIN sar_referrals r ON r.alert_id    = a.alert_id
WHERE a.sar_recommended = 1
  AND a.status IN ('OPEN', 'UNDER_REVIEW')
  AND r.referral_id IS NULL
ORDER BY a.priority ASC, a.scored_at ASC;


-- ---------------------------------------------------------------------------
-- Q4: Trigger frequency analysis — which rules fire most often?
--     Input to JMLSG para 5.3.1 monitoring system tuning review.
-- ---------------------------------------------------------------------------
SELECT
    trigger_name,
    COUNT(*)                                        AS alert_count,
    ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS pct_of_alerts,
    SUM(sar_recommended)                            AS sar_recommended_count,
    ROUND(AVG(risk_score), 4)                       AS mean_risk_score,
    ROUND(AVG(n_triggers), 2)                       AS mean_triggers_per_alert
FROM (
    -- Unnest the pipe-separated trigger_reasons into one row per trigger
    SELECT
        a.alert_id,
        a.sar_recommended,
        a.risk_score,
        a.n_triggers,
        TRIM(jt.trigger_name) AS trigger_name
    FROM aml_alerts a
    JOIN JSON_TABLE(
        CONCAT('["', REPLACE(a.trigger_reasons, '|', '","'), '"]'),
        '$[*]' COLUMNS (trigger_name VARCHAR(60) PATH '$')
    ) jt
    WHERE a.status != 'CLOSED'
      AND a.scored_at >= NOW() - INTERVAL 30 DAY
) triggers
GROUP BY trigger_name
ORDER BY alert_count DESC;


-- ---------------------------------------------------------------------------
-- Q5: Typology breakdown — alert volume and SAR rate by AML typology
--     Used for JMLSG para 5.3 typology calibration reports.
-- ---------------------------------------------------------------------------
SELECT
    COALESCE(a.typology, 'unknown')         AS typology,
    COUNT(*)                                AS total_alerts,
    SUM(a.priority = 1)                     AS p1_critical,
    SUM(a.priority = 2)                     AS p2_high,
    SUM(a.priority = 3)                     AS p3_medium,
    SUM(a.sar_recommended = 1)              AS sar_recommended,
    ROUND(SUM(a.sar_recommended) / COUNT(*) * 100, 2)  AS sar_rate_pct,
    ROUND(AVG(a.risk_score), 4)             AS mean_risk_score,
    ROUND(AVG(a.amount_gbp), 2)             AS mean_amount_gbp,
    SUM(a.amount_gbp)                       AS total_amount_gbp
FROM aml_alerts a
WHERE a.scored_at >= NOW() - INTERVAL 30 DAY
GROUP BY typology
ORDER BY total_alerts DESC;


-- ---------------------------------------------------------------------------
-- Q6: Alert fatigue metrics — FPR proxy and analyst workload
--     FCA SYSC 6.3: monitoring system effectiveness review.
--     Uses closed alerts with known true/false positive outcome.
-- ---------------------------------------------------------------------------
SELECT
    DATE(a.scored_at)                               AS review_date,
    COUNT(*)                                        AS total_alerts,
    SUM(a.sar_recommended = 1)                      AS sar_recommended,

    -- Outcome split (requires closed alerts with SAR filing outcome)
    SUM(CASE WHEN a.status = 'SAR_FILED'    THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN a.status = 'CLOSED'
              AND a.sar_recommended = 0     THEN 1 ELSE 0 END) AS true_negatives_est,
    SUM(CASE WHEN a.status = 'CLOSED'
              AND a.sar_recommended = 1     THEN 1 ELSE 0 END) AS false_positives_est,

    -- Fatigue index approximation: closed-as-FP / total alerted
    ROUND(
        SUM(CASE WHEN a.status = 'CLOSED' AND a.sar_recommended = 1 THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0) * 100,
        2
    )                                               AS fatigue_index_pct,

    -- Alert queue throughput (analyst capacity indicator per MLR 2017 Reg 19)
    SUM(CASE WHEN a.reviewed_at IS NOT NULL THEN 1 ELSE 0 END) AS alerts_reviewed,
    ROUND(
        AVG(CASE WHEN a.reviewed_at IS NOT NULL
            THEN TIMESTAMPDIFF(MINUTE, a.scored_at, a.reviewed_at) END),
        0
    )                                               AS avg_review_time_minutes

FROM aml_alerts a
WHERE a.scored_at >= NOW() - INTERVAL 30 DAY
GROUP BY DATE(a.scored_at)
ORDER BY review_date DESC;


-- ---------------------------------------------------------------------------
-- Q7: Mule account alert concentration
--     High mule-linked transaction volumes trigger enhanced monitoring
--     under MLR 2017 Reg 28 (Enhanced Due Diligence).
-- ---------------------------------------------------------------------------
SELECT
    CASE
        WHEN t.is_mule_sender   = 1 AND t.is_mule_receiver = 1 THEN 'BOTH'
        WHEN t.is_mule_sender   = 1                             THEN 'SENDER'
        WHEN t.is_mule_receiver = 1                             THEN 'RECEIVER'
        ELSE                                                         'NONE'
    END                                         AS mule_involvement_type,
    COUNT(a.alert_id)                           AS alert_count,
    SUM(a.sar_recommended)                      AS sar_recommended,
    ROUND(AVG(a.risk_score), 4)                 AS mean_risk_score,
    ROUND(AVG(a.n_triggers), 2)                 AS mean_triggers,
    SUM(a.amount_gbp)                           AS total_amount_gbp
FROM aml_alerts   a
JOIN transactions t ON t.transaction_id = a.transaction_id
WHERE a.scored_at >= NOW() - INTERVAL 30 DAY
GROUP BY mule_involvement_type
ORDER BY alert_count DESC;


-- ---------------------------------------------------------------------------
-- Q8: Structuring pattern — accounts with multiple POCA s.330 band transactions
--     Groups by sender; flags accounts with >= 2 structuring alerts in 7 days.
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id,
    COUNT(a.alert_id)                           AS structuring_alerts_7d,
    SUM(a.amount_gbp)                           AS total_structured_amount,
    MIN(a.amount_gbp)                           AS min_amount,
    MAX(a.amount_gbp)                           AS max_amount,
    MIN(a.scored_at)                            AS first_alert_at,
    MAX(a.scored_at)                            AS latest_alert_at,
    GROUP_CONCAT(a.alert_id ORDER BY a.scored_at SEPARATOR ' | ') AS alert_ids
FROM aml_alerts   a
JOIN transactions t ON t.transaction_id = a.transaction_id
WHERE a.typology    IN ('structuring')
   OR a.trigger_reasons LIKE '%STRUCTURING_AMOUNT%'
AND a.scored_at >= NOW() - INTERVAL 7 DAY
GROUP BY t.sender_account_id
HAVING structuring_alerts_7d >= 2
ORDER BY structuring_alerts_7d DESC, total_structured_amount DESC;


-- ---------------------------------------------------------------------------
-- Q9: High-risk account watchlist — accounts generating >= 3 alerts in 30 days
--     Feeds enhanced monitoring and EDD review queue (MLR 2017 Reg 28).
-- ---------------------------------------------------------------------------
SELECT
    t.sender_account_id                         AS account_id,
    acc.risk_segment,
    acc.is_mule_flagged,
    COUNT(DISTINCT a.alert_id)                  AS alert_count_30d,
    SUM(a.sar_recommended)                      AS sar_recommended_count,
    MAX(a.priority)                             AS highest_priority,
    ROUND(MAX(a.risk_score), 4)                 AS peak_risk_score,
    SUM(a.amount_gbp)                           AS total_alerted_amount_gbp,
    MAX(a.scored_at)                            AS latest_alert_at
FROM aml_alerts   a
JOIN transactions t   ON  t.transaction_id  = a.transaction_id
JOIN accounts     acc ON  acc.account_id    = t.sender_account_id
WHERE a.scored_at >= NOW() - INTERVAL 30 DAY
  AND a.status    != 'CLOSED'
GROUP BY t.sender_account_id, acc.risk_segment, acc.is_mule_flagged
HAVING alert_count_30d >= 3
ORDER BY sar_recommended_count DESC, alert_count_30d DESC;


-- ---------------------------------------------------------------------------
-- Q10: Daily MI summary — management information for compliance committee
--      JMLSG Part I para 5.3.1: monitoring system tuning and oversight
-- ---------------------------------------------------------------------------
SELECT
    DATE(a.scored_at)                               AS report_date,
    COUNT(*)                                        AS total_alerts,
    SUM(a.priority = 1)                             AS p1_critical,
    SUM(a.priority = 2)                             AS p2_high,
    SUM(a.priority = 3)                             AS p3_medium,
    SUM(a.sar_recommended = 1)                      AS sar_recommended,
    ROUND(SUM(a.sar_recommended) / COUNT(*) * 100, 2) AS sar_rate_pct,
    ROUND(AVG(a.risk_score), 4)                     AS mean_risk_score,
    ROUND(AVG(a.n_triggers), 2)                     AS mean_triggers,
    SUM(a.amount_gbp)                               AS total_amount_in_alerts,
    COUNT(DISTINCT t.sender_account_id)             AS unique_accounts_alerted,
    SUM(CASE WHEN a.status = 'SAR_FILED'   THEN 1 ELSE 0 END) AS sars_filed,
    SUM(CASE WHEN a.reviewed_at IS NOT NULL THEN 1 ELSE 0 END) AS alerts_reviewed
FROM aml_alerts   a
JOIN transactions t ON t.transaction_id = a.transaction_id
WHERE a.scored_at >= NOW() - INTERVAL 30 DAY
GROUP BY DATE(a.scored_at)
ORDER BY report_date DESC;
