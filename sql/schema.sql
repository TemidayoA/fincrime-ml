-- =============================================================================
-- FinCrime-ML · Transaction Monitoring Schema
-- MySQL 8.0+ compatible · InnoDB · UTF-8mb4
-- =============================================================================
--
-- Purpose
--   Persistent store for the FinCrime-ML dual-domain fraud + AML monitoring
--   pipeline. Captures raw transactions, model scores, SAR alert queue,
--   MLRO referrals, and the FCA SYSC 10A audit trail.
--
-- Regulatory alignment
--   FCA SYSC 6.3   — Transaction monitoring record retention (5 years minimum)
--   FCA SYSC 10A   — Automated decision audit trail requirements
--   POCA 2002 s.330— SAR filing obligation; sar_referrals table supports this
--   MLR 2017 Reg 40— Record-keeping: 5 years from end of business relationship
--   JMLSG Part I Ch.5 — Transaction monitoring indicators
--
-- Conventions
--   - All monetary amounts stored as DECIMAL(15,2) in originating currency
--   - All risk scores stored as DECIMAL(6,4) (0.0000 to 1.0000)
--   - All timestamps stored as DATETIME, application layer normalises to UTC
--   - IDs from Python pipeline are VARCHAR(50); internal PKs are BIGINT AUTO_INCREMENT
--   - Soft-delete pattern: deleted_at IS NULL = active record
-- =============================================================================

SET NAMES utf8mb4;
SET time_zone = '+00:00';

-- ---------------------------------------------------------------------------
-- accounts
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS accounts (
    account_id          VARCHAR(50)     NOT NULL,
    account_type        VARCHAR(30)     NOT NULL DEFAULT 'PERSONAL',   -- PERSONAL | BUSINESS | MULE_FLAGGED
    country_code        CHAR(2)         NOT NULL DEFAULT 'GB',
    risk_segment        VARCHAR(20)     NOT NULL DEFAULT 'STANDARD',   -- STANDARD | HIGH_RISK | PEP | SANCTIONED
    is_mule_flagged     TINYINT(1)      NOT NULL DEFAULT 0,
    mule_flagged_at     DATETIME                 DEFAULT NULL,
    onboarded_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at          DATETIME                 DEFAULT NULL,

    PRIMARY KEY (account_id),
    INDEX idx_accounts_mule       (is_mule_flagged),
    INDEX idx_accounts_risk       (risk_segment),
    INDEX idx_accounts_country    (country_code)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Account master. Mule flags updated by AML pipeline in real time.';


-- ---------------------------------------------------------------------------
-- transactions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    id                      BIGINT          NOT NULL AUTO_INCREMENT,
    transaction_id          VARCHAR(50)     NOT NULL,
    sender_account_id       VARCHAR(50)     NOT NULL,
    receiver_account_id     VARCHAR(50)     NOT NULL,
    amount_gbp              DECIMAL(15,2)   NOT NULL,
    currency                CHAR(3)         NOT NULL DEFAULT 'GBP',
    fx_rate_to_gbp          DECIMAL(12,6)   NOT NULL DEFAULT 1.000000,
    channel                 VARCHAR(30)     NOT NULL,   -- MOBILE_APP | CNP_ECOM | SWIFT | ATM
    transaction_type        VARCHAR(30)     NOT NULL,   -- TRANSFER | PAYMENT | CASH_OUT | CASH_IN
    country_origin          CHAR(2)         NOT NULL DEFAULT 'GB',
    country_destination     CHAR(2)         NOT NULL DEFAULT 'GB',
    is_cross_border         TINYINT(1)      NOT NULL DEFAULT 0,
    hour_of_day             TINYINT         NOT NULL,
    day_of_week             TINYINT         NOT NULL,   -- 0=Mon, 6=Sun
    typology                VARCHAR(30)     NOT NULL DEFAULT 'normal',
    structuring_flag        TINYINT(1)      NOT NULL DEFAULT 0,
    rapid_movement_flag     TINYINT(1)      NOT NULL DEFAULT 0,
    layering_depth          TINYINT         NOT NULL DEFAULT 0,
    is_mule_sender          TINYINT(1)      NOT NULL DEFAULT 0,
    is_mule_receiver        TINYINT(1)      NOT NULL DEFAULT 0,
    transacted_at           DATETIME        NOT NULL,
    created_at              DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_transaction_id    (transaction_id),
    INDEX idx_txn_sender            (sender_account_id, transacted_at),
    INDEX idx_txn_receiver          (receiver_account_id, transacted_at),
    INDEX idx_txn_transacted_at     (transacted_at),
    INDEX idx_txn_amount            (amount_gbp),
    INDEX idx_txn_typology          (typology),
    INDEX idx_txn_structuring       (structuring_flag, amount_gbp),
    INDEX idx_txn_mule              (is_mule_sender, is_mule_receiver),
    CONSTRAINT fk_txn_sender   FOREIGN KEY (sender_account_id)   REFERENCES accounts (account_id),
    CONSTRAINT fk_txn_receiver FOREIGN KEY (receiver_account_id) REFERENCES accounts (account_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Core transaction ledger. Indexed for velocity and typology queries.';


-- ---------------------------------------------------------------------------
-- model_versions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    id              BIGINT          NOT NULL AUTO_INCREMENT,
    model_name      VARCHAR(60)     NOT NULL,   -- AMLIsolationForest | GraphScorer | XGBClassifier | FinCrimeScorer
    version         VARCHAR(20)     NOT NULL,
    domain          VARCHAR(10)     NOT NULL,   -- fraud | aml | core
    strategy        VARCHAR(40)              DEFAULT NULL,  -- fusion strategy, if applicable
    trained_at      DATETIME                 DEFAULT NULL,
    deployed_at     DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active       TINYINT(1)      NOT NULL DEFAULT 1,
    auc_pr          DECIMAL(6,4)             DEFAULT NULL,
    roc_auc         DECIMAL(6,4)             DEFAULT NULL,
    notes           TEXT                     DEFAULT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_model_version (model_name, version),
    INDEX idx_model_active (is_active, model_name)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Model registry. Supports champion/challenger versioning per PRA SS1/23.';


-- ---------------------------------------------------------------------------
-- fraud_scores
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fraud_scores (
    id              BIGINT          NOT NULL AUTO_INCREMENT,
    transaction_id  VARCHAR(50)     NOT NULL,
    model_version   VARCHAR(20)     NOT NULL DEFAULT '0.1.0',
    fraud_score     DECIMAL(6,4)    NOT NULL,
    risk_tier       VARCHAR(10)     NOT NULL,  -- LOW | MEDIUM | HIGH | CRITICAL
    top_reason_1    VARCHAR(60)              DEFAULT NULL,
    top_reason_2    VARCHAR(60)              DEFAULT NULL,
    top_reason_3    VARCHAR(60)              DEFAULT NULL,
    shap_json       JSON                     DEFAULT NULL,  -- full SHAP vector
    scored_at       DATETIME        NOT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_fraud_score_txn (transaction_id, model_version),
    INDEX idx_fraud_score       (fraud_score),
    INDEX idx_fraud_tier        (risk_tier),
    INDEX idx_fraud_scored_at   (scored_at),
    CONSTRAINT fk_fraud_txn FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Fraud model output per transaction. One row per transaction per model version.';


-- ---------------------------------------------------------------------------
-- aml_scores
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_scores (
    id              BIGINT          NOT NULL AUTO_INCREMENT,
    transaction_id  VARCHAR(50)     NOT NULL,
    model_name      VARCHAR(60)     NOT NULL DEFAULT 'AMLIsolationForest',
    model_version   VARCHAR(20)     NOT NULL DEFAULT '0.1.0',
    aml_score       DECIMAL(6,4)    NOT NULL,
    risk_tier       VARCHAR(10)     NOT NULL,
    top_reason_1    VARCHAR(60)              DEFAULT NULL,
    top_reason_2    VARCHAR(60)              DEFAULT NULL,
    top_reason_3    VARCHAR(60)              DEFAULT NULL,
    shap_json       JSON                     DEFAULT NULL,
    scored_at       DATETIME        NOT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_aml_score_txn (transaction_id, model_name, model_version),
    INDEX idx_aml_score       (aml_score),
    INDEX idx_aml_tier        (risk_tier),
    INDEX idx_aml_scored_at   (scored_at),
    CONSTRAINT fk_aml_txn FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='AML model output per transaction. Supports both supervised and unsupervised models.';


-- ---------------------------------------------------------------------------
-- unified_scores
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS unified_scores (
    id                  BIGINT          NOT NULL AUTO_INCREMENT,
    transaction_id      VARCHAR(50)     NOT NULL,
    fraud_score         DECIMAL(6,4)             DEFAULT NULL,
    aml_score           DECIMAL(6,4)             DEFAULT NULL,
    unified_risk_score  DECIMAL(6,4)    NOT NULL,
    risk_tier           VARCHAR(10)     NOT NULL,
    fusion_strategy     VARCHAR(30)     NOT NULL DEFAULT 'weighted_average',
    fraud_weight        DECIMAL(4,2)             DEFAULT NULL,
    aml_weight          DECIMAL(4,2)             DEFAULT NULL,
    model_version       VARCHAR(20)     NOT NULL DEFAULT '0.1.0',
    scored_at           DATETIME        NOT NULL,
    created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_unified_txn (transaction_id, model_version),
    INDEX idx_unified_score     (unified_risk_score),
    INDEX idx_unified_tier      (risk_tier),
    INDEX idx_unified_scored_at (scored_at),
    CONSTRAINT fk_unified_txn FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='FinCrimeScorer fusion output. Combines fraud and AML signals per FCA SYSC 6.3.';


-- ---------------------------------------------------------------------------
-- aml_alerts   (SAR trigger scorer output)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aml_alerts (
    id                  BIGINT          NOT NULL AUTO_INCREMENT,
    alert_id            VARCHAR(50)     NOT NULL,
    transaction_id      VARCHAR(50)     NOT NULL,
    risk_score          DECIMAL(6,4)    NOT NULL,
    risk_tier           VARCHAR(10)     NOT NULL,
    priority            TINYINT         NOT NULL,   -- 1=CRITICAL, 2=HIGH, 3=MEDIUM
    n_triggers          TINYINT         NOT NULL DEFAULT 0,
    trigger_reasons     VARCHAR(255)    NOT NULL,   -- pipe-separated rule names
    sar_recommended     TINYINT(1)      NOT NULL DEFAULT 0,
    regulatory_refs     TEXT                     DEFAULT NULL,
    mlro_summary        TEXT                     DEFAULT NULL,
    amount_gbp          DECIMAL(15,2)            DEFAULT NULL,
    typology            VARCHAR(30)              DEFAULT NULL,
    -- Workflow status
    status              VARCHAR(20)     NOT NULL DEFAULT 'OPEN',  -- OPEN | UNDER_REVIEW | CLOSED | SAR_FILED
    assigned_to         VARCHAR(100)             DEFAULT NULL,    -- MLRO / analyst username
    reviewed_at         DATETIME                 DEFAULT NULL,
    review_notes        TEXT                     DEFAULT NULL,
    scored_at           DATETIME        NOT NULL,
    created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_alert_id          (alert_id),
    INDEX idx_alert_priority        (priority, risk_score DESC),
    INDEX idx_alert_status          (status, priority),
    INDEX idx_alert_sar             (sar_recommended, status),
    INDEX idx_alert_txn             (transaction_id),
    INDEX idx_alert_scored_at       (scored_at),
    INDEX idx_alert_typology        (typology),
    CONSTRAINT fk_alert_txn FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='SAR alert queue. Priority 1 alerts require immediate MLRO referral (POCA 2002 s.330).';


-- ---------------------------------------------------------------------------
-- sar_referrals
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sar_referrals (
    id                  BIGINT          NOT NULL AUTO_INCREMENT,
    referral_id         VARCHAR(50)     NOT NULL,
    alert_id            VARCHAR(50)     NOT NULL,
    transaction_id      VARCHAR(50)     NOT NULL,
    filed_by            VARCHAR(100)    NOT NULL,   -- MLRO username
    sar_reference       VARCHAR(100)             DEFAULT NULL,  -- NCA reference number post-filing
    filing_basis        VARCHAR(255)    NOT NULL,   -- POCA 2002 s.330 | s.331 | s.332
    filing_status       VARCHAR(20)     NOT NULL DEFAULT 'PENDING', -- PENDING | SUBMITTED | CONSENT_REQUESTED | CONSENT_GRANTED | REFUSED
    consent_deadline    DATETIME                 DEFAULT NULL,
    submitted_at        DATETIME                 DEFAULT NULL,
    notes               TEXT                     DEFAULT NULL,
    created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_referral_id     (referral_id),
    INDEX idx_sar_alert           (alert_id),
    INDEX idx_sar_status          (filing_status),
    INDEX idx_sar_submitted_at    (submitted_at),
    CONSTRAINT fk_sar_alert FOREIGN KEY (alert_id) REFERENCES aml_alerts (alert_id)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='SAR filing tracker. Consent regime tracked per POCA 2002 s.335-336.';


-- ---------------------------------------------------------------------------
-- audit_log   (FCA SYSC 10A)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGINT          NOT NULL AUTO_INCREMENT,
    event_id        VARCHAR(50)     NOT NULL DEFAULT (UUID()),
    scorer          VARCHAR(60)     NOT NULL,   -- class name: SARScorer, AMLIsolationForest, etc.
    model_version   VARCHAR(20)     NOT NULL,
    event           VARCHAR(50)     NOT NULL,   -- train | predict | score | explain | evaluate
    transaction_id  VARCHAR(50)              DEFAULT NULL,
    n_records       INT                      DEFAULT NULL,
    metadata_json   JSON                     DEFAULT NULL,
    event_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    INDEX idx_audit_scorer      (scorer, event),
    INDEX idx_audit_txn         (transaction_id),
    INDEX idx_audit_event_at    (event_at),
    INDEX idx_audit_event_type  (event)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='FCA SYSC 10A automated decision audit trail. Immutable; no UPDATE or DELETE permitted.';


-- =============================================================================
-- Views
-- =============================================================================

-- Active alert queue — primary MLRO work queue
CREATE OR REPLACE VIEW v_active_alerts AS
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
    a.status,
    a.scored_at,
    t.sender_account_id,
    t.receiver_account_id,
    t.channel,
    t.country_origin,
    t.country_destination,
    t.is_cross_border
FROM aml_alerts   a
JOIN transactions t ON t.transaction_id = a.transaction_id
WHERE a.status = 'OPEN'
ORDER BY a.priority ASC, a.risk_score DESC;


-- SAR-recommended alerts not yet referred
CREATE OR REPLACE VIEW v_sar_pending AS
SELECT
    a.alert_id,
    a.transaction_id,
    a.risk_score,
    a.risk_tier,
    a.trigger_reasons,
    a.mlro_summary,
    a.amount_gbp,
    a.scored_at,
    TIMESTAMPDIFF(HOUR, a.scored_at, NOW()) AS hours_open
FROM aml_alerts  a
LEFT JOIN sar_referrals r ON r.alert_id = a.alert_id
WHERE a.sar_recommended = 1
  AND a.status IN ('OPEN', 'UNDER_REVIEW')
  AND r.referral_id IS NULL
ORDER BY a.priority ASC, a.scored_at ASC;


-- Daily alert MI — management information summary
CREATE OR REPLACE VIEW v_daily_alert_mi AS
SELECT
    DATE(scored_at)                                        AS alert_date,
    COUNT(*)                                               AS total_alerts,
    SUM(priority = 1)                                      AS p1_critical,
    SUM(priority = 2)                                      AS p2_high,
    SUM(priority = 3)                                      AS p3_medium,
    SUM(sar_recommended = 1)                               AS sar_recommended,
    ROUND(AVG(risk_score), 4)                              AS mean_risk_score,
    ROUND(SUM(sar_recommended) / COUNT(*) * 100, 2)        AS sar_rate_pct
FROM aml_alerts
GROUP BY DATE(scored_at)
ORDER BY alert_date DESC;
