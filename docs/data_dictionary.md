# Data Dictionary v1 — FinCrime-ML Transaction Schema

**Version:** 1.0  
**Date:** 2024-01-07  
**Author:** Temidayo Akindahunsi  
**Status:** Active

---

## Overview

This document defines the canonical field schema for all transaction data
flowing through FinCrime-ML. Three schemas are in scope: the **core synthetic
transaction schema** (produced by `SyntheticTransactionGenerator`), the
**digital payments schema** (produced by `generate_digital_payments`), and the
**IEEE-CIS harmonised schema** (produced by `IeeeCisLoader`). A fourth section
maps schema fields to the UK regulatory frameworks that govern their use in
transaction monitoring and fraud detection systems.

The core schema is the lingua franca of the fraud pipeline. All external data
sources — whether real datasets loaded via adapters or synthetic data from the
generators — must be harmonised to this schema before entering the feature
engineering stage. Fields not available in an external source are assigned
documented sentinel values rather than being silently dropped, enabling
downstream code to distinguish "field was absent in source" from "field had a
genuine zero/null value".

---

## 1. Core Transaction Schema

Produced by `fincrime_ml.core.data.synth_cards.SyntheticTransactionGenerator.generate()`.
The canonical reference for all fraud pipeline modules.

| Field | Type | Nullable | Example | Description |
|-------|------|----------|---------|-------------|
| `transaction_id` | `str` | No | `TXN-ABCDEFGH` | Unique transaction reference. Format: `TXN-` followed by 8 uppercase alphanumeric characters. Collision probability negligible at expected dataset sizes (< 10M rows). |
| `account_id` | `str` | No | `ACC0001234` | Masked account identifier. Format: `ACC` followed by 7 zero-padded digits. Used as the grouping key for velocity and behavioural features. |
| `merchant_id` | `str` | No | `MER-A1B2C3D4E5` | Merchant reference. Format: `MER-` followed by 10 alphanumeric characters. |
| `mcc` | `str` | No | `7995` | ISO 18245 Merchant Category Code. Four-digit string. Used to classify transaction purpose and assign baseline risk tier. See Section 4 for MCC risk mapping. |
| `mcc_name` | `str` | No | `Gambling / Betting` | Human-readable MCC description. Derived from the ISO 18245 registry. |
| `mcc_risk` | `str` | No | `high` | MCC risk tier: `low`, `medium`, or `high`. Assigned per the UK Finance MCC risk segmentation. `unknown` is used as a sentinel when the source dataset has no MCC equivalent (see IEEE-CIS harmonisation). |
| `channel` | `str` | No | `CNP_ECOM` | Payment channel. Enumerated values defined in Section 2. The channel is the primary typology discriminator: CNP channels carry the highest card fraud risk; POS channels dominate skimming scenarios. |
| `amount_gbp` | `float` | No | `142.50` | Transaction amount in GBP (or the sentinel currency for external datasets). Rounded to two decimal places. Right-skewed distribution; fraud rows exhibit systematically higher amounts relative to account baseline. |
| `currency` | `str` | No | `GBP` | ISO 4217 currency code. Synthetic data uses `GBP` throughout. IEEE-CIS harmonised data uses `USD`. |
| `country_origin` | `str` | No | `GB` | ISO 3166-1 alpha-2 code for the transaction origin country (cardholder / payer location). Used in geographic displacement and high-risk corridor features. |
| `country_destination` | `str` | No | `US` | ISO 3166-1 alpha-2 code for the transaction destination country (merchant / payee location). Combined with `country_origin` to derive `is_international` and corridor risk. |
| `timestamp` | `datetime` | No | `2024-03-15 02:47:11` | UTC datetime of transaction authorisation. The source of all derived temporal features. |
| `hour_of_day` | `int` | No | `2` | Hour extracted from `timestamp` (0–23). Off-hours transactions (0–6 h) are a leading indicator for CNP fraud and ATO attacks. dtype is `int32` (from `dt.hour`). |
| `day_of_week` | `int` | No | `4` | Day of week from `timestamp`. 0 = Monday, 6 = Sunday. |
| `is_weekend` | `int` | No | `0` | Binary flag: 1 if `day_of_week` ∈ {5, 6}, else 0. |
| `account_age_days` | `int` | No | `365` | Number of days since the account was opened at transaction time. New accounts (< 90 days) are a bust-out risk indicator. |
| `account_avg_spend` | `float` | No | `68.40` | Account's historical mean transaction amount. Used to compute amount deviation (z-score) in feature engineering. Drawn from log-normal distribution during synthesis. |
| `account_spend_stddev` | `float` | No | `21.30` | Standard deviation of the account's historical transaction amounts. Combined with `account_avg_spend` to produce a per-account spend envelope. |
| `is_international` | `int` | No | `1` | Binary flag: 1 if `country_origin ≠ country_destination`, else 0. |
| `high_risk_corridor` | `int` | No | `0` | Binary flag: 1 if either `country_origin` or `country_destination` is in the FATF grey/black list set {IR, KP, AE}. |
| `is_mule_account` | `int` | No | `0` | Binary flag: 1 if the account is designated as a mule account in the account population. Approximately 0.8% of synthetic accounts are flagged. Used in AML typology injection. |
| `swift_bic` | `str` | Yes | `BARCGB22XXX` | SWIFT Bank Identifier Code. Populated for `WIRE` channel transactions only; `None` for all other channels. Format: 8 or 11 characters per ISO 9362. |
| `iban` | `str` | Yes | `GB29NWBK60161331926819` | International Bank Account Number of the beneficiary. Populated for `WIRE` channel only; `None` otherwise. Synthetic IBANs are structurally plausible but check digits are not validated. |
| `is_fraud` | `int` | No | `0` | Binary fraud label. 1 = confirmed fraud, 0 = legitimate. The **target variable** for all fraud pipeline models. Corresponds to `isFraud` in IEEE-CIS and is populated by `TypologyInjector` during synthetic data preparation. |

---

## 2. Channel Enumeration

The `channel` field classifies the payment method and physical context of the
transaction. Channel choice drives both the applicable fraud typology and the
relevant regulatory framework (PSR 2017 / PSD2 SCA requirements differ by
channel).

| Value | Full name | SCA required | Primary fraud risk |
|-------|-----------|--------------|-------------------|
| `POS` | Point of Sale (card-present) | No (chip & PIN satisfies) | Card-present skimming, lost/stolen card |
| `CNP_ECOM` | Card-Not-Present — E-commerce | Yes (PSD2 SCA) | CNP fraud, ATO-initiated purchases |
| `CNP_MOTO` | Card-Not-Present — Mail/Telephone Order | SCA exemption applies | CNP fraud |
| `CONTACTLESS` | Contactless card-present | No (value limit applies) | Lost/stolen low-value |
| `WIRE` | SWIFT wire transfer | Out-of-scope for PSD2 SCA | AML structuring, layering |
| `MOBILE_APP` | Mobile banking application | Yes (app-native SCA) | ATO following credential theft |
| `OPEN_BANKING` | PSD2 Open Banking (PISP/AISP) | Yes — near-universal SCA | APP fraud (Authorised Push Payment) |
| `BNPL` | Buy Now Pay Later | Partial SCA (provider-dependent) | BNPL first-party fraud, ATO |
| `CRYPTO_OFFRAMP` | Cryptocurrency exchange off-ramp | Provider-level controls only | APP fraud, money mule layering |
| `DIGITAL_WALLET` | Digital wallet (Apple Pay, Google Pay, etc.) | Device-bound SCA | ATO, new-device takeover |

Strong Customer Authentication (SCA) is mandated by PSR 2017 / PSD2 Article 97
for electronic remote payments. Channels where SCA is absent or inconsistently
applied carry materially higher fraud risk; this is reflected in the
`provider_fraud_risk` field in the digital payments schema.

---

## 3. Digital Payments Schema

Produced by `SyntheticTransactionGenerator.generate_digital_payments()`. Covers
Open Banking, BNPL, digital wallets, and cryptocurrency off-ramp transactions.
This schema is a narrower, channel-specific view that extends the core schema
with provider-level fields.

| Field | Type | Nullable | Example | Description |
|-------|------|----------|---------|-------------|
| `payment_id` | `str` | No | `DPY-ABCDEFGHIJ` | Unique payment reference. Format: `DPY-` followed by 10 uppercase characters. |
| `account_id` | `str` | No | `ACC0001234` | Account identifier — same format as core schema. |
| `provider` | `str` | No | `Klarna` | Payment provider name. One of the providers defined in `DIGITAL_PAYMENT_PROVIDERS` (e.g. Revolut, Monzo, Klarna, TrueLayer, Coinbase). |
| `payment_type` | `str` | No | `bnpl` | Provider type category: `digital_wallet`, `bnpl`, `open_banking`, or `crypto_offramp`. |
| `amount_gbp` | `float` | No | `249.99` | Transaction amount. Fraud rows are drawn from a higher-mean lognormal distribution (μ=5.2) vs legitimate rows (μ=4.1). |
| `currency` | `str` | No | `GBP` | ISO 4217 currency code. |
| `ip_country` | `str` | No | `GB` | ISO 3166-1 alpha-2 code of the IP address country at time of transaction. IP–account country mismatch is a strong ATO signal. |
| `device_type` | `str` | No | `mobile` | Device category: `mobile`, `desktop`, or `tablet`. |
| `is_3ds_authenticated` | `int` | No | `1` | Binary: 1 if transaction passed 3-D Secure (Verified by Visa / Mastercard Identity Check). Fraudulent transactions have systematically lower 3DS rates — SCA bypass is a defining CNP fraud characteristic under PSR 2017 / PSD2. |
| `provider_fraud_risk` | `str` | No | `high` | Provider-level fraud risk tier: `low`, `medium`, or `high`. Reflects 3DS adoption rates and APP fraud exposure per UK Finance 2023 data. |
| `timestamp` | `datetime` | No | `2024-06-01 14:22:05` | UTC transaction datetime. |
| `hour_of_day` | `int` | No | `14` | Hour extracted from `timestamp` (0–23). |
| `day_of_week` | `int` | No | `5` | Day of week (0=Monday, 6=Sunday). |
| `is_weekend` | `int` | No | `1` | Binary weekend flag. |
| `is_international_ip` | `int` | No | `0` | Binary: 1 if `ip_country` is not `GB` or `OTHER`. Signals a potential geographic mismatch between the account's home country and the transaction origin. |
| `is_fraud` | `int` | No | `0` | Binary fraud label — same semantics as core schema. |

---

## 4. IEEE-CIS Harmonised Schema

Produced by `fincrime_ml.core.data.loaders.IeeeCisLoader`. Maps the Kaggle
IEEE-CIS Fraud Detection dataset (Vesta Corporation, 2019) to the FinCrime-ML
internal schema. Columns with no IEEE-CIS equivalent are assigned the sentinel
values listed below; these sentinels allow downstream code to identify fields
that are structurally absent rather than merely zero-valued.

| Field | Source | Sentinel | Notes |
|-------|--------|----------|-------|
| `transaction_id` | `TransactionID` (prefixed `TXN-`) | — | Direct map with prefix. |
| `account_id` | `card1` | `ACC_UNKNOWN` | `card1` is a masked card number feature. Format: `ACC{card1:07.0f}`. |
| `merchant_id` | — | `MER-UNKNOWN` | No merchant identifier in IEEE-CIS. |
| `mcc` | — | `0000` | IEEE-CIS has no MCC field. |
| `mcc_name` | — | `Unknown` | Derived from `mcc` sentinel. |
| `mcc_risk` | — | `unknown` | Deliberately distinct from `low`/`medium`/`high`. Conservative treatment per JMLSG Ch. 5. |
| `channel` | `ProductCD` | `CNP_ECOM` | W/H → `CNP_ECOM`; C/R → `POS`; S → `CNP_MOTO`. Unknown values default to `CNP_ECOM`. |
| `amount_gbp` | `TransactionAmt` | — | USD amounts; `currency` is set to `USD`. |
| `currency` | — | `USD` | Dataset is US-centric. |
| `country_origin` | — | `US` | `addr2` is an encoded numeric field; ISO mapping is unavailable. |
| `country_destination` | — | `US` | Domestic default. |
| `hour_of_day` | `TransactionDT` | — | `(TransactionDT % 86400) // 3600` |
| `day_of_week` | `TransactionDT` | — | `(TransactionDT // 86400) % 7` |
| `is_weekend` | Derived | — | `day_of_week` ∈ {5, 6} |
| `is_international` | — | `0` | Geography unavailable; conservative default. |
| `high_risk_corridor` | — | `0` | Geography unavailable; conservative default. |
| `is_mule_account` | — | `0` | No account designation in IEEE-CIS. |
| `swift_bic` | — | `None` | No wire transfer records in IEEE-CIS. |
| `iban` | — | `None` | No wire transfer records in IEEE-CIS. |
| `is_fraud` | `isFraud` | — | Direct map. |
| `transaction_dt_raw` | `TransactionDT` | — | Retained raw seconds offset for feature engineering. |
| `email_domain_payer` | `P_emaildomain` | `None` | Payer email domain — strong ATO and CNP fraud signal in literature. |
| `email_domain_payee` | `R_emaildomain` | `None` | Payee email domain. |
| `device_type` | `DeviceType` (identity join) | `NaN` | Populated only for rows with an identity record. `NaN` signals absent, not unknown. |

---

## 5. MCC Risk Classification

Merchant Category Codes are assigned to one of three risk tiers based on UK
Finance MCC risk segmentation. This classification is used in feature
engineering (as a direct categorical feature), in typology injection (fraud
rows are biased toward high-risk MCCs), and in rule-based pre-screening
(JMLSG Part I Ch. 5 mandates documented rationale for threshold differences
between merchant risk categories).

| MCC | Merchant Category | Risk Tier | Fraud relevance |
|-----|-------------------|-----------|-----------------|
| 5411 | Grocery Stores | `low` | Low fraud rate; common in card-testing small-amount transactions |
| 5812 | Eating Places / Restaurants | `low` | Low fraud rate |
| 5912 | Drug Stores / Pharmacies | `low` | Low fraud rate |
| 5661 | Shoe Stores | `low` | Low fraud rate |
| 5651 | Family Clothing Stores | `low` | Low fraud rate |
| 4111 | Transportation | `low` | Low fraud rate; skimming testing venue |
| 5999 | Miscellaneous Retail | `medium` | Elevated; frequently used in bust-out fraud |
| 7011 | Hotels and Lodging | `medium` | CNP fraud — card-absent booking context |
| 5734 | Computer and Software Stores | `medium` | CNP fraud — high-value digital goods |
| 5045 | Computers / Peripherals (B2B) | `medium` | Elevated; large-ticket CNP |
| 7995 | Gambling / Betting | `high` | Bust-out fraud; AML layering |
| 6051 | Non-Financial Institutions (Crypto) | `high` | AML layering / integration |
| 4829 | Money Transfer / Remittance | `high` | AML structuring; CNP fraud proceeds |
| 6011 | ATM / Cash Advance | `high` | Bust-out; skimming extraction; AML smurfing |
| 5094 | Jewellery / Watches | `high` | CNP fraud — high-value portable assets |

---

## 6. Country Corridor Risk Classification

Transaction corridors are classified based on FATF grey and black list
membership, used to populate `high_risk_corridor` and to weight country
sampling in synthetic data generation.

| Country | ISO Code | FATF Status | Corridor weight (synthetic) |
|---------|----------|-------------|----------------------------|
| United Kingdom | GB | Not listed | 0.45 |
| United States | US | Not listed | 0.15 |
| Germany | DE | Not listed | 0.08 |
| France | FR | Not listed | 0.06 |
| Nigeria | NG | Grey list | 0.04 |
| UAE | AE | Grey list | 0.04 |
| Hong Kong | HK | Not listed | 0.03 |
| China | CN | Not listed | 0.03 |
| Russia | RU | Not listed | 0.02 |
| Iran | IR | **Black list** | 0.01 |
| North Korea | KP | **Black list** | 0.01 |
| Other | OTHER | — | 0.08 |

The `high_risk_corridor` flag is set to 1 when either `country_origin` or
`country_destination` is IR, KP, or AE. This deliberately includes AE (grey
list) due to its frequent appearance in UK financial crime cases involving trade
finance and remittance corridors.

---

## 7. Regulatory Field Mappings

The table below maps schema fields to the UK and international regulatory
obligations that govern their use in a production transaction monitoring system.
This mapping is provided to assist compliance teams in evidencing that the
model's input features are aligned to regulatory requirements (FCA SYSC 6.3,
JMLSG Part I Ch. 5) and that the feature set covers the risk dimensions
identified by supervisory guidance.

| Field(s) | Regulatory framework | Obligation |
|----------|---------------------|------------|
| `amount_gbp` | POCA 2002 s.330; MLR 2017 Reg. 28 | Transaction monitoring must consider amount thresholds. POCA s.330 establishes the disclosure trigger; MLR 2017 Reg. 28 requires customer activity to be monitored against their expected transaction profile, of which amount is the primary dimension. |
| `amount_gbp` | FCA SYSC 6.3.1R | Firms must maintain adequate policies and procedures for transaction monitoring. Amount-based rules are the most basic form of such monitoring and must be documented. |
| `country_origin`, `country_destination`, `high_risk_corridor` | FATF Recommendations 10–12 | Customer due diligence (CDD) and enhanced due diligence (EDD) obligations apply to transactions involving high-risk jurisdictions. `high_risk_corridor = 1` triggers EDD eligibility under MLR 2017 Reg. 33. |
| `channel`, `is_3ds_authenticated` | PSR 2017; PSD2 Art. 97 | Strong Customer Authentication is mandatory for electronic remote payments. Absence of SCA (captured by `is_3ds_authenticated = 0` for CNP channels) is a reportable failure and a primary CNP fraud risk factor. |
| `mcc`, `mcc_risk` | JMLSG Part I Ch. 5.3 | JMLSG guidance requires documented rationale for applying different monitoring thresholds to different merchant categories. `mcc_risk` encodes this segmentation. |
| `account_age_days`, `account_avg_spend`, `account_spend_stddev` | MLR 2017 Reg. 28(11) | Firms must monitor transactions against the customer's established profile. These fields represent the account behavioural baseline against which deviation is measured. |
| `is_mule_account` | POCA 2002 s.328 | Participating in an arrangement that facilitates the acquisition, retention, use or control of criminal property. Mule account flags trigger SAR consideration. |
| `swift_bic`, `iban` | MLR 2017 Reg. 19; FATF Rec. 16 | Wire transfer monitoring requires beneficiary identification. BIC and IBAN are the primary identifiers for SWIFT payments and must be captured for correspondent banking due diligence. |
| `is_fraud` (label) | FCA SYSC 10A | Firms using automated decision systems must maintain records sufficient to reconstruct decisions. The label column is the ground truth anchor for model validation under PRA SS1/23. |
| `timestamp`, `hour_of_day` | FCA SYSC 10A.1.1R | Transaction record-keeping obligations require accurate timestamps. Temporal features are also relevant to FCA expectations on real-time fraud monitoring. |
| `email_domain_payer`, `email_domain_payee` | MLR 2017 Reg. 28 | Email domain anomalies (e.g. free webmail domains in high-value transactions) are a documented ATO and CNP fraud indicator. Included as optional monitoring input. |

---

## 8. Label Conventions

FinCrime-ML uses two binary label columns, one per domain. The distinction is
architecturally enforced and has direct regulatory significance.

| Label column | Domain | Possible values | Trigger |
|---|---|---|---|
| `is_fraud` | Payment fraud (`fincrime_ml.fraud`) | 0 / 1 | Card block; dispute process; PSR 2017 reimbursement obligation |
| `is_suspicious` | AML (`fincrime_ml.aml`) | 0 / 1 | SAR submission to NCA under POCA 2002 s.330; tipping-off constraint applies |

These labels must not be conflated. An `is_fraud = 1` event may be communicated
to the cardholder immediately (the payment is blocked and the customer informed).
An `is_suspicious = 1` event initiates a SAR workflow; the subject must not be
informed under any circumstances (POCA 2002 s.333A tipping-off offence). This
distinction is the primary rationale for maintaining separate pipeline modules,
as recorded in ADR 001.

---

## 9. Sentinel Value Summary

Sentinel values mark fields that are structurally absent in a given data source,
as opposed to fields that are present but zero or null. Downstream feature
engineering must handle sentinels explicitly rather than treating them as
ordinary categorical values.

| Sentinel | Type | Meaning |
|----------|------|---------|
| `"unknown"` | `mcc_risk` | MCC data was unavailable in the source dataset. Apply conservative (high-risk) treatment per JMLSG Ch. 5 guidance. |
| `"0000"` | `mcc` | No MCC field in source. |
| `"Unknown"` | `mcc_name` | Derived from `mcc = "0000"`. |
| `"MER-UNKNOWN"` | `merchant_id` | No merchant identifier in source. |
| `"USD"` | `currency` | IEEE-CIS dataset amounts are USD; not equivalent to GBP model inputs. |
| `"US"` | `country_origin` / `country_destination` | Geographic data unavailable; US default for IEEE-CIS. |
| `0` | `is_international`, `high_risk_corridor` | Conservative default when geography is unavailable. |
| `"ACC_UNKNOWN"` | `account_id` | `card1` was NaN in source row. |
| `None` | `swift_bic`, `iban` | Non-wire transaction; fields are not applicable. |
| `NaN` | `device_type` | Transaction had no corresponding identity record (IEEE-CIS partial identity coverage). |

---

## References

- UK Finance: [Annual Fraud Report 2023](https://www.ukfinance.org.uk/policy-and-guidance/reports-and-publications)
- FCA: [SYSC 6.3 — Financial crime systems and controls](https://www.handbook.fca.org.uk/handbook/SYSC/6/3.html)
- FCA: [SYSC 10A — Algorithmic trading and automated decision systems](https://www.handbook.fca.org.uk/handbook/SYSC/10A/)
- HM Treasury: [POCA 2002 s.330 — Failure to disclose](https://www.legislation.gov.uk/ukpga/2002/29/section/330)
- HM Treasury: [POCA 2002 s.333A — Tipping off](https://www.legislation.gov.uk/ukpga/2002/29/section/333A)
- HM Treasury: [MLR 2017 — Money Laundering Regulations](https://www.legislation.gov.uk/uksi/2017/692/contents)
- FCA / PSR: [Payment Services Regulations 2017](https://www.legislation.gov.uk/uksi/2017/752/contents)
- European Parliament: [PSD2 — Payment Services Directive 2](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32015L2366)
- PRA: [SS1/23 — Model risk management principles](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss)
- JMLSG: [Part I Chapter 5 — Transaction monitoring](https://www.jmlsg.org.uk/guidance/)
- FATF: [40 Recommendations](https://www.fatf-gafi.org/en/topics/fatf-recommendations.html)
- ISO 18245: Merchant category codes
- ISO 9362: SWIFT BIC standard
- ISO 13616: IBAN standard
