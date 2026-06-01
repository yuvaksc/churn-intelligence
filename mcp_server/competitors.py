"""
Simulated competitor intelligence by US state.
Feature dictionary explaining all model features for agent context.
"""

COMPETITOR_DATA = {
    "California": {
        "top_competitor":       "Pacific Fiber Co",
        "avg_competitor_price": "$62/month",
        "current_promotions":   "50% off first 3 months + free professional installation",
        "competitor_strengths": ["Symmetrical upload/download speeds", "No annual contracts"],
        "our_advantage":        "24/7 local tech support, no data caps, 99.9% uptime SLA",
    },
    "New York": {
        "top_competitor":       "Empire State Telecom",
        "avg_competitor_price": "$68/month",
        "current_promotions":   "Free streaming bundle (3 services) for 12 months",
        "competitor_strengths": ["TV bundle options", "Lower advertised base price"],
        "our_advantage":        "Broader coverage including outer boroughs, better customer ratings",
    },
    "Texas": {
        "top_competitor":       "Lone Star Broadband",
        "avg_competitor_price": "$58/month",
        "current_promotions":   "$200 switching credit + 3 months free service",
        "competitor_strengths": ["Aggressive pricing", "No annual contract lock-in"],
        "our_advantage":        "Stronger rural and suburban coverage, no throttling",
    },
    "Florida": {
        "top_competitor":       "Sunshine Networks",
        "avg_competitor_price": "$61/month",
        "current_promotions":   "First 6 months at 40% off for switchers",
        "competitor_strengths": ["Price for basic tiers", "Seasonal promotional pricing"],
        "our_advantage":        "Storm-hardened infrastructure, faster outage recovery",
    },
    "Illinois": {
        "top_competitor":       "Midwest Connect",
        "avg_competitor_price": "$64/month",
        "current_promotions":   "No installation fee + free router for new customers",
        "competitor_strengths": ["Lower entry price", "Flexible no-contract options"],
        "our_advantage":        "Higher speeds on fiber tiers, bundled security suite",
    },
    "Washington": {
        "top_competitor":       "Cascade Fiber",
        "avg_competitor_price": "$67/month",
        "current_promotions":   "Free gigabit upgrade for first year",
        "competitor_strengths": ["Gigabit speeds at competitive price", "Local brand trust"],
        "our_advantage":        "More service plans, established customer support network",
    },
    "DEFAULT": {
        "top_competitor":       "National Fiber Networks",
        "avg_competitor_price": "$65/month",
        "current_promotions":   "First 2 months free for verified switchers",
        "competitor_strengths": ["Price", "Promotional switching incentives"],
        "our_advantage":        "Reliability, local support presence, no hidden fees",
    },
}

# Describes every feature in the 35-column model input so agents
# can interpret SHAP values in plain English.
FEATURE_DICTIONARY = {
    "Tenure Months": (
        "How many months the customer has been with us. "
        "Longer tenure = lower churn risk. Under 12 months is high-risk."
    ),
    "Contract": (
        "Contract commitment: Month-to-month (1) highest churn risk, "
        "One year (12), Two year (24) most stable. "
        "Encoded as months of commitment."
    ),
    "Monthly Charges": (
        "Current monthly bill in USD. Higher charges correlate with churn, "
        "especially for month-to-month customers."
    ),
    "Total Charges": (
        "Lifetime spend with the company. "
        "Low total with high monthly = new expensive customer = very high risk."
    ),
    "Charge Ratio": (
        "Monthly Charges / Total Charges. "
        "High ratio means recent customer paying a lot relative to history. Strong churn signal."
    ),
    "Avg Monthly Charge": (
        "Total Charges / Tenure Months. "
        "Compares current bill to historical average — price increase sensitivity indicator."
    ),
    "Services Count": (
        "Number of add-on services subscribed (Online Security, Backup, "
        "Device Protection, Tech Support, Streaming TV, Streaming Movies). "
        "0-1 services = high churn risk. 4+ = sticky customer."
    ),
    "High Risk Flag": (
        "Binary: 1 if Month-to-month contract AND Monthly Charges above dataset median ($64.76). "
        "Captures the most dangerous churn segment directly."
    ),
    "Tenure Bin": (
        "Bucketed tenure: 0=under 1yr, 1=1-3yr, 2=3-5yr, 3=over 5yr. "
        "Bin 0 is dramatically higher risk."
    ),
    "Sticky Payment": (
        "Binary: 1 if Paperless Billing=Yes AND automatic payment method. "
        "Sticky payment customers churn less (lower friction to stay)."
    ),
    "Charge Index": (
        "Monthly Charges relative to median for same Internet+Phone service bundle. "
        "Index > 1.2 means customer is paying above average for their bundle type."
    ),
    "Internet Service": (
        "DSL, Fiber optic, or No internet. "
        "Fiber optic customers have the highest churn rate in this dataset."
    ),
    "Online Security": (
        "Add-on security service. Customers without it churn significantly more. "
        "Good retention lever — offer it free."
    ),
    "Tech Support": (
        "Add-on tech support. Same pattern as Online Security. "
        "Absence correlates strongly with churn."
    ),
    "Payment Method": (
        "Electronic check correlates with high churn. "
        "Automatic payments (bank transfer, credit card) are more stable."
    ),
    "Senior Citizen": "Binary: 1 if customer is 65+. Slight positive churn correlation.",
    "Gender": "Male/Female. Minimal churn impact in this dataset.",
    "Partner": "Has a partner: more stable, less likely to churn.",
    "Dependents": "Has dependents: more stable, less likely to churn.",
    "Phone Service": "Has phone service. Mostly a segmentation feature.",
    "Paperless Billing": "Paperless billing enabled. Often co-occurs with electronic check.",
}


def get_competitor_info(state: str = "DEFAULT") -> dict:
    return COMPETITOR_DATA.get(state, COMPETITOR_DATA["DEFAULT"])