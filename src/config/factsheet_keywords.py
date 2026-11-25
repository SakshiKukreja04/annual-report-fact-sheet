"""Generic fact-sheet keywords. Not company-specific."""

from __future__ import annotations

import re

# Pages that can hold a segment-revenue table or the segment note.
SEGMENT_PAGE = re.compile(
    r"segment revenue|reportable segment|segment information|operating segments|segment[-\s]wise",
    re.I,
)
NO_SEPARATE_SEGMENTS = re.compile(r"no separate reportable segments?", re.I)

# Pages that can hold the company overview used for one_liner and summary.
OVERVIEW_PAGE = re.compile(
    r"business overview|our business|company overview|about the company",
    re.I,
)

# Subsidiary, joint-venture, associate, and AOC-1 list pages.
LISTING_PAGE = re.compile(
    r"form\s*aoc-?\s*1|statement containing salient features|"
    r"investment in subsidiaries|investment in associates|investment in joint ventures|"
    r"list of material unlisted subsidiar|name of the subsidiar|"
    r"name of the joint venture|name of the associate",
    re.I,
)

# Tables of unrelated parties, or related-party transactions with no investee list, are skipped.
RELATED_PARTY_HEADER = re.compile(r"unrelated\s+part|related\s+part", re.I)
LISTING_HEADER = re.compile(
    r"subsidiar|joint\s+venture|\bassociate\b|aoc-?\s*1|name of the entity|name of the company",
    re.I,
)

NAME_HEADER = re.compile(
    r"name of the (?:subsidiar\w*|joint venture|associate|entity|company)|"
    r"^name$|particulars",
    re.I,
)
ACTIVITY_HEADER = re.compile(
    r"principal activit|nature of (?:business|activit)|"
    r"description of (?:business|activit)|main activit|\bactivit",
    re.I,
)

# A unit phrase printed on the page or above the table. Never a default.
UNIT_PHRASE = re.compile(
    r"(?:₹|rs\.?|inr|usd|\$)\s*(?:in\s+)?(?:crore|million|lakh|billion|thousand)s?"
    r"|(?:in\s+)?(?:₹|rs\.?|inr)\s*(?:crore|million|lakh|billion)"
    r"|rs\.?\s+in\s+\w+",
    re.I,
)

REVENUE_HEADER = re.compile(r"revenue|turnover|sales", re.I)
INTER_HEADER = re.compile(r"inter[-\s]?segment", re.I)
EXTERNAL_HEADER = re.compile(r"external", re.I)
TOTAL_LABEL = re.compile(r"^total\b", re.I)
ELIMINATION_LABEL = re.compile(r"eliminat|inter[-\s]?segment", re.I)

# Operations language used to choose an overview page. Not a company list.
OPERATIONS_DENSITY = re.compile(
    r"subsidiar|mines?|produces?|operates?|coalfields",
    re.I,
)
STRATEGY_PAGE = re.compile(
    r"\bstrateg(?:y|ic)\b|vision and mission|materiality|corporate governance",
    re.I,
)
ESG_PAGE = re.compile(
    r"\besg\b|sustainability|environmental,? social|climate change|net zero",
    re.I,
)
BUSINESS_TOPIC = re.compile(
    r"subsidiar|joint venture|\bmines?\b|produc\w*|\bsales?\b|e-?\s*auction|\bcustomers?\b|\bchannels?\b",
    re.I,
)

# Generic nouns that are not a project name on their own.
GENERIC_HEAD = re.compile(r"^(mines?|plants?|divisions?|units?|blocks?|labs?)$", re.I)
GENERIC_MODIFIER = {
    "underground", "captive", "power", "coal", "handling", "testing", "first", "class",
    "open", "thermal", "solar", "the", "of", "and", "new", "old", "main", "central",
    "western", "eastern", "northern", "southern", "virtual", "international", "eco",
}

# Accounting-schedule language. An asset name from these pages needs an activity sentence.
ACCOUNTING_PAGE = re.compile(
    r"property,?\s+plant|capital work[-\s]?in[-\s]?progress|\bcwip\b|"
    r"accounting polic|notes to the|development expenditure|stripping activity|"
    r"surveyed off assets|capital expenditure",
    re.I,
)
POLICY_TERM = re.compile(
    r"property,?\s+plant|surveyed off assets|development expenditure|stripping activity asset",
    re.I,
)
COUNTERPARTY = re.compile(
    r"\bm/s\b|\bcontractor\b|awarded to|handed over to|\bcustomer\b|\bbank\b|\bfund\b",
    re.I,
)
GOVERNANCE_ITEM = re.compile(
    r"\b(vigilance|committee|pension|award|welfare|grievance|remuneration|nomination)\b",
    re.I,
)
FUTURE_CUE = re.compile(
    r"\bwill\b|\bproposed\b|plans to|to be set up|mou to explore|under construction|yet to commence",
    re.I,
)
PAST_CUE = re.compile(
    r"has started|commissioned|was developed|from\s+(?:19|20)\d{2}|\bawarded\b",
    re.I,
)
TARGET_LINE = re.compile(
    r"\b\d+(?:\.\d+)?\s*gw\b.{0,48}\btarget\b|capacity\s*[-–]\s*\d|target of\s+\d",
    re.I,
)
SALES_CHANNEL = re.compile(
    r"e-?\s*auction|\bfsa\b|\bnrs\b|power sector|non[-\s]?power|\bimports?\b",
    re.I,
)
CORE_PRODUCT = re.compile(
    r"non-coking coal|coking coal|washed coal|thermal coal|coal grades?\s+g1\s*[-–]\s*g17|"
    r"\bexports?\b|power generation",
    re.I,
)

# Notes, schedules, disputes, and related-party disclosures are not operating pages.
SOURCE_JUNK_PAGE = re.compile(
    r"notes to the (?:standalone |consolidated )?financial|"
    r"capital work[-\s]?in[-\s]?progress|\bcwip\b|"
    r"property,?\s+plant and equipment|"
    r"\blitigation\b|contingent liabilit|related party disclosure",
    re.I,
)
BIO_START = re.compile(r"^(?:he|she|shri|smt\.?|mr\.?|mrs\.?)\b", re.I)
AWARD_ITEM = re.compile(r"\b(award|title conferred|winner|runners?-up)\b", re.I)
FACILITY_TAIL = re.compile(r"(?:mines?|plants?|divisions?|labs?|units?)$", re.I)
# A shared topic key has to be more specific than these words.
TOPIC_GENERIC = {
    "coal", "power", "energy", "mine", "mines", "washery", "india", "national",
    "clean", "washed", "coking", "thermal", "limited", "company", "project",
    "plant", "system", "centre", "center", "phase", "based", "joint", "venture",
    "water", "treatment", "rapid", "loading", "monitoring", "rescue", "medical",
    "regional", "area", "open", "cast", "underground", "testing", "research",
    "training", "workshop", "unit", "division", "lab",
}
TOPIC_ALIASES = (
    (re.compile(r"synthetic natural gas|\bsng\b|coal gasification", re.I), "coal gasification"),
    (re.compile(r"fuel supply agreements?|\bfsas?\b", re.I), "fuel supply agreement"),
    (re.compile(r"digital dispensar\w*|telemedicine", re.I), "remote care"),
    (re.compile(r"vehicle tracking|\bvts\b", re.I), "vehicle tracking"),
    (re.compile(r"pumped?\s+storage", re.I), "pump storage"),
    (re.compile(r"direct shear", re.I), "direct shear"),
)
