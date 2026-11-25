"""Generic keyword lists for page ranking. Not company-specific."""

from __future__ import annotations

import re

# Business / operations vocabulary shared across industrials, utilities, and miners.
POSITIVE_TERMS = (
    r"aluminium|aluminum|copper|alumina|bauxite|smelter|refinery|capacity|products?|brands?|"
    r"subsidiar\w*|joint ventures?|associates?|segments?|manufactur\w*|plants?|business|"
    r"product portfolio|"
    r"coal|lignite|washery|mining|mines?|mineral|ore|evacuation|auction|e-auction|fsa|"
    r"linkage|offtake|thermal|power|transmission|grid|substation|renewable|solar|wind|"
    r"gasification|coking|non-?coking|rail|corridor|logistics|port|jetty|"
    r"transmission lines?|substations?|sub-stations?|ckt-km|ckt\s*km|mva|hvdc|"
    r"telecom|consultancy|spv|tbcb|jv"
)

NEGATIVE_TERMS = (
    r"auditor|key audit matter|\bkam\b|accounting polic|corporate social responsibility|\bcsr\b|"
    r"related party|remuneration|income tax|financial instrument"
)

ESG_HR_TERMS = (
    r"emissions?|ghg|biodiversity|grievance|training|employees?|safety|engagement|waste|water|"
    r"carbon|decarbonis\w*|net[- ]zero"
)

PRIORITY_TERMS = (
    r"subsidiar|reportable segment|product portfolio|\bcapacity\b|joint venture|"
    r"e-auction|washery|coal gasification|segment information|"
    r"transmission lines?|substations?|hvdc|tbcb|spv|consultancy"
)

CAPACITY_PATTERN = r"\b\d+(?:\.\d+)?\s*(?:KTPA|MTPA|MT|MTY|MW|tonnes?)\b"
PRODUCT_PORTFOLIO_PATTERN = r"product portfolio"
REVENUE_PATTERN = (
    r"reportable segment|revenue from|segment revenue|aoc-?\s*1|subsidiar|joint venture|"
    r"e-auction|fuel supply"
)
CSR_SCOPE_PATTERN = (
    r"\b(hospitals?|schools?|community|foundation|scholarships?|health camps?|skills?|"
    r"hostels?|anganwadi|villages?)\b"
)

# Cover / index / TOC-like labels used when deciding not to keep recovered front pages.
NAV_LABELS = frozenset({"TOC", "COVER", "INDEX"})

# CSR spend / policy boilerplate. A page is boilerplate only when these dominate and R&D is absent.
CSR_SPEND_PATTERN = (
    r"csr expenditure|amount spent on csr|prescribed csr|unspent csr|"
    r"csr obligation|average net profit|section\s*135|csr policy|"
    r"annual report on csr|increase in csr expenses"
)
RD_PATTERN = r"research\s*(?:&|and)\s*development|\br\s*&\s*d\b|\bresearch\b"

# OCR text below either floor is not sent to the LLM.
OCR_LATIN_MIN = 0.55
OCR_DICT_MIN = 0.08

# Notice, proxy, and e-voting language. A page is governance from its own text.
GOVERNANCE_NOTICE_PATTERN = (
    r"notice of (?:the )?.{0,80}annual general meeting"
    r"|annual general meeting"
    r"|\bagm\b"
    r"|e-?\s*voting"
    r"|remote e-?\s*voting"
    r"|\bproxy\b"
    r"|explanatory statement"
    r"|resolved further that"
    r"|recommends passing of the resolution"
    r"|\bDIN\s*:?\s*\d"
    r"|date of appointment"
    r"|brief resume"
    r"|expertise in specific functional area"
)

# Minimum recovered text length before a front-matter gap can be overridden.
FRONT_RECOVERED_CHARS = 80
# Score floor for keeping recovered front pages when labels are ambiguous.
FRONT_KEEP_SCORE = 0


def compile_positive() -> re.Pattern[str]:
    return re.compile(rf"\b({POSITIVE_TERMS})\b", re.I)


def compile_negative() -> re.Pattern[str]:
    return re.compile(rf"auditor|key audit matter|\bkam\b|accounting polic|corporate social responsibility|\bcsr\b|"
                      r"related party|remuneration|income tax|financial instrument", re.I)


def compile_esg_hr() -> re.Pattern[str]:
    return re.compile(rf"\b({ESG_HR_TERMS})\b", re.I)


def compile_priority() -> re.Pattern[str]:
    return re.compile(PRIORITY_TERMS, re.I)
