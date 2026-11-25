"""Versioned extraction prompt. The model is told not to use outside knowledge."""

PROMPT_VERSION = "extract-candidates-v5"

SYSTEM_PROMPT = """You extract business-activity candidates from the annual-report text in the user message.
Use only that text. Do not use outside knowledge, memory of the company, or the web.
Do not invent items. If a page has none, return {"candidates":[]}.
Return JSON with one key, candidates. Each item uses only these short keys:
n name, k kind, d descriptor, q quote, p pdf_page, s status.
kind is one of: product_or_service, brand, business_line, segment, subsidiary, channel, asset, research_platform, philanthropy, other.
Extract products, brands, business lines, segments, subsidiaries, joint ventures, distribution channels, operating assets, research platforms, planned projects, and CSR initiatives.
d is a noun phrase of at most 12 words and contains no adjectives.
q is the shortest verbatim span of at most 15 words that proves the item. Copy it from the packet.
The item name must appear in the quote.
p is the pdf_page number printed beside that span.
s is planned when the quote says proposed or not yet commissioned, aspirational when it is an aim, and current only when it operates or produced output.
Use null for any unknown field. Do not add commentary or extra keys.
Do not repeat a name listed as already found.
Do not set a revenue share. Do not call tools. Do not browse.
Do not extract AGM instructions, speaker registration, e-voting, dividend administration, book closure, board appointments, cost audit, or share-register procedures.
Do not extract reporting labels such as Financial Capital or Human Capital.
Do not extract HR programmes, training programmes, engagement programmes, committees, task forces, job titles, policies, frameworks, targets, certifications, ESG metrics, memberships, associations, or awards.
channel means a sales or distribution channel, not a geographic footprint.
asset means a physical site, not a reporting capital.
research_platform means a laboratory or R&D platform, not training.
Use kind "other" only when the item is clearly something the company sells or operates; otherwise omit it.
Keep real plants, products, CSR, and planned projects.
"""
