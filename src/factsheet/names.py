"""Assignment name key. Equal keys merge. A proper subset does not."""

from __future__ import annotations

import re

from src.config.factsheet_keywords import TOPIC_ALIASES, TOPIC_GENERIC

_KEEP_ENDING = {"business", "gas", "glass", "analysis", "basis"}
_COMPANY_SUBJECT = re.compile(r"^\s*(?:the\s+company|our\s+company|company)\b", re.I)
_COMPANY_ENTITY = {"company", "the company", "parent"}


def assignment_key(name: str) -> str:
    """Fold &/and, Ltd/Limited/Pvt/Private, plurals, and word order.

    The key is the full token set. "Novelis" and "Novelis UK" stay distinct.
    """
    raw = (name or "").casefold().replace("&", " and ")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    tokens = []
    for token in raw.split():
        token = {"ltd": "limited", "pvt": "private"}.get(token, token)
        if token in {"limited", "private"}:
            continue
        token = _singular(token)
        if token:
            tokens.append(token)
    return " ".join(sorted(tokens))


def same_item(left: str, right: str) -> bool:
    key = assignment_key(left)
    return bool(key) and key == assignment_key(right)


def normalized_phrase(name: str) -> str:
    """Word order kept, for edit distance. Parenthetical acronyms are removed."""
    text = (name or "").casefold().replace("&", " and ")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for token in text.split():
        token = {"ltd": "limited", "pvt": "private"}.get(token, token)
        if token in {"limited", "private"}:
            continue
        token = _singular(token)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def proper_subset(left: str, right: str) -> bool:
    """True when one name's tokens sit inside the other. Those stay separate."""
    left_tokens = set(normalized_phrase(left).split())
    right_tokens = set(normalized_phrase(right).split())
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    return left_tokens < right_tokens or right_tokens < left_tokens


def should_merge(left: str, right: str, list_keys: set[str] | None = None) -> bool:
    """Equal names, one spelling variant, or the same topic head.

    A proper subset stays separate. Two list rows stay separate. Names that
    differ by one capitalized word (Eastern and Western) stay separate.
    """
    if proper_subset(left, right) or distinct_proper_conflict(left, right):
        return False
    if list_keys:
        left_key, right_key = assignment_key(left), assignment_key(right)
        if left_key != right_key and (left_key in list_keys or right_key in list_keys):
            return False
    if same_item(left, right) or spelling_variant(left, right):
        return True
    return topic_match(left, right)


_TOPIC_NOISE = {
    "phase", "project", "plant", "system", "center", "centre", "collaboration",
    "mou", "machine", "testing", "based", "largest", "started", "excellence",
    "joint", "venture", "scale", "non", "limited", "private", "the", "of",
    "for", "and", "with", "on", "in", "at", "to", "a", "an",
}


def _acronym_keys(name: str) -> set[str]:
    keys = set()
    for acronym in re.findall(r"\(([A-Z][A-Z0-9]{3,})\)", name or ""):
        keys.add(acronym.casefold())
    for token in re.findall(r"\b([A-Za-z][A-Za-z0-9]{3,})\b", name or ""):
        uppers = sum(1 for char in token if char.isupper())
        if uppers >= 3 and len(token) >= 4:
            keys.add(token.casefold())
    return keys


def head_phrase(name: str) -> str:
    """Leading content words, in order, with trailing generic words removed."""
    tokens = []
    for token in normalized_phrase(name).split():
        if token in _TOPIC_NOISE or token in _COMPARE_STOP:
            continue
        if token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        tokens.append(token)
    while tokens and tokens[-1] in TOPIC_GENERIC:
        tokens.pop()
    return " ".join(tokens[:3])


def topic_keys(name: str) -> set[str]:
    """Acronyms, the leading head phrase, and generic topic aliases. Not a company list."""
    keys = {"acr:" + acronym for acronym in _acronym_keys(name)}
    phrase = head_phrase(name)
    parts = phrase.split()
    if len(parts) >= 2 and any(len(part) >= 4 and part not in TOPIC_GENERIC and not part.isdigit() for part in parts):
        keys.add("head:" + phrase)
    for pattern, canon in TOPIC_ALIASES:
        if pattern.search(name or ""):
            keys.add("alias:" + canon)
    return keys


def topic_match(left: str, right: str) -> bool:
    """True when both names share an acronym, an alias, or the same leading head phrase."""
    if not left or not right or distinct_proper_conflict(left, right) or proper_subset(left, right):
        return False
    shared = topic_keys(left) & topic_keys(right)
    return any(key.startswith(("acr:", "alias:", "head:")) for key in shared)


_COMPARE_STOP = {"of", "for", "and", "the", "with", "a", "an", "to", "in", "on"}


def _content_tokens(name: str) -> list[str]:
    return [token for token in normalized_phrase(name).split() if token not in _COMPARE_STOP]


def spelling_variant(left: str, right: str) -> bool:
    """True when the names differ by one near-typo token, such as Seva and Sewa.

    A different first letter is a different word (Eastern and Western), not a typo.
    """
    left_tokens = set(_content_tokens(left))
    right_tokens = set(_content_tokens(right))
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    only_left = left_tokens - right_tokens
    only_right = right_tokens - left_tokens
    if len(only_left) != 1 or len(only_right) != 1:
        return False
    from rapidfuzz.distance import Levenshtein

    first, second = next(iter(only_left)), next(iter(only_right))
    if min(len(first), len(second)) < 4 or first[0] != second[0]:
        return False
    return Levenshtein.distance(first, second) <= 2


def distinct_proper_conflict(left: str, right: str) -> bool:
    """Names that differ by one capitalized word and are not a spelling variant."""
    if same_item(left, right) or spelling_variant(left, right) or proper_subset(left, right):
        return False
    left_only = set(_content_tokens(left)) - set(_content_tokens(right))
    right_only = set(_content_tokens(right)) - set(_content_tokens(left))
    if len(left_only) != 1 or len(right_only) != 1:
        return False
    left_word, right_word = next(iter(left_only)), next(iter(right_only))
    return _capitalized(left, left_word) and _capitalized(right, right_word)


def _capitalized(name: str, token: str) -> bool:
    for word in re.findall(r"[A-Za-z][A-Za-z'-]*", name or ""):
        folded = word.casefold()
        stem = folded[:-1] if folded.endswith("s") and not folded.endswith("ss") else folded
        if folded == token or stem == token:
            return word[:1].isupper()
    return False


def actor_entity(quote: str, entity: str, subsidiary_names: list[str]) -> str:
    """Use the subsidiary or JV named as the actor. Keep company when it is the subject."""
    current = (entity or "").strip()
    text = quote or ""
    if _COMPANY_SUBJECT.search(text) and current.casefold() in _COMPANY_ENTITY:
        return current or entity
    names = [item for item in subsidiary_names if item and len(item) >= 3]
    hits = [name for name in names if name.casefold() in text.casefold()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        for name in sorted(hits, key=len, reverse=True):
            if re.search(rf"\b(?:of|at|by|from|spv of)\s+{re.escape(name)}\b", text, re.I):
                return name
        return current or entity
    acronyms = {}
    for name in names:
        letters = "".join(token[0] for token in name.split() if token[:1].isalpha() and token.casefold() not in {"limited", "ltd", "private", "pvt", "of", "and", "the"})
        if len(letters) >= 2:
            acronyms.setdefault(letters.upper(), []).append(name)
    for match in re.finditer(r"\b(?:of|at|by|from)\s+(?:the\s+)?([A-Z]{2,6})\b", text):
        owners = acronyms.get(match.group(1).upper()) or []
        if len(owners) == 1:
            return match.group(1)
    actor_tokens = []
    for token, owners in acronyms.items():
        if len(owners) != 1:
            continue
        if re.search(
            rf"(?:^|[.]\s+){token}\b|\b{token}\s+(?:is|was|has|have|operates|operated|produced|developed|commissioned|started|owns|owned)\b",
            text,
        ):
            actor_tokens.append(token)
    if len(set(actor_tokens)) == 1:
        return actor_tokens[0]
    return current or entity


def _singular(token: str) -> str:
    if token in _KEEP_ENDING or len(token) <= 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
