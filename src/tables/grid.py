"""Grid cleanup, headers, and continuation links.

Continuation links tables. It does not copy rows onto another page.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_NUMBER = re.compile(r"^[\(\)\-–—]?\d[\d,\.\s%]*$")
_TOKEN = re.compile(r"[a-z]{3,}")


def clean_cell(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_numeric_cell(value: str) -> bool:
    text = clean_cell(value).replace(" ", "")
    if not text or text in {"-", "–", "—", "nil", "n.a.", "na", "n.a"}:
        return True
    return bool(_NUMBER.match(text.replace("%", "")))


def drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    keep = [index for index in range(width) if any(row[index].strip() for row in padded)]
    if not keep:
        return []
    return [[row[index] for index in keep] for row in padded]


def drop_empty_rows(rows: list[list[str]]) -> list[list[str]]:
    return [row for row in rows if any(cell.strip() for cell in row)]


def prepare_grid(rows: list[list]) -> list[list[str]]:
    cleaned = [[clean_cell(cell) for cell in row] for row in rows]
    return drop_empty_columns(drop_empty_rows(cleaned))


def _is_year(value: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}", clean_cell(value)))


def _is_amount(value: str) -> bool:
    text = clean_cell(value).replace(",", "").replace(" ", "")
    if _is_year(text):
        return False
    return bool(re.fullmatch(r"[\(\)\-–—]?\d+(?:\.\d+)?%?", text))


def row_is_header(row: list[str]) -> bool:
    """A header can contain years. A row with amounts is body text."""
    filled = [cell for cell in row if cell.strip()]
    if len(filled) < 2 or any(_is_amount(cell) for cell in filled):
        return False
    labels = [cell for cell in filled if not is_numeric_cell(cell) and not _is_year(cell)]
    return len(labels) >= 1


def split_header(rows: list[list[str]], max_header_rows: int = 3) -> tuple[list[str], list[list[str]]]:
    """Take leading label rows as the header. Numeric rows stay in the body."""
    header_rows: list[list[str]] = []
    rest = list(rows)
    while rest and len(header_rows) < max_header_rows and row_is_header(rest[0]):
        header_rows.append(rest.pop(0))
    if not header_rows:
        return [], rest
    width = max(len(row) for row in header_rows)
    headers = []
    for index in range(width):
        parts = [row[index] for row in header_rows if index < len(row) and row[index].strip()]
        headers.append(" ".join(parts))
    body = [row + [""] * (width - len(row)) for row in rest]
    body = [row[:width] for row in body]
    return headers, body


def grid_score(rows: list[list[str]]) -> float:
    """Prefer a compact grid with stable columns and filled cells."""
    if len(rows) < 2:
        return 0.0
    width = max(len(row) for row in rows)
    if width < 2 or width > 24:
        return 0.0
    cells = len(rows) * width
    filled = sum(1 for row in rows for cell in row if cell.strip())
    if cells == 0 or filled / cells < 0.25:
        return 0.0
    consistent = sum(1 for row in rows if len(row) == width) / len(rows)
    size = min(len(rows), 40) / 40
    return round(0.45 * consistent + 0.35 * (filled / cells) + 0.20 * size, 4)


def header_tokens(headers: list[str]) -> set[str]:
    return set(_TOKEN.findall(" ".join(headers).casefold()))


def header_similarity(left: list[str], right: list[str]) -> float:
    """Header overlap. Column count alone is not enough."""
    if not left or not right:
        return 0.0
    if abs(len(left) - len(right)) > 1:
        return 0.0
    left_tokens = header_tokens(left)
    right_tokens = header_tokens(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return 0.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union)


def link_continuations(tables: list[dict], minimum: float = 0.6) -> list[dict]:
    """Point adjacent-page tables at each other when the headers match.

    Rows stay on the page they were read from.
    """
    for table in tables:
        table["continuation_from"] = None
        table["continuation_to"] = None
    pairs = []
    for earlier in tables:
        for later in tables:
            if earlier["ticker"] != later["ticker"]:
                continue
            if later["pdf_page"] != earlier["pdf_page"] + 1:
                continue
            score = header_similarity(earlier.get("headers") or [], later.get("headers") or [])
            if score >= minimum:
                pairs.append((score, earlier, later))
    pairs.sort(key=lambda item: item[0], reverse=True)
    used_from = set()
    used_to = set()
    for _score, earlier, later in pairs:
        if id(earlier) in used_from or id(later) in used_to:
            continue
        earlier["continuation_to"] = later["table_id"]
        later["continuation_from"] = earlier["table_id"]
        used_from.add(id(earlier))
        used_to.add(id(later))
    return tables


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[dict]]] = []
        self._table: list[list[dict]] | None = None
        self._row: list[dict] | None = None
        self._cell: dict | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {key: value for key, value in attrs}
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {
                "text": "",
                "rowspan": _span(attributes.get("rowspan")),
                "colspan": _span(attributes.get("colspan")),
            }

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def html_to_grid(html: str) -> list[list[str]]:
    """Expand rowspan and colspan so each body row has the same columns."""
    parser = _HtmlTableParser()
    parser.feed(html or "")
    grids = []
    for table in parser.tables:
        grids.append(_expand_spans(table))
    if not grids:
        return []
    return max(grids, key=lambda grid: sum(len(row) for row in grid))


def _expand_spans(table: list[list[dict]]) -> list[list[str]]:
    occupied: set[tuple[int, int]] = set()
    placed: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(table):
        column = 0
        for cell in row:
            while (row_index, column) in occupied:
                column += 1
            text = clean_cell(cell["text"])
            for row_offset in range(cell["rowspan"]):
                for column_offset in range(cell["colspan"]):
                    point = (row_index + row_offset, column + column_offset)
                    occupied.add(point)
                    placed[point] = text if row_offset == 0 and column_offset == 0 else ""
            column += cell["colspan"]
    if not placed:
        return []
    row_count = max(point[0] for point in placed) + 1
    column_count = max(point[1] for point in placed) + 1
    return [[placed.get((row, column), "") for column in range(column_count)] for row in range(row_count)]
