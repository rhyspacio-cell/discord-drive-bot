"""Deterministic aggregation for Coordination Letter date-list questions."""

import re
from datetime import date

from modules.drive import is_list_aggregation_query


MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|"
    r"Sept|Oct|Nov|Dec"
)
DATE_PATTERN = (
    rf"(?:{MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+\d{{4}}"
    rf"|\d{{4}}-\d{{1,2}}-\d{{1,2}}"
)
PLANNED_ACTIVITY_PATTERN = re.compile(
    rf"\bthat\s+(?P<company>.+?)\s+will\s+be\s+"
    rf"(?P<activity>[^.\n]*?)\bon\s+(?P<date>{DATE_PATTERN})\b",
    re.IGNORECASE,
)
DATE_VALUE_PATTERN = re.compile(
    rf"(?P<month>{MONTH_PATTERN})\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s+"
    r"(?P<year>\d{4})",
    re.IGNORECASE,
)


def is_coordination_letter_aggregation(question, search_plan):
    """Return whether an explicit-range aggregation targets Coordination Letters."""
    if not is_list_aggregation_query(search_plan):
        return False
    if re.search(
        r"\bactual(?:ly)?\b.*\b(?:entry|enter(?:ed)?)\b",
        str(question),
        re.IGNORECASE,
    ):
        return False

    plan_terms = []
    for key in (
        "required_terms",
        "phrases",
        "optional_terms",
        "context_terms",
    ):
        values = search_plan.get(key, [])
        if isinstance(values, list):
            plan_terms.extend(str(value) for value in values)

    answer_context = " ".join(
        [
            str(question),
            str(search_plan.get("intent", "")),
            str(search_plan.get("answer_type", "")),
            *plan_terms,
        ]
    )
    return re.search(
        r"\bcoordination\s+letters?\b",
        answer_context,
        re.IGNORECASE,
    ) is not None


def parse_planned_date(value):
    """Parse an explicit month-name or ISO planned-activity date."""
    date_match = DATE_VALUE_PATTERN.fullmatch(value.strip())

    if date_match:
        month = date_match.group("month")
        day = int(date_match.group("day"))
        year = int(date_match.group("year"))
        month_names = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }
        try:
            return date(year, month_names[month.casefold()], day)
        except ValueError:
            return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _date_bound(search_plan, key, question):
    value = (search_plan.get("time_range") or {}).get(key)
    if value is None:
        if key == "to" and re.search(
            r"\bto\s+present\b",
            str(question),
            re.IGNORECASE,
        ):
            return date.today()
        return None

    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def extract_coordination_letter_records(question, documents, search_plan):
    """Extract evidence-backed company/planned-date records for this query type."""
    if not is_coordination_letter_aggregation(question, search_plan):
        return None

    lower_bound = _date_bound(search_plan, "from", question)
    upper_bound = _date_bound(search_plan, "to", question)
    records = []
    unresolved = []

    for document in documents:
        name = str(document.get("name", "Unnamed file"))
        text = document.get("text", "")
        if not isinstance(text, str):
            text = ""

        if not re.search(
            r"\bcoordination\s+letters?\b",
            f"{name}\n{text[:500]}",
            re.IGNORECASE,
        ):
            continue

        matches = list(PLANNED_ACTIVITY_PATTERN.finditer(text))
        if not matches:
            unresolved.append(name)
            continue

        found_supported_event = False
        for match in matches:
            company = re.sub(
                r"\s+",
                " ",
                match.group("company"),
            ).strip().rstrip(".").strip()
            planned_date = parse_planned_date(match.group("date"))
            if not company or planned_date is None:
                unresolved.append(name)
                continue

            found_supported_event = True
            if lower_bound and planned_date < lower_bound:
                continue
            if upper_bound and planned_date > upper_bound:
                continue

            records.append(
                {
                    "company": company,
                    "date": planned_date,
                    "source": name,
                    "evidence": match.group(0),
                }
            )

        if not found_supported_event and not any(
            unresolved_name == name
            for unresolved_name in unresolved
        ):
            unresolved.append(name)

    return {
        "records": records,
        "unresolved": unresolved,
    }


def format_coordination_letter_records(aggregation):
    """Render the authoritative aggregation without asking the LLM to select records."""
    grouped_records = {}
    display_names = {}

    for record in aggregation["records"]:
        company_key = record["company"].casefold()
        display_names.setdefault(company_key, record["company"])
        grouped_records.setdefault(company_key, []).append(record)

    lines = ["Companies and planned activity dates from the retrieved Coordination Letters:"]

    if grouped_records:
        for company_key, company_records in grouped_records.items():
            company_records.sort(key=lambda record: record["date"])
            date_values = [
                f"{record['date'].strftime('%B')} {record['date'].day}, "
                f"{record['date'].year} ({record['source']})"
                for record in company_records
            ]
            lines.append(
                f"- {display_names[company_key]}: "
                + "; ".join(date_values)
            )
    else:
        lines.append("- No company/planned activity date could be established from the supplied letters.")

    if aggregation["unresolved"]:
        lines.append("")
        lines.append("Unresolved Coordination Letters:")
        lines.extend(
            f"- {name}: company or planned activity date not established from the extracted text."
            for name in aggregation["unresolved"]
        )

    return "\n".join(lines)