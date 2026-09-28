from datetime import date, timedelta

from modules.coordination_aggregation import (
    extract_coordination_letter_records,
    format_coordination_letter_records,
)
from modules.llm import answer_drive_question


QUESTION = (
    "From January 2026 to present, which companies entered IPI and list their "
    "dates SEARCH: COORDINATION LETTERS"
)
SEARCH_PLAN = {
    "intent": "ipi coordination letters",
    "required_terms": ["coordination letter", "coordination letters"],
    "phrases": [],
    "optional_terms": ["companies", "ipi"],
    "context_terms": ["company", "dates"],
    "exclude_terms": [],
    "time_range": {"from": "2026-01-01", "to": None},
    "answer_type": "companies",
    "confidence": 0.8,
}


def coordination_letter(name, text):
    return {
        "name": name,
        "modifiedTime": "2026-09-29T00:00:00Z",
        "score": 35,
        "text": text,
    }


def test_june_17_uses_planned_date_not_submission_date():
    document = coordination_letter(
        "Coordination Letter 06-17-2026.docx",
        """COORDINATION LETTER
DATE SUBMITTED: JUNE 17, 2026
We would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the wastewater treatment facility area on JUNE 18, 2026.""",
    )

    result = extract_coordination_letter_records(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert result["records"] == [
        {
            "company": "OSTREA MINERAL LABORATORIES INC",
            "date": date(2026, 6, 18),
            "source": "Coordination Letter 06-17-2026.docx",
            "evidence": (
                "that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater "
                "at the wastewater treatment facility area on JUNE 18, 2026"
            ),
        }
    ]
    assert result["records"][0]["date"] != date(2026, 6, 17)
    assert "JUNE 18, 2026" in result["records"][0]["evidence"]
    assert "JUNE 17, 2026" not in result["records"][0]["evidence"]


def test_june_25_rescheduling_uses_new_planned_date():
    document = coordination_letter(
        "Coordination Letter 06-25-2026.docx",
        """COORDINATION LETTER
DATE SUBMITTED: JUNE 25, 2026
We would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the wastewater treatment facility area on JUNE 26, 2026. This activity is a rescheduling of the wastewater sampling that was originally scheduled on June 18, 2026.""",
    )

    result = extract_coordination_letter_records(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert [record["date"] for record in result["records"]] == [
        date(2026, 6, 26)
    ]
    assert "JUNE 26, 2026" in result["records"][0]["evidence"]
    assert "June 18, 2026" not in result["records"][0]["evidence"]
    assert "June 18, 2026" not in format_coordination_letter_records(result)


def test_multiple_ostrea_dates_and_companies_are_preserved():
    documents = [
        coordination_letter(
            f"Coordination Letter {label}.docx",
            f"COORDINATION LETTER\nWe would like to inform you that {company} will be sampling wastewater at the facility on {planned_date}.",
        )
        for label, company, planned_date in (
            ("01-26-2026", "OSTREA MINERAL LABORATORIES INC.", "JANUARY 26, 2026"),
            ("01-27-2026", "OSTREA MINERAL LABORATORIES INC.", "JANUARY 27, 2026"),
            ("03-05-2026", "OSTREA MINERAL LABORATORIES INC.", "MARCH 05, 2026"),
            ("06-18-2026", "OSTREA MINERAL LABORATORIES INC.", "JUNE 18, 2026"),
            ("06-26-2026", "OSTREA MINERAL LABORATORIES INC.", "JUNE 26, 2026"),
            ("02-04-2026", "Shimadzu Philippines Corporation", "FEBRUARY 04, 2026"),
        )
    ]

    result = extract_coordination_letter_records(
        QUESTION,
        documents,
        SEARCH_PLAN,
    )
    rendered = format_coordination_letter_records(result)

    ostrea_records = [
        record for record in result["records"]
        if record["company"].casefold().startswith("ostrea")
    ]
    assert [record["date"] for record in ostrea_records] == [
        date(2026, 1, 26),
        date(2026, 1, 27),
        date(2026, 3, 5),
        date(2026, 6, 18),
        date(2026, 6, 26),
    ]
    assert "Shimadzu Philippines Corporation: February 4, 2026" in rendered
    assert "OSTREA MINERAL LABORATORIES INC: January 26, 2026" in rendered
    assert "June 18, 2026" in rendered
    assert "June 26, 2026" in rendered


def test_letter_without_supported_planned_date_is_unresolved():
    document = coordination_letter(
        "Coordination Letter undated.docx",
        """COORDINATION LETTER
DATE SUBMITTED: JUNE 17, 2026
OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility.""",
    )

    result = extract_coordination_letter_records(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert result["records"] == []
    assert result["unresolved"] == ["Coordination Letter undated.docx"]
    assert "JUNE 17, 2026" not in format_coordination_letter_records(result)


def test_planned_dates_outside_requested_range_are_not_aggregated_or_unresolved():
    document = coordination_letter(
        "Coordination Letter 12-31-2025.docx",
        """COORDINATION LETTER
We would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility on DECEMBER 31, 2025.""",
    )

    result = extract_coordination_letter_records(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert result == {"records": [], "unresolved": []}


def test_to_present_excludes_future_planned_activity_dates():
    future_date = date.today() + timedelta(days=1)
    document = coordination_letter(
        "Coordination Letter future.docx",
        f"""COORDINATION LETTER
We would like to inform you that ESCO Life Sciences Group will be conducting a site visit at the laboratory on {future_date.strftime('%B')} {future_date.day}, {future_date.year}.""",
    )

    result = extract_coordination_letter_records(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert result == {"records": [], "unresolved": []}


def test_ordinary_question_keeps_llm_answer_path(monkeypatch):
    generated_prompts = []

    def fake_generate(prompt):
        generated_prompts.append(prompt)
        return "Existing LLM answer."

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        fake_generate,
    )
    ordinary_plan = {
        "intent": "research documents",
        "required_terms": ["research"],
        "phrases": [],
        "optional_terms": [],
        "context_terms": [],
        "exclude_terms": [],
        "answer_type": "document",
    }
    documents = [
        {
            "name": "Research notes.txt",
            "modifiedTime": "2026-09-29T00:00:00Z",
            "score": 10,
            "text": "Research details.",
        }
    ]

    answer = answer_drive_question(
        "Summarize my research notes",
        documents,
        ordinary_plan,
    )

    assert answer == "Existing LLM answer."
    assert len(generated_prompts) == 1


def test_actual_entry_date_question_does_not_use_planned_date_rule():
    actual_entry_question = (
        "List the actual date of entry from January 2026 to present "
        "SEARCH: COORDINATION LETTERS"
    )

    result = extract_coordination_letter_records(
        actual_entry_question,
        [
            coordination_letter(
                "Coordination Letter 06-17-2026.docx",
                """COORDINATION LETTER
DATE SUBMITTED: JUNE 17, 2026
We would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility on JUNE 18, 2026.""",
            )
        ],
        SEARCH_PLAN,
    )

    assert result is None


def test_coordination_aggregation_answer_does_not_depend_on_llm(monkeypatch):
    def unexpected_llm_call(_prompt):
        raise AssertionError("Coordination aggregation must be deterministic")

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        unexpected_llm_call,
    )
    documents = [
        coordination_letter(
            "Coordination Letter 06-17-2026.docx",
            """COORDINATION LETTER
DATE SUBMITTED: JUNE 17, 2026
We would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility on JUNE 18, 2026.""",
        ),
        coordination_letter(
            "Coordination Letter 06-25-2026.docx",
            """COORDINATION LETTER
DATE SUBMITTED: JUNE 25, 2026
We would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility on JUNE 26, 2026.""",
        ),
    ]

    answer = answer_drive_question(QUESTION, documents, SEARCH_PLAN)

    assert "OSTREA MINERAL LABORATORIES INC: June 18, 2026" in answer
    assert "June 26, 2026" in answer