import json
from datetime import date, timedelta

from modules.coordination_aggregation import (
    _coordination_event_matches_constraints,
    _explicit_company_scope,
    _planned_activity_matches,
    evaluate_specialized_handlers,
    extract_coordination_letter_records,
    format_coordination_letter_records,
)
from modules.llm import answer_drive_question
from bot import format_drive_answer


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


def test_coordination_keyword_later_in_text_is_still_accepted():
    long_preamble = "A" * 700
    document = coordination_letter(
        "Submission Memo.docx",
        f"""{long_preamble}
COORDINATION LETTER
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
            "source": "Submission Memo.docx",
            "evidence": (
                "that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater "
                "at the wastewater treatment facility area on JUNE 18, 2026"
            ),
        }
    ]


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


def test_coordination_letter_accepts_alternate_planned_activity_grammar():
    document = coordination_letter(
        "Coordination Letter 06-17-2026.docx",
        "COORDINATION LETTER\n"
        "OSTREA MINERAL LABORATORIES INC. will conduct wastewater sampling "
        "at the facility on JUNE 18, 2026.",
    )

    result = extract_coordination_letter_records(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["company"] == "OSTREA MINERAL LABORATORIES INC"
    assert record["date"] == date(2026, 6, 18)
    assert "will conduct wastewater sampling" in record["evidence"]


def test_coordination_event_requires_explicit_person_role_binding():
    constraints = {
        "subject_entity": "John Smith",
        "subject_type": "person",
    }
    unrelated_text = (
        "COORDINATION LETTER\n"
        "John Smith was mentioned in an unrelated memo.\n"
        "We would like to inform your office that Alpha Marine Services will be "
        "conducting an equipment inspection on October 10, 2026."
    )
    contact_text = (
        "COORDINATION LETTER\n"
        "Contact: John Smith.\n"
        "We would like to inform your office that Alpha Marine Services will be "
        "conducting an equipment inspection on October 10, 2026."
    )
    unrelated_match = _planned_activity_matches(unrelated_text)[0]
    contact_match = _planned_activity_matches(contact_text)[0]

    assert not _coordination_event_matches_constraints(
        unrelated_match,
        unrelated_text,
        constraints,
    )
    assert _coordination_event_matches_constraints(
        contact_match,
        contact_text,
        constraints,
    )


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

    assert "traceable evidence" in answer.lower()
    assert len(generated_prompts) == 1


def test_coordination_handler_requires_true_record_evidence(monkeypatch):
    generated_prompts = []

    def fake_generate(prompt):
        generated_prompts.append(prompt)
        return "Generic evidence answer."

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        fake_generate,
    )

    document = coordination_letter(
        "Coordination Letter 06-17-2026.docx",
        """COORDINATION LETTER
This is a general company coordination memo. No planned activity date or company record is stated here.""",
    )

    answer = answer_drive_question(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert "couldn't find evidence" in answer.lower()
    assert generated_prompts == []


def test_coordination_filename_only_candidate_never_reaches_answer_llm(monkeypatch):
    def unexpected_generate(_prompt):
        raise AssertionError("Filename-only evidence must not reach the answer LLM")

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        unexpected_generate,
    )

    document = {
        "name": "Coordination Letter 06-17-2026.docx",
        "modifiedTime": "2026-09-29T00:00:00Z",
        "score": 40,
        "text": "This is a generic memo about coordination with no company or scheduled activity date.",
    }

    answer = answer_drive_question(
        QUESTION,
        [document],
        SEARCH_PLAN,
    )

    assert "couldn't find evidence" in answer.lower()


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


def test_date_to_company_lookup_uses_planned_activity_date():
    question = (
        "Which company is scheduled for June 18, 2026? "
        "SEARCH: COORDINATION LETTERS"
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
We would like to inform you that GENNEX will be sampling wastewater at the facility on JUNE 26, 2026.""",
        ),
    ]
    search_plan = {
        "intent": "coordination letters",
        "required_terms": ["coordination letter"],
        "phrases": [],
        "optional_terms": ["company", "date"],
        "context_terms": ["scheduled"],
        "exclude_terms": [],
        "time_range": {"from": None, "to": None},
        "answer_type": "company",
        "confidence": 0.9,
    }

    result = extract_coordination_letter_records(
        question,
        documents,
        search_plan,
    )

    assert [(record["company"], record["date"]) for record in result["records"]] == [
        ("OSTREA MINERAL LABORATORIES INC", date(2026, 6, 18))
    ]
    assert "GENNEX" not in format_coordination_letter_records(result)


def test_exact_document_date_constraint_selects_matching_coordination_letter():
    question = (
        "What planned activity or sampling date is stated in the Coordination Letter "
        "submitted on June 25, 2026?"
    )
    search_plan = {
        "intent": "coordination letters",
        "required_terms": ["coordination letter"],
        "phrases": [],
        "optional_terms": ["date"],
        "context_terms": ["planned activity", "sampling"],
        "exclude_terms": [],
        "time_range": {"from": None, "to": None},
        "answer_type": "date",
        "confidence": 0.9,
        "document_dates": ["2026-06-25"],
    }
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
We would like to inform you that GENNEX will be sampling wastewater at the facility on JUNE 26, 2026.""",
        ),
    ]

    result = extract_coordination_letter_records(
        question,
        documents,
        search_plan,
    )

    assert [(record["company"], record["date"]) for record in result["records"]] == [
        ("GENNEX", date(2026, 6, 26))
    ]
    assert "OSTREA" not in format_coordination_letter_records(result)


def test_date_not_present_does_not_associate_company():
    question = (
        "Which company is associated with the March 5, 2026 planned activity?"
    )
    documents = [
        coordination_letter(
            "Coordination Letter 03-04-2026.docx",
            """COORDINATION LETTER
DATE SUBMITTED: MARCH 04, 2026
We would like to inform you that ALPHA INDUSTRIES will be sampling wastewater at the facility on MARCH 04, 2026.""",
        ),
        coordination_letter(
            "Coordination Letter 03-06-2026.docx",
            """COORDINATION LETTER
DATE SUBMITTED: MARCH 06, 2026
We would like to inform you that BETA INDUSTRIES will be sampling wastewater at the facility on MARCH 06, 2026.""",
        ),
    ]

    result = extract_coordination_letter_records(
        question,
        documents,
        {
            "intent": "coordination letters",
            "required_terms": ["coordination letter"],
            "phrases": [],
            "optional_terms": ["company"],
            "context_terms": ["planned activity"],
            "exclude_terms": [],
            "time_range": {"from": None, "to": None},
            "answer_type": "company",
            "confidence": 0.9,
            "activity_dates": ["2026-03-05"],
        },
    )

    assert result == {"records": [], "unresolved": []}


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


def test_ostrea_company_scope_excludes_other_companies():
    ostrea_question = (
        "For OSTREA, what are all the planned activity or sampling dates "
        "mentioned in the Coordination Letters from January 2026 to present? "
        "Include every applicable date, including multiple dates for the same company."
    )
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
            ("04-14-2026", "GENNEX", "APRIL 16, 2026"),
            ("08-05-2026", "KUBOTA Water and Environment Philippines Corporation", "AUGUST 07, 2026"),
        )
    ]

    result = extract_coordination_letter_records(
        ostrea_question,
        documents,
        SEARCH_PLAN,
    )

    assert [record["date"] for record in result["records"]] == [
        date(2026, 1, 26),
        date(2026, 1, 27),
        date(2026, 3, 5),
        date(2026, 6, 18),
        date(2026, 6, 26),
    ]
    assert all(
        record["company"].startswith("OSTREA")
        for record in result["records"]
    )
    assert "GENNEX" not in format_coordination_letter_records(result)
    assert "KUBOTA" not in format_coordination_letter_records(result)


def test_company_scope_is_generic_for_non_ostrea_company():
    gennex_question = (
        "For GENNEX, list all planned activity dates in Coordination Letters "
        "from January 2026 to present."
    )
    documents = [
        coordination_letter(
            "Coordination Letter 04-14-2026.docx",
            "COORDINATION LETTER\nWe would like to inform you that GENNEX. will be presenting their company profile at the EHS OFFICE on APRIL 16, 2026.",
        ),
        coordination_letter(
            "Coordination Letter 06-17-2026.docx",
            "COORDINATION LETTER\nWe would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility on JUNE 18, 2026.",
        ),
    ]

    result = extract_coordination_letter_records(
        gennex_question,
        documents,
        SEARCH_PLAN,
    )

    assert [(record["company"], record["date"]) for record in result["records"]] == [
        ("GENNEX", date(2026, 4, 16))
    ]


def test_coordination_company_scope_does_not_reclassify_all_caps_person_names():
    assert _explicit_company_scope(
        "For OSTREA, list all planned activity dates in Coordination Letters."
    ) == "OSTREA"
    assert _explicit_company_scope(
        "For JOHN SMITH, list all planned activity dates in Coordination Letters."
    ) is None


def test_person_constraint_limits_coordination_letter_records():
    question = (
        "What is the planned activity date in the Coordination Letter "
        "for John Smith?"
    )
    plan = {
        "intent": "coordination letter planned activity date",
        "required_terms": ["coordination letter"],
        "phrases": [],
        "optional_terms": [],
        "context_terms": ["planned activity date"],
        "exclude_terms": [],
        "time_range": {"from": None, "to": None},
        "answer_type": "date",
        "confidence": 0.9,
    }
    documents = [
        coordination_letter(
            "Coordination_Letter_John_Smith.docx",
            """COORDINATION LETTER
Contact: John Smith.
We would like to inform you that Alpha Marine Services will be conducting an equipment inspection at the site on OCTOBER 10, 2026.""",
        ),
        coordination_letter(
            "Coordination_Letter_Jane_Smith.docx",
            """COORDINATION LETTER
Contact: Jane Smith.
We would like to inform you that Beta Engineering will be conducting an equipment inspection at the site on NOVEMBER 12, 2026.""",
        ),
    ]

    result = extract_coordination_letter_records(question, documents, plan)

    assert [(record["company"], record["date"]) for record in result["records"]] == [
        ("Alpha Marine Services", date(2026, 10, 10))
    ]


def test_coordination_filename_does_not_prove_document_type():
    question = (
        "What is the planned activity date in the Coordination Letter "
        "for John Smith?"
    )
    document = coordination_letter(
        "Coordination_Letter_John_Smith.docx",
        "John Smith's equipment inspection is scheduled for October 10, 2026.",
    )

    assert evaluate_specialized_handlers(question, [document], SEARCH_PLAN) is None


def test_unrestricted_primary_aggregation_preserves_all_ten_records():
    documents = [
        coordination_letter(
            f"Coordination Letter {label}.docx",
            f"COORDINATION LETTER\nWe would like to inform your office that {company} will be conducting or sampling at the facility on {planned_date}.",
        )
        for label, company, planned_date in (
            ("01-05-2026", "METTLER TOLEDO PHILIPPINES INC.", "JANUARY 06, 2026"),
            ("01-26-2026", "OSTREA MINERAL LABORATORIES INC.", "JANUARY 26, 2026"),
            ("01-27-2026", "OSTREA MINERAL LABORATORIES INC.", "JANUARY 27, 2026"),
            ("02-03-2026", "Shimadzu Philippines Corporation", "FEBRUARY 04, 2026"),
            ("03-04-2026", "OSTREA MINERAL LABORATORIES INC.", "MARCH 05, 2026"),
            ("04-08-2026", "KRYPTON INTERNATIONAL RESOURCES INC.", "APRIL 08, 2026"),
            ("04-14-2026", "GENNEX", "APRIL 16, 2026"),
            ("06-17-2026", "OSTREA MINERAL LABORATORIES INC.", "JUNE 18, 2026"),
            ("06-25-2026", "OSTREA MINERAL LABORATORIES INC.", "JUNE 26, 2026"),
            ("08-05-2026", "KUBOTA Water and Environment Philippines Corporation", "AUGUST 07, 2026"),
        )
    ]

    result = extract_coordination_letter_records(
        QUESTION,
        documents,
        SEARCH_PLAN,
    )

    assert len(result["records"]) == 10
    assert {record["date"] for record in result["records"]} == {
        date(2026, 1, 6),
        date(2026, 1, 26),
        date(2026, 1, 27),
        date(2026, 2, 4),
        date(2026, 3, 5),
        date(2026, 4, 8),
        date(2026, 4, 16),
        date(2026, 6, 18),
        date(2026, 6, 26),
        date(2026, 8, 7),
    }
    assert "requested_company" not in result


def test_deterministic_references_only_include_record_sources(monkeypatch):
    def unexpected_llm_call(_prompt):
        raise AssertionError("Deterministic aggregation must not call the LLM")

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        unexpected_llm_call,
    )
    source_document = coordination_letter(
        "Coordination Letter 06-17-2026.docx",
        "COORDINATION LETTER\nWe would like to inform you that OSTREA MINERAL LABORATORIES INC. will be sampling wastewater at the facility on JUNE 18, 2026.",
    )
    source_document["candidate_id"] = "coord-source-1"
    no_record_document = coordination_letter(
        "Coordination Letter undated.docx",
        "COORDINATION LETTER\nNo company or planned activity date is stated.",
    )
    search_audit = {
        "used": [source_document["name"], no_record_document["name"], "PACIO_RESUME"],
        "analyzed": [source_document["name"], no_record_document["name"], "PACIO_RESUME"],
        "skipped": [],
        "failed": [],
        "empty": [],
    }

    answer = answer_drive_question(
        QUESTION,
        [
            source_document,
            no_record_document,
            coordination_letter("PACIO_RESUME", "Resume content."),
        ],
        SEARCH_PLAN,
        search_audit,
    )
    displayed = format_drive_answer(answer, search_audit)

    assert search_audit["used"] == [source_document["name"]]
    references_section = displayed.split(
        "**Files analyzed but not used**"
    )[0]
    assert f"`{source_document['name']}`" in references_section
    assert f"`{no_record_document['name']}`" not in references_section
    assert "`PACIO_RESUME`" not in references_section
    assert f"`{no_record_document['name']}`" in displayed
    assert "`PACIO_RESUME`" in displayed


def test_reference_firewall_discards_unvalidated_names_from_audit():
    search_audit = {
        "used": ["PACIO_RESUME", "Evidence.docx"],
        "analyzed": ["PACIO_RESUME", "Evidence.docx"],
        "validated_evidence_document_ids": ["evidence-1"],
        "validated_evidence_spans": [
            {"document_id": "evidence-1", "evidence_span": "A verified claim span."}
        ],
        "reference_evidence": [
            {
                "document_id": "evidence-1",
                "file_name": "Evidence.docx",
                "evidence_span": "fabricated unsupported quote",
                "claim": "Answer.",
                "supports_final_claim": True,
            },
            {
                "document_id": "resume-1",
                "file_name": "PACIO_RESUME",
                "evidence_span": "John Smith's resume text",
                "claim": "Answer.",
                "supports_final_claim": True,
            },
        ],
        "assessment_records": [
            {"file_name": "Evidence.docx", "eligible": True},
            {"file_name": "PACIO_RESUME", "eligible": False},
        ],
        "skipped": [],
        "failed": [],
        "empty": [],
    }

    displayed = format_drive_answer("Answer.", search_audit)

    assert "**Reference files used**" not in displayed
    assert "`PACIO_RESUME`" in displayed
    assert "`Evidence.docx`" in displayed
    assert search_audit["reference_firewall_rejected"] == 2


def test_legacy_used_names_do_not_become_references(monkeypatch):
    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        lambda _prompt: "Ordinary answer.",
    )
    search_audit = {
        "used": ["Coordination Letter 06-25-2026.docx", "PACIO_RESUME"],
        "analyzed": ["Coordination Letter 06-25-2026.docx", "PACIO_RESUME"],
        "skipped": [],
        "failed": [],
        "empty": [],
    }
    answer = answer_drive_question(
        "What is in this file?",
        [
            {
                "name": "Notes.txt",
                "modifiedTime": "2026-09-29T00:00:00Z",
                "score": 1,
                "text": "Ordinary document text.",
            }
        ],
        {"intent": "document", "answer_type": "document"},
        search_audit,
    )
    displayed = format_drive_answer(answer, search_audit)

    assert search_audit["used"] == []
    assert "**Reference files used**" not in displayed
    assert "`Coordination Letter 06-25-2026.docx`" in displayed
    assert "`PACIO_RESUME`" in displayed


def test_pacio_resume_cannot_be_reference_without_claim_support(monkeypatch):
    question = "What is the planned activity date for John Smith?"
    plan = {
        "query_constraints": {
            "subject_entity": "John Smith",
            "subject_type": "person",
            "activity": "equipment inspection",
            "requested_field": "planned_activity_date",
            "document_type": None,
        }
    }
    evidence_span = (
        "Person: John Smith\n"
        "Activity: Equipment Inspection\n"
        "Activity Date: October 10, 2026"
    )
    claim = "John Smith's planned equipment inspection date is October 10, 2026."
    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        lambda _prompt: json.dumps({
            "answer": claim,
            "claims": [{
                "claim": claim,
                "evidence": [{"document_id": "activity-1", "quote": evidence_span}],
            }],
        }),
    )
    candidates = [
        {
            "name": "activity_record.docx",
            "candidate_id": "activity-1",
            "modifiedTime": "2026-09-29T00:00:00Z",
            "text": evidence_span,
        },
        {
            "name": "PACIO_RESUME",
            "candidate_id": "resume-1",
            "modifiedTime": "2026-09-29T00:00:00Z",
            "text": "John Smith's biography mentions a project and an inspection career.",
        },
    ]
    audit = {
        "used": ["PACIO_RESUME"],
        "analyzed": [document["name"] for document in candidates],
        "skipped": [],
        "failed": [],
        "empty": [],
    }

    answer = answer_drive_question(question, candidates, plan, audit)
    displayed = format_drive_answer(answer, audit)
    references = displayed.split("**Files analyzed but not used**")[0]

    assert "`activity_record.docx`" in references
    assert "`PACIO_RESUME`" not in references
    assert "`PACIO_RESUME`" in displayed
    assert audit["used"] == ["activity_record.docx"]
    assert audit["assessments"]["PACIO_RESUME"]["eligible"] is False