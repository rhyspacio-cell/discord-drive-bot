import json

import pytest

from modules.evidence import (
    EvidenceRejectionReason,
    EvidenceState,
    EvidenceValidationError,
    assess_candidate,
    validate_answer_evidence,
    validate_candidates,
)
from modules.llm import answer_drive_question, generate_answer_from_evidence
from modules.query_constraints import extract_query_constraints
from bot import format_drive_answer
from modules.evidence_aggregation import evaluate_specialized_handlers


def test_project_risk_assessments_rejects_vba_and_project_explorer_content(monkeypatch):
    question = "What are the risks identified for the project?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidates = [
        {
            "name": "project_risk_assessment.docx",
            "text": "Project Risk Assessment: The identified project risks are schedule delay and equipment failure.",
        },
        {
            "name": "3.3 VBA_Project_Guide.pdf.pdf",
            "text": "Excel VBA guide. Use the Project Explorer to inspect modules and macros.",
        },
        {
            "name": "10.3 MSDN - Use the Project Explorer.html",
            "text": "Documentation for the VBA Project Explorer window.",
        },
        {
            "name": "config-project.js",
            "text": "const project = createProject(); // configures an application project",
        },
    ]
    audit = {}

    result = validate_candidates(question, candidates, plan, audit)

    assert [doc["name"] for doc in result.validated_evidence.documents] == [
        "project_risk_assessment.docx"
    ]
    vba = result.assessments["3.3 VBA_Project_Guide.pdf.pdf"]
    assert not vba.eligible
    assert EvidenceRejectionReason.REQUESTED_FIELD_MISSING in vba.rejection_reasons
    assert EvidenceRejectionReason.RELATIONSHIP_UNSUPPORTED in vba.rejection_reasons
    assert audit["assessments"][vba.file_name]["eligible"] is False
    assert "project_risk_assessment.docx" not in str(
        [doc["name"] for doc in result.validated_evidence.documents[1:]]
    )


def test_project_risk_heading_and_bullet_section_are_coherent_evidence():
    question = "What are the risks identified for the project?"
    plan = {"query_constraints": extract_query_constraints(question)}
    document = {
        "name": "project_risk_assessment.docx",
        "text": "Project Risk Assessment\nIdentified risks:\n- Schedule delay\n- Equipment failure",
    }

    assessment = validate_candidates(question, [document], plan).assessments[
        document["name"]
    ]

    assert assessment.subject_match
    assert assessment.activity_match
    assert assessment.requested_field_match
    assert assessment.relationship_supported
    assert assessment.eligible


def test_structured_activity_date_is_valid_generic_evidence():
    question = "What is the planned activity date for John Smith?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidates = [
        {
            "name": "activity_record.docx",
            "text": "Person: John Smith\nActivity: Equipment Inspection\nActivity Date: October 10, 2026",
        },
        {
            "name": "authorization_test.docx",
            "text": "Maria Santos is authorized to represent John Smith. The inspection was discussed, but no planned date is stated.",
        },
    ]

    result = validate_candidates(question, candidates, plan)

    assert [doc["name"] for doc in result.validated_evidence.documents] == [
        "activity_record.docx"
    ]
    assessment = result.assessments["activity_record.docx"]
    assert assessment.subject_match
    assert assessment.requested_field_match
    assert assessment.relationship_supported
    assert assessment.eligible
    assert EvidenceRejectionReason.REQUESTED_FIELD_MISSING in result.assessments[
        "authorization_test.docx"
    ].rejection_reasons


def test_unseen_project_meeting_document_uses_generic_evidence_path():
    question = "What was discussed in the project meeting?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "project_status_report.docx",
        "text": "Project meeting notes: The project meeting discussed the equipment inspection schedule and delivery risks.",
    }

    result = validate_candidates(question, [candidate], plan)
    constraints = plan["query_constraints"]

    assert constraints["subject_entity"] == "project"
    assert constraints["subject_type"] == "project"
    assert constraints["activity"] == "project meeting"
    assert constraints["requested_field"] == "discussion"
    assert constraints["document_type"] is None
    assert result.assessments[candidate["name"]].eligible


def test_project_manager_report_binds_equipment_inspection_and_date():
    question = "What date is the equipment inspection planned for John Smith?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "project_status_report.docx",
        "text": (
            "PROJECT STATUS REPORT\n"
            "Project: Cebu Port Upgrade\n"
            "Project Manager: John Smith\n"
            "Company: Alpha Marine Services\n"
            "Planned Activity:\n"
            "The team plans to inspect the Cebu Port equipment on October 10, 2026."
        ),
    }

    assessment = validate_candidates(question, [candidate], plan).assessments[
        candidate["name"]
    ]

    assert assessment.subject_match
    assert assessment.activity_match
    assert assessment.requested_field_match
    assert assessment.relationship_supported
    assert assessment.eligible


def test_explicit_participant_role_binds_the_requested_activity_subject():
    question = "What is the planned activity date for John Smith?"
    document = {
        "name": "activity_record.docx",
        "text": (
            "Participant: John Smith\n"
            "Activity: Equipment Inspection\n"
            "Activity Date: October 10, 2026"
        ),
    }

    assessment = assess_candidate(
        question,
        document,
        {"query_constraints": extract_query_constraints(question)},
    )

    assert assessment.subject_match is True
    assert assessment.relationship_supported
    assert assessment.eligible


@pytest.mark.parametrize(
    "non_subject_role",
    [
        "Document Author",
        "Document Recipient",
        "Employer",
        "Related Entity",
        "Mentioned Entity",
        "Representative",
    ],
)
def test_non_subject_entity_roles_do_not_bind_activity_claims(non_subject_role):
    question = "What is the planned activity date for John Smith?"
    document = {
        "name": "activity_record.docx",
        "text": (
            f"{non_subject_role}: John Smith\n"
            "Participant: Maria Santos\n"
            "Activity: Equipment Inspection\n"
            "Activity Date: October 10, 2026"
        ),
    }

    assessment = assess_candidate(
        question,
        document,
        {"query_constraints": extract_query_constraints(question)},
    )

    assert assessment.subject_match is False
    assert not assessment.relationship_supported
    assert not assessment.eligible


def test_project_meeting_relationship_survives_repeated_subject_term():
    question = "What was discussed in the project meeting?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "project_status_report.docx",
        "text": "A coordination letter was discussed during the project meeting, but this document is a project status report.",
    }

    assessment = validate_candidates(question, [candidate], plan).assessments[
        candidate["name"]
    ]

    assert assessment.activity_match
    assert assessment.relationship_supported
    assert assessment.eligible


def test_filename_only_coordination_type_is_rejected():
    question = (
        "What is the planned activity date in the Coordination Letter "
        "for John Smith?"
    )
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "Coordination_Letter_John_Smith.docx",
        "text": "John Smith's equipment inspection is scheduled for October 10, 2026.",
    }

    assessment = validate_candidates(question, [candidate], plan).assessments[
        candidate["name"]
    ]

    assert not assessment.eligible
    assert assessment.filename_only_match
    assert EvidenceRejectionReason.DOCUMENT_TYPE_MISMATCH in assessment.rejection_reasons
    assert EvidenceRejectionReason.FILENAME_ONLY in assessment.rejection_reasons


def test_incidental_coordination_letter_mention_does_not_prove_document_type():
    question = (
        "What is the planned activity date in the Coordination Letter "
        "for John Smith?"
    )
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "project_status_report.docx",
        "text": (
            "PROJECT STATUS REPORT\n"
            "Project Manager: John Smith\n"
            "The planned equipment inspection is on October 10, 2026.\n"
            "A Coordination Letter was discussed during the meeting."
        ),
    }

    assessment = validate_candidates(question, [candidate], plan).assessments[
        candidate["name"]
    ]

    assert assessment.document_type_match is False
    assert not assessment.eligible
    assert EvidenceRejectionReason.DOCUMENT_TYPE_MISMATCH in assessment.rejection_reasons


def test_inline_company_representative_is_not_activity_subject():
    question = "What is the planned activity date for Maria Santos?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "coordination_letter_test.docx",
        "text": (
            "COORDINATION LETTER Date: September 15, 2026 To: Operations Manager "
            "Company: Alpha Marine Services This letter coordinates the planned "
            "activity for John Smith. Planned Activity Date: October 10, 2026 "
            "Location: Cebu Port Company Representative: Maria Santos."
        ),
    }

    assessment = validate_candidates(question, [candidate], plan).assessments[
        candidate["name"]
    ]

    assert not assessment.eligible
    assert assessment.subject_match is False
    assert EvidenceRejectionReason.SUBJECT_MISMATCH in assessment.rejection_reasons


def test_company_date_must_bind_to_requested_company():
    question = "What is the planned activity date for Alpha Marine Services?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "coordination_letter_test.docx",
        "text": (
            "PROJECT STATUS REPORT\n"
            "Project Manager: John Smith\n"
            "Company: Alpha Marine Services\n"
            "This letter coordinates the planned activity for John Smith.\n"
            "Planned Activity Date: October 10, 2026"
        ),
    }

    assessment = validate_candidates(question, [candidate], plan).assessments[
        candidate["name"]
    ]

    assert not assessment.eligible
    assert EvidenceRejectionReason.RELATIONSHIP_UNSUPPORTED in assessment.rejection_reasons


def test_raw_candidates_are_rejected_before_answer_generation():
    with pytest.raises(EvidenceValidationError):
        generate_answer_from_evidence(
            "What is the planned activity date for John Smith?",
            [{"name": "candidate.docx", "text": "John Smith planned activity date is October 10, 2026."}],
        )


def test_coordination_evidence_binds_person_and_excludes_other_letters():
    question = (
        "What is the planned activity date in the Coordination Letter "
        "for John Smith?"
    )
    plan = {"query_constraints": extract_query_constraints(question)}
    audit = {"analyzed": [], "used": []}
    candidates = [
        {
            "name": "Coordination_Letter_John_Smith.docx",
            "candidate_id": "coord-john",
            "modifiedTime": "2026-09-29",
            "text": "COORDINATION LETTER\nContact: John Smith.\nWe would like to inform you that Alpha Marine Services will be conducting an equipment inspection on October 10, 2026.",
        },
        {
            "name": "Coordination_Letter_Jane_Smith.docx",
            "candidate_id": "coord-jane",
            "modifiedTime": "2026-09-29",
            "text": "COORDINATION LETTER\nContact: Jane Smith.\nWe would like to inform you that Beta Engineering will be conducting an equipment inspection on November 12, 2026.",
        },
    ]
    from modules.llm import answer_drive_question
    answer = answer_drive_question(question, candidates, plan, audit)

    assert "Alpha Marine Services: October 10, 2026" in answer
    assert "Beta Engineering" not in answer
    assert audit["used"] == ["Coordination_Letter_John_Smith.docx"]


def test_unvalidated_candidates_cannot_reach_answer_generation():
    with pytest.raises(EvidenceValidationError):
        generate_answer_from_evidence("Question", [{"name": "candidate.docx", "text": "text"}])


def test_no_valid_evidence_is_not_reported_as_no_readable_files(monkeypatch):
    def unexpected_llm_call(_prompt):
        raise AssertionError("Rejected evidence must not reach the answer LLM")

    monkeypatch.setattr("modules.llm.generate_local_response", unexpected_llm_call)
    question = "What are the risks identified for the project?"
    plan = {"query_constraints": extract_query_constraints(question)}
    audit = {"analyzed": ["3.3 VBA_Project_Guide.pdf.pdf"], "used": []}
    candidate = {
        "name": "3.3 VBA_Project_Guide.pdf.pdf",
        "text": "Excel VBA guide. Use the Project Explorer to inspect modules.",
    }

    answer = answer_drive_question(question, [candidate], plan, audit)

    assert "readable files" in answer.lower()
    assert "sufficient evidence" in answer.lower()
    assert audit["state"] == EvidenceState.CANDIDATES_FOUND_BUT_NO_VALID_EVIDENCE.value


def test_no_candidates_and_extraction_failure_have_distinct_states():
    empty = validate_candidates("Summarize this", [])
    failed_audit = {"assessments": {}, "analyzed": ["unreadable.docx"], "failed": [
        {"name": "unreadable.docx", "reason": "read failure"}
    ]}
    validate_candidates("Summarize this", [], search_audit=failed_audit)

    assert empty.state is EvidenceState.NO_CANDIDATES
    assert failed_audit["state"] == EvidenceState.EXTRACTION_FAILURE.value


def test_assessment_diagnostics_have_stable_rejection_codes():
    question = "When is John Smith's equipment inspection?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidate = {
        "name": "OSH_Inspection.pdf",
        "text": "Jane Smith's safety inspection is scheduled for October 10, 2026.",
    }

    result = validate_candidates(question, [candidate], plan)
    assessment = result.assessments[candidate["name"]]

    assert not assessment.eligible
    assert EvidenceRejectionReason.SUBJECT_MISMATCH in assessment.rejection_reasons
    assert EvidenceRejectionReason.ACTIVITY_MISMATCH in assessment.rejection_reasons
    assert assessment.to_dict()["rejection_reasons"]


def test_activity_matching_accepts_plural_variation_but_not_other_activity():
    question = "When is the equipment inspection scheduled?"
    plan = {"query_constraints": extract_query_constraints(question)}
    matching = {
        "name": "inspection_record.docx",
        "text": "The equipment inspections are scheduled for October 10, 2026.",
    }
    unrelated = {
        "name": "inspection_record.docx",
        "text": "The equipment maintenance is scheduled for October 10, 2026.",
    }

    assert validate_candidates(question, [matching], plan).assessments[
        matching["name"]
    ].eligible
    rejected = validate_candidates(question, [unrelated], plan).assessments[
        unrelated["name"]
    ]
    assert not rejected.eligible
    assert EvidenceRejectionReason.ACTIVITY_MISMATCH in rejected.rejection_reasons


def test_project_risk_audit_uses_only_validated_evidence(monkeypatch):
    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("Project-risk claims should be rendered deterministically")
        ),
    )
    question = "What are the risks identified for the project?"
    plan = {"query_constraints": extract_query_constraints(question)}
    audit = {"analyzed": [], "used": []}
    candidates = [
        {
            "name": "project_risk_assessment.docx",
            "candidate_id": "risk-1",
            "modifiedTime": "2026-09-29",
            "text": "Project Risk Assessment: Identified project risks include delay and cost overrun.",
        },
        {
            "name": "config-project.js",
            "modifiedTime": "2026-09-29",
            "text": "VBA Project Explorer configuration.",
        },
        {
            "name": "Arduino Programming Book.pdf",
            "candidate_id": "arduino-1",
            "modifiedTime": "2026-09-29",
            "text": "Arduino sketches are managed through the Project Explorer. Activity setup uses a microcontroller.",
        },
    ]

    answer = answer_drive_question(question, candidates, plan, audit)

    assert answer == "The identified project risks are delay and cost overrun."
    assert audit["used"] == ["project_risk_assessment.docx"]
    assert audit["reference_evidence"][0]["evidence_span"] == candidates[0]["text"]


def test_authorization_answer_uses_direct_relationship_evidence(monkeypatch):
    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("Explicit authorization relations should be rendered deterministically")
        ),
    )
    question = "Who is authorized to represent John Smith?"
    plan = {"query_constraints": extract_query_constraints(question)}
    assert plan["query_constraints"]["subject_entity"] == "John Smith"
    assert plan["query_constraints"]["requested_field"] == "authorization_role"
    documents = [
        {
            "name": "authorization_test.docx",
            "candidate_id": "auth-1",
            "modifiedTime": "2026-09-29",
            "text": "John Smith authorizes Maria Santos to represent him during the equipment inspection.",
        },
        {
            "name": "activity_record.docx",
            "modifiedTime": "2026-09-29",
            "text": "John Smith's equipment inspection is scheduled for October 10, 2026.",
        },
        {
            "name": "project_risk_assessment.docx",
            "modifiedTime": "2026-09-29",
            "text": "Project risks include schedule delay and equipment failure.",
        },
        {
            "name": "project_status_report.docx",
            "modifiedTime": "2026-09-29",
            "text": "Project Manager: John Smith. Planned equipment inspection date: October 10, 2026.",
        },
        {
            "name": "PACIO_RESUME",
            "modifiedTime": "2026-09-29",
            "text": "John Smith's resume describes his career and project work.",
        },
    ]
    audit = {"used": [], "analyzed": [doc["name"] for doc in documents]}

    answer = answer_drive_question(question, documents, plan, audit)

    assert answer == "Maria Santos is authorized to represent John Smith."
    assert audit["used"] == ["authorization_test.docx"]
    assert audit["reference_evidence"][0]["claim"] == answer
    assert audit["reference_evidence"][0]["evidence_span"] == documents[0]["text"]
    displayed = format_drive_answer("Maria Santos is authorized to represent John Smith.", audit)
    reference_section = displayed.split("**Files analyzed but not used**")[0]
    assert "`authorization_test.docx`" in reference_section
    assert "`project_status_report.docx`" not in reference_section
    assert "`PACIO_RESUME`" not in reference_section
    assert "`project_status_report.docx`" in displayed
    assert "`PACIO_RESUME`" in displayed


def test_unrelated_noise_document_never_reaches_answer_llm(monkeypatch):
    captured = {}
    question = "What is the planned activity date for John Smith?"
    plan = {"query_constraints": extract_query_constraints(question)}
    target_span = (
        "Person: John Smith\n"
        "Activity: Equipment Inspection\n"
        "Activity Date: October 10, 2026"
    )
    claim = "John Smith's planned equipment inspection date is October 10, 2026."
    def fake_generate(prompt):
        captured["prompt"] = prompt
        return claim

    monkeypatch.setattr("modules.llm.generate_local_response", fake_generate)
    candidates = [
        {
            "name": "activity_record.docx",
            "candidate_id": "target-1",
            "modifiedTime": "2026-09-29",
            "text": target_span,
        },
        {
            "name": "unrelated_noise_document.docx",
            "candidate_id": "noise-1",
            "modifiedTime": "2026-09-29",
            "text": (
                "John Smith's biography mentions October 10, 2026.\n"
                "A project handbook discusses activity planning and inspection procedures."
            ),
        },
    ]
    audit = {
        "used": ["unrelated_noise_document.docx"],
        "analyzed": [item["name"] for item in candidates],
        "skipped": [],
        "failed": [],
        "empty": [],
    }

    answer = answer_drive_question(question, candidates, plan, audit)

    assert answer == claim
    assert "activity_record.docx" in captured["prompt"]
    assert target_span in captured["prompt"]
    assert "unrelated_noise_document.docx" not in captured["prompt"]
    assert audit["used"] == []
    assert audit["reference_evidence"] == []
    assert audit["assessments"]["unrelated_noise_document.docx"]["eligible"] is False


def test_salary_without_explicit_evidence_does_not_create_references():
    question = "What is John Smith's salary?"
    plan = {"query_constraints": extract_query_constraints(question)}
    candidates = [
        {
            "name": "PACIO_RESUME",
            "candidate_id": "resume-salary-1",
            "text": "John Smith has a long career in project operations and inspections.",
        },
        {
            "name": "unrelated_noise_document.docx",
            "candidate_id": "noise-salary-1",
            "text": "John Smith attended the project meeting on October 10, 2026.",
        },
    ]
    audit = {
        "used": ["PACIO_RESUME"],
        "analyzed": [item["name"] for item in candidates],
        "skipped": [],
        "failed": [],
        "empty": [],
    }

    answer = answer_drive_question(question, candidates, plan, audit)

    assert "sufficient evidence" in answer.lower()
    assert audit["used"] == []
    assert audit["reference_evidence"] == []


def test_answer_llm_returns_plain_text_from_validated_evidence(monkeypatch):
    captured = {}
    question = "What is the planned activity date for John Smith?"
    evidence_span = (
        "Person: John Smith\n"
        "Activity: Equipment Inspection\n"
        "Activity Date: October 10, 2026"
    )
    claim = "John Smith's planned equipment inspection date is October 10, 2026."
    answer_text = f"{claim} PACIO_RESUME confirms an unrelated project fact."

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return answer_text

    monkeypatch.setattr("modules.llm.generate_local_response", fake_generate)
    candidate = {
        "name": "activity_record.docx",
        "candidate_id": "activity-uncited-1",
        "modifiedTime": "2026-09-29",
        "text": evidence_span,
    }
    audit = {"used": ["PACIO_RESUME"], "analyzed": ["activity_record.docx", "PACIO_RESUME"]}

    answer = answer_drive_question(question, [candidate], search_audit=audit)

    assert answer == answer_text
    assert "VALIDATED EVIDENCE" in captured["prompt"]
    assert evidence_span in captured["prompt"]
    assert '"claims"' not in captured["prompt"]
    assert audit["used"] == []
    assert audit["reference_evidence"] == []


def test_project_risk_paraphrases_have_deterministic_validated_provenance(monkeypatch):
    questions = [
        "What are the risks identified for the project?",
        "What risks were identified during the project?",
    ]
    risk_text = (
        "Project Risk Assessment\n"
        "Identified risks:\n"
        "- Equipment unavailability\n"
        "- Weather delay\n"
        "- Scheduling conflict"
    )
    claim = (
        "The identified project risks are Equipment unavailability, "
        "Weather delay, and Scheduling conflict."
    )
    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("Validated project risks should be rendered deterministically")
        ),
    )
    candidates = [
        {
            "name": "project_risk_assessment.docx",
            "file_id": "risk-report-1",
            "candidate_id": "risk-report-1",
            "modifiedTime": "2026-09-29",
            "text": risk_text,
        },
        {
            "name": "Arduino Programming Book.pdf",
            "file_id": "arduino-1",
            "candidate_id": "arduino-1",
            "modifiedTime": "2026-09-29",
            "text": "Arduino project guide explains Project Explorer and risk assessment tools for electronics projects.",
        },
        {
            "name": "PACIO_RESUME",
            "file_id": "resume-1",
            "candidate_id": "resume-1",
            "modifiedTime": "2026-09-29",
            "text": "John Smith has experience managing projects and inspections.",
        },
    ]
    outcomes = []
    candidate_orders = [candidates, list(reversed(candidates))]
    for question in questions:
        constraints = extract_query_constraints(question)
        for candidate_order in candidate_orders:
            audit = {
                "used": [],
                "analyzed": [item["name"] for item in candidate_order],
            }
            answer = answer_drive_question(
                question,
                candidate_order,
                {"query_constraints": constraints},
                audit,
            )
            outcomes.append((
                answer,
                tuple(audit["validated_evidence_document_ids"]),
                tuple(audit["used"]),
                tuple(
                    (item["document_id"], item["evidence_span"], item["claim"])
                    for item in audit["reference_evidence"]
                ),
                (
                    constraints["subject_entity"],
                    constraints["activity"],
                    constraints["requested_field"],
                    constraints["document_type"],
                ),
            ))

    assert len(set(outcomes)) == 1
    assert outcomes[0][0] == claim
    assert outcomes[0][1] == ("risk-report-1",)
    assert outcomes[0][2] == ("project_risk_assessment.docx",)
    assert all(
        not item.eligible
        for item in validate_candidates(
            questions[0],
            candidates,
            {"query_constraints": extract_query_constraints(questions[0])},
        ).assessments.values()
        if item.file_name == "Arduino Programming Book.pdf"
    )


def test_authorization_evidence_must_bind_requested_operator():
    question = "What equipment is John Smith authorized to operate?"
    constraints = extract_query_constraints(question)
    valid_text = (
        "John Smith is authorized to operate the suction tanker "
        "with plate number GAR 8970."
    )
    wrong_person_text = (
        "John Smith is listed as the requester, while Rhobert Z. Quilala is "
        "authorized to operate the suction tanker with plate number GAR 8970."
    )

    valid = assess_candidate(
        question,
        {"name": "authorization.docx", "text": valid_text},
        {"query_constraints": constraints},
    )
    wrong_person = assess_candidate(
        question,
        {"name": "authorization.docx", "text": wrong_person_text},
        {"query_constraints": constraints},
    )

    assert valid.eligible
    assert valid.evidence_span == valid_text
    assert not wrong_person.eligible
    assert EvidenceRejectionReason.RELATIONSHIP_UNSUPPORTED in wrong_person.rejection_reasons


def test_operator_answer_is_direct_and_wrong_person_letter_fails_closed(monkeypatch):
    question = "What equipment is John Smith authorized to operate?"
    plan = {"query_constraints": extract_query_constraints(question)}
    valid_text = (
        "John Smith is authorized to operate the suction tanker "
        "with plate number GAR 8970."
    )
    valid_document = {
        "name": "John authorization.docx",
        "file_id": "john-auth-1",
        "candidate_id": "john-auth-1",
        "modifiedTime": "2026-09-29",
        "text": valid_text,
    }
    valid_audit = {"used": [], "analyzed": [valid_document["name"]]}
    valid_answer = answer_drive_question(question, [valid_document], plan, valid_audit)

    assert valid_answer == (
        "John Smith is authorized to operate the suction tanker "
        "with plate number GAR 8970."
    )
    assert valid_audit["used"] == ["John authorization.docx"]
    assert valid_audit["reference_evidence"][0]["document_id"] == "john-auth-1"

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("Wrong-entity authorization evidence must not reach the LLM")
        ),
    )
    wrong_document = {
        "name": "RHOBERT Z. QUILALA Authorization Letter.docx",
        "file_id": "other-person-auth-1",
        "candidate_id": "other-person-auth-1",
        "modifiedTime": "2026-09-29",
        "text": (
            "John Smith is listed as the requester, while Rhobert Z. Quilala is "
            "authorized to operate the suction tanker with plate number GAR 8970."
        ),
    }
    wrong_audit = {
        "used": [wrong_document["name"]],
        "analyzed": [wrong_document["name"]],
    }

    wrong_answer = answer_drive_question(question, [wrong_document], plan, wrong_audit)

    assert "sufficient evidence" in wrong_answer.lower()
    assert wrong_audit["used"] == []
    assert wrong_audit["reference_evidence"] == []


def test_coordination_letter_paraphrases_keep_schema_and_claim_validation():
    questions = [
        "What does the Coordination Letter say about John Smith's planned activity date?",
        "What activity is described in the Coordination Letter for John Smith?",
        "What date does the Coordination Letter specify for John Smith?",
    ]
    document = {
        "name": "coordination_letter_test.docx",
        "file_id": "coord-letter-1",
        "candidate_id": "coord-letter-1",
        "text": (
            "COORDINATION LETTER\n"
            "Contact: John Smith.\n"
            "We would like to inform you that Alpha Marine Services will be "
            "conducting an equipment inspection on October 10, 2026."
        ),
    }

    assessments = []
    for question in questions:
        constraints = extract_query_constraints(question)
        assessment = assess_candidate(
            question,
            document,
            {"query_constraints": constraints},
        )
        handler = evaluate_specialized_handlers(
            question,
            [document],
            {"query_constraints": constraints},
        )
        assert constraints["requested_document_type"] == "Coordination Letter"
        assert handler is not None
        assert assessment.document_type_match is True
        assert assessment.eligible
        assessments.append(assessment)

    assert {assessment.evidence_span for assessment in assessments} == {
        document["text"]
    }


def test_coordination_event_requires_a_local_subject_role_not_a_name_mention():
    question = (
        "What is the planned activity date in the Coordination Letter "
        "for John Smith?"
    )
    document = {
        "name": "coordination_letter_test.docx",
        "text": (
            "COORDINATION LETTER\n"
            "John Smith was mentioned in an unrelated memo.\n"
            "We would like to inform your office that Alpha Marine Services "
            "will be conducting an equipment inspection on October 10, 2026."
        ),
    }
    plan = {"query_constraints": extract_query_constraints(question)}

    assessment = assess_candidate(question, document, plan)
    handler = evaluate_specialized_handlers(question, [document], plan)

    assert not assessment.eligible
    assert EvidenceRejectionReason.RELATIONSHIP_UNSUPPORTED in assessment.rejection_reasons
    assert handler is None
