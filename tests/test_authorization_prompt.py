import json
import re

from modules.llm import answer_drive_question
from modules.search import (
    _build_search_planner_prompt,
    interpret_search_request,
)


def test_active_prompts_have_no_person_specific_steering(monkeypatch):
    planner_prompt = _build_search_planner_prompt(
        "Who is given authorization to bring cellphones within the company? "
        "SEARCH: AUTHORIZATION LETTERS"
    )
    captured = {}

    def capture_answer_prompt(prompt):
        captured["prompt"] = prompt
        return "The authorized person is the employee named in the letter."

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        capture_answer_prompt,
    )
    answer_drive_question(
        "Who is given authorization to bring cellphones within the company? "
        "SEARCH: AUTHORIZATION LETTERS",
        [
            {
                "name": "Authorization Letter.docx",
                "modifiedTime": "2026-09-29T00:00:00Z",
                "score": 40,
                "text": (
                    "This letter formally authorizes Employee A to bring a "
                    "cellphone. RHYS JOHN PACIO is the supervisor and approver."
                ),
            }
        ],
        {"intent": "authorization letters", "answer_type": "person"},
    )

    instructions = captured["prompt"].split("Here are the files:", 1)[0]
    for prompt in (planner_prompt, instructions):
        assert re.search(r"rhys\s+(?:john\s+)?pacio", prompt, re.IGNORECASE) is None


def test_answer_prompt_distinguishes_authorized_person_from_approver(monkeypatch):
    captured = {}
    expected_answer = "IAN JESSON R. CAÑAZARES is the person authorized to bring a cellphone."

    def capture_answer_prompt(prompt):
        captured["prompt"] = prompt
        return expected_answer

    monkeypatch.setattr(
        "modules.llm.generate_local_response",
        capture_answer_prompt,
    )
    document_text = (
        "This letter is to formally authorize Environmental Officer "
        "IAN JESSON R. CAÑAZARES to bring a cellphone during the designated "
        "schedule. RHYS JOHN PACIO, Wastewater Treatment Supervisor, is "
        "listed under Approved by."
    )

    answer = answer_drive_question(
        "Who is given authorization to bring cellphones within the company? "
        "SEARCH: AUTHORIZATION LETTERS",
        [
            {
                "name": "Authorization Letter.docx",
                "modifiedTime": "2026-09-29T00:00:00Z",
                "score": 48,
                "text": document_text,
            }
        ],
        {"intent": "authorization letters", "answer_type": "person"},
    )

    prompt = captured["prompt"]
    instructions, supplied_documents = prompt.split("Here are the files:", 1)
    assert "IAN JESSON R. CAÑAZARES" in supplied_documents
    assert "RHYS JOHN PACIO" in supplied_documents
    assert "expressly named as receiving the authorization" in instructions
    assert "approver, supervisor, signatory, reviewer, or witness" in instructions
    assert "Do not infer that a person has an authorization" in instructions
    assert re.search(r"rhys\s+(?:john\s+)?pacio", instructions, re.IGNORECASE) is None
    assert "traceable evidence" in answer.lower()


def test_authorization_search_planner_prompt_is_generic_and_preserves_anchor(monkeypatch):
    captured = {}
    question = (
        "Who is given authorization to bring cellphones within the company? "
        "SEARCH: AUTHORIZATION LETTERS"
    )

    def capture_planner_prompt(prompt, response_format=None):
        captured["prompt"] = prompt
        return json.dumps(
            {
                "intent": "authorization letters",
                "required_terms": ["authorization"],
                "phrases": [],
                "optional_terms": ["cellphones"],
                "context_terms": ["authorized person"],
                "exclude_terms": [],
                "time_range": {"from": None, "to": None},
                "answer_type": "person",
                "confidence": 0.9,
            }
        )

    monkeypatch.setattr(
        "modules.search.generate_local_response",
        capture_planner_prompt,
    )

    plan = interpret_search_request(question)

    assert "authorization letter" in " ".join(
        plan["required_terms"] + plan["phrases"] + plan["optional_terms"]
    )
    assert re.search(
        r"rhys\s+(?:john\s+)?pacio",
        captured["prompt"],
        re.IGNORECASE,
    ) is None
    assert "Who is authorized to operate the laboratory vehicle?" in captured["prompt"]