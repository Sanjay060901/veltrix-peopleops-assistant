"""TASK §5 precedence: authority, scope, recency, unresolved conflict, upcoming change, past date.

Documents mirror the invented worked example in TASK §5.2.
"""
from datetime import date

from app.ingestion.models import Chunk, DocMeta
from app.retrieval.eligibility import can_read_doc, eligible_docs, upcoming_docs
from app.retrieval.precedence import Claim, resolve, values_equal


def doc(doc_id, doc_type, regions, eff, access="all"):
    return DocMeta(doc_id, f"{doc_id}.md", doc_id, doc_type, tuple(regions), date.fromisoformat(eff), access)


V1 = doc("V1", "policy", ["GLOBAL"], "2025-01-01")
V2 = doc("V2", "policy", ["GLOBAL"], "2026-04-01")
FAQ = doc("FAQ", "faq", ["GLOBAL"], "2026-05-01")
DE = doc("DE", "addendum", ["DE"], "2025-06-01")
V3 = doc("V3", "policy", ["GLOBAL"], "2027-01-01")
DOCS = [V1, V2, FAQ, DE, V3]
VALUES = {"V1": "2 days", "V2": "3 days", "FAQ": "2 days", "DE": "5 days", "V3": "4 days"}
TODAY = date(2026, 9, 15)


def claims_for(docs, values=VALUES, rule="R1"):
    out = []
    for i, d in enumerate(docs):
        chunk = Chunk(f"{d.doc_id}#1", d.doc_id, "1. Section", "1. Section", "text")
        out.append(Claim(f"F{i}", rule, "volunteer days", values[d.doc_id], f"{values[d.doc_id]} stated", chunk, d))
    return out


def answer(region, reference=TODAY):
    visible = eligible_docs(DOCS, "employee", region, reference) + (
        upcoming_docs(DOCS, "employee", region, reference) if reference == TODAY else []
    )
    (res,) = resolve(claims_for(visible), reference, include_upcoming=reference == TODAY)
    return res


def test_india_employee_recency_wins_and_upcoming_is_mentioned():
    res = answer("IN")
    assert [c.doc.doc_id for c in res.winners] == ["V2"]
    assert not res.conflict
    assert res.decided_by == "recency"  # authority removes the FAQ, v1/v2 tie on scope
    assert [c.doc.doc_id for c in res.upcoming] == ["V3"]


def test_faq_loses_on_authority():
    claims = claims_for([V2, FAQ], {"V2": "3 days", "FAQ": "2 days"})
    (res,) = resolve(claims, TODAY)
    assert [c.doc.doc_id for c in res.winners] == ["V2"] and res.decided_by == "authority"


def test_germany_employee_scope_wins_and_upcoming_not_mentioned():
    res = answer("DE")
    assert [c.doc.doc_id for c in res.winners] == ["DE"]
    assert res.upcoming == []  # the addendum would still win on scope in 2027


def test_past_reference_date_uses_only_documents_in_force():
    res = answer("IN", reference=date(2025, 8, 1))
    assert [c.doc.doc_id for c in res.winners] == ["V1"]
    assert res.upcoming == []


def test_scope_beats_recency():
    newer_global = doc("G", "policy", ["GLOBAL"], "2026-06-01")
    claims = claims_for([DE, newer_global], {"DE": "5 days", "G": "6 days"})
    (res,) = resolve(claims, TODAY)
    assert [c.doc.doc_id for c in res.winners] == ["DE"] and res.decided_by == "scope"


def test_unresolved_conflict_keeps_both():
    a = doc("A", "policy", ["GLOBAL"], "2026-01-01")
    b = doc("B", "handbook", ["GLOBAL"], "2026-01-01")
    (res,) = resolve(claims_for([a, b], {"A": "30 days", "B": "45 days"}), TODAY)
    assert res.conflict and {c.doc.doc_id for c in res.winners} == {"A", "B"}


def test_agreeing_documents_are_not_a_conflict():
    a = doc("A", "policy", ["GLOBAL"], "2026-04-01")
    b = doc("B", "policy", ["GLOBAL"], "2026-04-01")
    (res,) = resolve(claims_for([a, b], {"A": "12 working days", "B": "12 days"}), TODAY)
    assert not res.conflict and len(res.winners) == 2


def test_upcoming_change_for_new_rule_is_mentioned():
    (res,) = resolve(claims_for([V3]), TODAY)
    assert res.winners == [] and [c.doc.doc_id for c in res.upcoming] == ["V3"]


def test_upcoming_same_value_is_not_mentioned():
    claims = claims_for([V2, V3], {"V2": "3 days", "V3": "3 days"})
    (res,) = resolve(claims, TODAY)
    assert res.upcoming == []


def test_value_comparison():
    assert values_equal("12 working days", "12 days")
    assert not values_equal("10 days", "10 weeks")
    assert not values_equal("31 March", "30 June")
    assert values_equal("up to 2 days per week", "2 days a week")
    assert not values_equal("up to 2 days per week", "up to 3 days per week")
    # paraphrased procedural statements agree; a one-sided negation does not
    assert values_equal("according to the accrual schedule", "earned monthly per the accrual schedule")
    assert not values_equal("deducted from balance", "not deducted from balance")


def test_access_levels_per_role():
    assert can_read_doc("employee", "all") and not can_read_doc("employee", "managers") and not can_read_doc("employee", "hr")
    assert can_read_doc("manager", "managers") and not can_read_doc("manager", "hr")
    assert all(can_read_doc("hr", level) for level in ("all", "managers", "hr"))


def test_eligibility_filters_region_date_and_access():
    restricted = doc("M", "handbook", ["GLOBAL"], "2026-01-01", "managers")
    docs = DOCS + [restricted]
    ids = {d.doc_id for d in eligible_docs(docs, "employee", "IN", TODAY)}
    assert ids == {"V1", "V2", "FAQ"}
    assert "M" in {d.doc_id for d in eligible_docs(docs, "manager", "IN", TODAY)}
    assert "DE" in {d.doc_id for d in eligible_docs(docs, "employee", "DE", TODAY)}
    assert {d.doc_id for d in upcoming_docs(docs, "employee", "IN", TODAY)} == {"V3"}
