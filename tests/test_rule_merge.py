"""Deterministic safety net for rule grouping: split rules are re-united before precedence."""
from __future__ import annotations

from datetime import date

from app.ingestion.models import Chunk, DocMeta
from app.retrieval.precedence import (
    Claim, description_overlap, descriptions_compatible, merge_similar_rules, resolve, value_kinds,
)

TODAY = date(2026, 9, 15)


def doc(doc_id, doc_type="policy", regions=("GLOBAL",), eff="2025-01-01"):
    return DocMeta(doc_id, f"{doc_id}.md", doc_id, doc_type, tuple(regions), date.fromisoformat(eff), "all")


def claim(cid, rule_id, rule, value, d):
    return Claim(cid, rule_id, rule, value, value, Chunk(f"{d.doc_id}#1", d.doc_id, "s", "s", "t"), d)


POLICY = doc("POLICY")
MEMO = doc("MEMO", "memo", ("IN", "US"), "2026-07-01")
FAQ = doc("FAQ", "faq", eff="2026-05-01")


def test_split_rule_is_merged_and_scope_then_decides():
    claims = [
        claim("F1", "R1", "work from home days per week", "up to 2 days per week", POLICY),
        claim("F2", "R2", "work from home days per week for the Bengaluru and Austin offices", "up to 3 days per week", MEMO),
        claim("F3", "R3", "Work-from-home days a week", "2 days a week", FAQ),
    ]
    # Without the merge each claim would "apply" on its own and the answer would mix values.
    assert len(resolve(claims, TODAY)) == 3
    merged = merge_similar_rules(claims)
    assert {c.rule_id for c in merged} == {"R1"}
    (res,) = resolve(merged, TODAY)
    assert [c.doc.doc_id for c in res.winners] == ["MEMO"] and not res.conflict


def test_embedding_similarity_merges_paraphrases():
    claims = [
        claim("F1", "R1", "sick leave entitlement", "12 working days", POLICY),
        claim("F2", "R2", "paid sick days each year", "10 days", FAQ),
    ]
    assert len({c.rule_id for c in merge_similar_rules(claims)}) == 2  # word overlap alone is too low
    similar = lambda a, b: 0.95  # noqa: E731 - stands in for the embedding model
    assert len({c.rule_id for c in merge_similar_rules(claims, similar)}) == 1


def test_different_kinds_of_value_are_never_merged():
    claims = [
        claim("F1", "R1", "carry forward of annual leave limit", "10 days", POLICY),
        claim("F2", "R2", "carry forward of annual leave deadline", "31 March", MEMO),
    ]
    assert len({c.rule_id for c in merge_similar_rules(claims, lambda a, b: 0.99)}) == 2


def test_two_values_from_one_document_are_different_rules():
    claims = [
        claim("F1", "R1", "parental leave primary caregiver weeks", "20 weeks", POLICY),
        claim("F2", "R2", "parental leave secondary caregiver weeks", "4 weeks", POLICY),
    ]
    assert len({c.rule_id for c in merge_similar_rules(claims, lambda a, b: 0.99)}) == 2


def test_unrelated_rules_stay_apart():
    claims = [
        claim("F1", "R1", "sick leave days per year", "12 days", POLICY),
        claim("F2", "R2", "bereavement leave days", "5 days", FAQ),
    ]
    assert len({c.rule_id for c in merge_similar_rules(claims, lambda a, b: 0.6)}) == 2


def test_merge_is_deterministic_and_keeps_claims():
    claims = [
        claim("F3", "R9", "home working days per week", "2 days per week", FAQ),
        claim("F1", "R2", "home working days per week", "3 days per week", MEMO),
        claim("F2", "R5", "home working days per week", "2 days per week", POLICY),
    ]
    runs = [[(c.claim_id, c.rule_id) for c in merge_similar_rules(claims)] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert {r for _, r in runs[0]} == {"R2"} and [c for c, _ in runs[0]] == ["F3", "F1", "F2"]


def test_value_kinds_and_overlap():
    assert value_kinds("31 March") == {"date"}
    assert value_kinds("up to 3 days per week") == {"day"}
    assert value_kinds("not deducted") == {"text"}
    assert value_kinds("45") == {"number"}
    assert description_overlap("work from home days", "work from home days for Austin office") == 1.0
    assert description_overlap("sick", "sick days") == 0.0  # too short to judge


def test_bands_with_different_numbers_stay_apart():
    v3 = doc("V3", eff="2027-01-01")
    claims = [
        claim("F1", "R1", "annual leave accrual, less than 2 years of service", "1.5 days per month", POLICY),
        claim("F2", "R2", "annual leave accrual, 2 to less than 5 years of service", "2.0 days per month", v3),
    ]
    assert len({c.rule_id for c in merge_similar_rules(claims, lambda a, b: 0.99)}) == 2


def test_contrasting_qualifiers_stay_apart_even_when_embeddings_agree():
    claims = [
        claim("F1", "R1", "parental leave for the primary caregiver", "20 weeks", POLICY),
        claim("F2", "R2", "parental leave for the secondary caregiver", "2 weeks", FAQ),
    ]
    assert len({c.rule_id for c in merge_similar_rules(claims, lambda a, b: 0.99)}) == 2


def test_description_guards():
    assert descriptions_compatible("work from home days per week", "work from home days per week, Austin office")
    assert not descriptions_compatible("accrual under 2 years", "accrual 2 to 5 years")
    assert not descriptions_compatible("paid leave days", "unpaid leave days")
