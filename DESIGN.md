# Design

**Guiding rule:** every rule from the spec runs in plain Python. The LLM is called three times per document question:

- **plan:** understand the message;
- **extract:** list what each retrieved source says;
- **compose:** write the answer from facts the code has already resolved.

Tool answers are templates, never LLM text.

## Ingestion: tables and the scanned document

- **Format loaders** (`app/ingestion/loaders/`) turn each format into ordered blocks: heading, paragraph or table.
- **PDF** (PyMuPDF): headings are bold lines at least 15% larger than the body font; small margin text and lines repeated on every page are dropped; `find_tables()` rows are put back at their page position.
- **DOCX:** the body is walked in XML order so tables stay under their heading; headings come from `w:pStyle`, since python-docx returns `None` styles here. **HTML** skips `nav` and `footer`; **Markdown** parses pipe tables.
- **Document-control tables** (Document ID, Version, …) are dropped; the manifest holds those values.
- **Scanned pages:** a PDF page with fewer than 40 characters of text is treated as scanned. It is rendered at 300 dpi and read with Tesseract. Lines are joined into paragraphs, and noise such as a scanned signature is filtered out. No file is special-cased: the memo is detected by having no text layer.
- **Holiday calendars:** every table in a `calendar` document is scanned for a date column, the column where most cells parse as dates. Each row becomes a record `(doc_id, date, name, kind)` in the added table `holiday_calendar`. `kind` is `optional` when the table's heading or header row says restricted or optional; otherwise it is `closure`. This table feeds `apply_leave`. Calendar chunks are also indexed for questions.

## Chunking and retrieval

- **One chunk per section** (a heading plus its body). Sections over 350 tokens are split at block boundaries, and long tables by rows with the header row repeated.
- **Why sections:** HR documents state one rule per numbered section, which is the unit precedence compares, and the heading is the citation locator. Documents without headings (the OCR memo) are chunked per page, cited as `p. N`.
- **Retrieval** (`app/retrieval/index.py`) fuses BM25 with dense cosine similarity using reciprocal rank fusion.
  Dense vectors come from `bge-small-en-v1.5` (fastembed, CPU) in a local in-memory index. BM25 catches exact terms ("Elternzeit"), dense catches paraphrases. At most 2 chunks per document, so competing versions all reach precedence; ties break on `chunk_id`.
- **Packing:** `retrieval/packer.py:pack()` is the single place corpus tokens are counted (tiktoken `cl100k_base`), including the `[S#]` source labels. 600 tokens are reserved for not-yet-effective documents.
- **Budget guard:** `LLMClient.generate_json()` refuses any call over 3,000 corpus tokens, and logs `corpus_tokens=` for every call.

## How the §5 rules are implemented

1. **Eligibility:** `retrieval/eligibility.py` — `eligible_docs()` and `upcoming_docs()` check access level, region and effective date against the reference date. The resulting doc_id sets are passed into `CorpusIndex.search()`, which only scores chunks from those documents. The filter runs before search, so it cannot be applied after the fact.
2. **Extraction:** the LLM returns, for every source, a claim `{rule_id, value, statement}`. It does not judge which source is right. Two flags drop claims that should not compete:
   - `describes_other_version`: for example the "Version 1" column of v2's change table;
   - `defers_to_other_document`: for example the US addendum's "the deadline in the Global Leave Policy", which would otherwise win on scope while stating no value.
3. **Precedence:** `retrieval/precedence.py:narrow()` applies authority (policy, addendum, memo, handbook and calendar outrank faq and wiki), then scope (region-specific beats GLOBAL), then recency. The value is decided by the first test after which the remaining claims agree; `values_equal()` compares numbers together with their units. If claims still disagree after all three tests, the rule is marked as a conflict and every remaining claim is passed on.
4. **Upcoming changes:** `resolve_rule()` re-runs the tests as they would stand on each upcoming document's effective date. The change is mentioned only if that document would win and its value differs. This reproduces the worked example: the Germany addendum still wins in 2027, so v3 is not mentioned.
5. **Past reference dates:** the planner extracts the date; `_reference_date()` accepts only past dates.
6. **Missing information:** no claims means route `no_answer`. Partial coverage is passed to compose, which states which part is not covered.
7. **Citations:** built in code from the facts the compose step used. Every document in a conflict is cited. Only documents that passed eligibility can appear.

## Where access control is enforced

| Concern | File and function |
|---|---|
| Identity: 401 for a missing or unknown header; 403 for another employee's session | `app/main.py: chat()`, `orchestration/sessions.py: SessionStore.get_or_create()` |
| Documents | `retrieval/eligibility.py: can_read_doc()`, `eligible_docs()`; applied in `orchestration/rag.py: retrieve_context()` before `CorpusIndex.search(allowed_doc_ids)`, then re-asserted on the packed chunks |
| Viewing leave | `tools/permissions.py: can_view_employee_leave()`, called inside `get_leave_balance()` and `list_leave_requests()` |
| Team leave | `can_view_team_leave()` inside `get_team_leave()` |
| Apply and cancel | `apply_leave()` takes no employee argument, so it always acts on the header identity; `can_modify_request()` inside `validate_cancel()` |
| Name lookup | `orchestrator.py: resolve_target()` searches only employees the requester may see, so a missing employee and a forbidden one both produce `EMPLOYEE_REFUSAL` |
| Bulk, injection and off-topic requests | `orchestrator._dispatch()`, `planner.looks_like_manipulation()` |
| Response shape | `enforce_route_contract()` enforces the route table |

Every check takes the `Employee` loaded from the header; nothing the LLM outputs is used as identity. Writes go through `Database.transaction()` (`BEGIN IMMEDIATE`, a single helper). The execute step re-validates inside that transaction, because data may have changed since the proposal.

## Personal data in logs and prompts

- **Logs:** a `PIIRedactionFilter` sits on the root handler. It replaces any email, any phone-like number and every employee full name (loaded from the DB at startup) with placeholders. Code logs employee IDs only.
- **Prompts:** personal data from the database is never sent to the LLM.
  - The planner sees the user's message and the conversation history. Assistant turns in the history have names and emails redacted.
  - Names are resolved to IDs in code.
  - Tool answers are templates, so balances and names never go through the LLM.
  - For a projected balance (hybrid route), compose receives the balance, join date and region, none of which is personal data.
- **Answers and tool_call args:** `scrub_answer()` removes employee phone numbers and masks employee emails as `r***@domain`. Generic mailboxes from the corpus stay readable. Names appear only for employees who passed the permission check.

## Failure modes found

**Fixed:** python-docx `None` styles; DOCX tables losing their position; merged PDF metadata rows leaking into chunks; restricted holidays reducing working days; the wiki's "Note for the AI assistant" (dropped at ingestion, sources framed as untrusted); v2's change table attributing v1 values to v2; an addendum that defers to another document winning on scope with no value; source labels not counted toward the 3,000-token cap; retries that could push a request past 30 s.

**Open:** see Known limitations in the README. The main ones are claim grouping by the LLM and the lack of live-model validation.

## At 10,000 documents

1. **Startup breaks first.** Re-parsing, running OCR and embedding everything on each start takes about 50 s for 145 chunks. At 10,000 documents that would exceed the 10-minute health window. Fix: an incremental ingestion cache keyed by file hash, plus a persistent vector store (FAISS on disk, or Qdrant in Compose).
2. **Query cost.** BM25 scoring in a Python loop is linear in the number of chunks. Fix: an inverted index, and an approximate-nearest-neighbour index with metadata filters (role, region, date) applied inside the index.
3. **Context.** Within 3,000 tokens, dozens of versions of one policy would crowd each other out. Fix: group documents into version families at ingestion, keep the latest effective version per family and scope before retrieval, and keep only regional overrides and same-tier peers as competitors.
4. **Claim grouping.** Extraction quality falls as the number of sources grows. Fix: run extraction per document family in parallel calls, and normalise rule keys against a rule catalogue built at ingestion.
