# Design

**Guiding rule:** every rule from the spec runs in plain Python. Per document question the LLM is called three times: **plan** (understand the message; no corpus text), **extract** (list what each retrieved source states) and **compose** (write the answer from facts the code has already resolved). Tool answers are templates, never LLM text.

## Ingestion: tables and the scanned document

- **Loaders** (`app/ingestion/loaders/`) turn each format into ordered heading/paragraph/table blocks.
  - **PDF** (PyMuPDF): headings are bold lines ≥15% larger than the body font, and running headers/footers are dropped. `find_tables()` rows are put back at their page position.
  - **DOCX:** the body is walked in XML order, so tables stay under their heading.
  - **HTML** skips `nav`/`footer`; **Markdown** parses pipe tables.
  - Document-control tables are dropped, because the manifest holds those values.
- **Scanned pages:** any PDF page with under 40 characters of text is rendered at 300 dpi and read by Tesseract. Lines are joined into paragraphs and noise is filtered. No file is special-cased.
- **Holiday calendars:** in `calendar` documents, the column where most cells parse as dates becomes rows `(doc_id, date, name, kind)` in the added table `holiday_calendar`. Tables headed restricted/optional are `optional`, not office closures. `apply_leave` reads this table.
- **Cache:** parsed chunks, holidays and vectors are cached by each file's SHA-256, so only changed files are re-processed. A watcher re-ingests on content change without a restart.

## Chunking and retrieval

- **One chunk per section** (heading + body, ≤350 tokens). Long tables are split by rows with the header repeated. Sections are used because HR documents state one rule per numbered section, which is exactly the unit precedence compares, and the heading is the citation locator. Documents without headings are chunked per page (`p. N`).
- **Hybrid search** (`retrieval/index.py`): BM25 for exact terms ("Elternzeit") fused by reciprocal rank with cosine similarity from a local CPU embedding model (`bge-small-en-v1.5`, for paraphrases). Each document contributes at most 450 tokens, so competing versions all reach precedence. Ties break on `chunk_id`, which keeps results deterministic.
- **Budget:** `retrieval/packer.py:pack()` is the single place corpus tokens are counted (tiktoken `cl100k_base`, source labels included); 600 tokens are reserved for not-yet-effective documents. `LLMClient.generate_json()` refuses any call over 3,000 and logs `corpus_tokens` for every call.

## How the §5 rules are implemented

1. **Eligibility** (`retrieval/eligibility.py: eligible_docs()`, `upcoming_docs()`): access level, region and effective date versus the reference date. The resulting doc_id set is passed into `CorpusIndex.search()`, which scores nothing else. Filtering happens before retrieval, never after.
2. **Extraction:** one claim `{rule_id, value, statement}` per source, without judging. Claims flagged `describes_other_version` (v2's "Version 1" column) or `defers_to_other_document` (an addendum pointing at the global deadline) are dropped.
3. **Grouping safety net** (`merge_similar_rules()`): rules the LLM split are re-united deterministically. The merge is conservative and happens only when all of these hold:
   - the descriptions are subsets of each other, or have near-identical embeddings;
   - the numbers in the descriptions match and no contrasting qualifier differs (primary/secondary);
   - the values are the same kind (days vs a date);
   - no document appears in both groups.
4. **Precedence** (`narrow()`): authority (policy/addendum/memo/handbook/calendar over faq/wiki), then scope (regional over GLOBAL), then recency. The value is decided by the first test after which claims agree; `values_equal()` compares numbers with their units. Disagreement after all three tests is an **unresolved conflict**: every value is given and cited.
5. **Upcoming changes** (`resolve_rule()`): the tests are re-run as of each future effective date. A change is mentioned only if that document would win with a different value; this reproduces the worked example in §5.2.
6. **Past dates:** the planner extracts them; only dates before today become the reference date.
7. **Missing information:** no claims gives `no_answer`; partial coverage is stated by compose. **Citations** are built in code from the facts used, so only eligible documents can appear.

## Where access control is enforced

| Concern | File: function |
|---|---|
| Identity (401), session owner (403) | `main.py: chat()`, `sessions.py: SessionStore.get_or_create()`; optional gateway HMAC in `api/identity.py: verify()` |
| Documents | `eligibility.py: can_read_doc()`, applied in `rag.py: retrieve_context()` before `CorpusIndex.search(allowed_doc_ids)`, re-asserted after packing |
| Leave data | `tools/permissions.py: can_view_employee_leave()` inside `get_leave_balance()`, `list_leave_requests()`; `can_view_team_leave()` inside `get_team_leave()` |
| Writes | `apply_leave()` has no employee argument (always the header identity); `can_modify_request()` inside `validate_cancel()` |
| Name lookup | `orchestrator.py: resolve_target()` searches only visible employees, so a missing and a forbidden employee get the same refusal |
| Bulk, injection, off-topic, route table | `orchestrator._dispatch()`, `planner.looks_like_manipulation()`, `enforce_route_contract()` |

Every check uses the `Employee` loaded from the header, never LLM output. Writes run in one `Database.transaction()` (`BEGIN IMMEDIATE`) that re-validates before writing.

## Personal data in logs and prompts

- **Logs:** a `PIIRedactionFilter` on the root handler replaces emails, phone-like numbers and every employee name (loaded from the DB) with placeholders. Code logs employee IDs only.
- **Prompts:** DB personal data is never sent to the LLM. Names are resolved to IDs in code, tool answers are templates, and assistant turns in the planner's history are redacted. For projected balances, compose receives only the balance, join date and region.
- **Answers / tool args:** `scrub_answer()` removes employee phone numbers and masks employee emails (`r***@domain`). Names are shown only after a permission check.

## Failure modes found

**Fixed:**
- **Parsing:** python-docx `None` styles; DOCX tables losing their position; PDF metadata rows leaking into chunks; restricted holidays reducing working days.
- **Safety:** the wiki's "Note for the AI assistant" (dropped at ingestion, and sources are framed as untrusted).
- **Precedence:**
  - v2's change table attributing v1 values to v2;
  - a deferring addendum winning on scope with no value;
  - a memo and a policy extracted as separate rules (prompt fix plus the merge step);
  - paraphrases reported as false conflicts;
  - an accrual table cut off by a per-document chunk cap.
- **Limits:** labels not counted toward the token cap; Gemini's 10 s minimum deadline; retries that could exceed 30 s.

**Open:** rule grouping still starts from the LLM, and the merge step is deliberately conservative. Answer wording varies slightly between models. Live testing used `gemini-3.5-flash-lite` because of free-tier quotas.

## At 10,000 documents

1. **The first ingestion breaks first:** about 65 s for 30 documents would exceed the 10-minute health window. Fix: parallel parsing/OCR, report healthy once a first slice is indexed, and keep the SHA-256 cache plus a persistent ANN store (FAISS/Qdrant).
2. **Query cost:** Python BM25 is linear in the number of chunks. Fix: an inverted index, and ANN with role/region/date filters inside the index.
3. **Context:** many versions of one policy would crowd the 3,000 tokens. Fix: version families at ingestion, keeping the latest per family and scope, plus regional overrides.
4. **Grouping:** extraction degrades with more sources. Fix: per-family extraction in parallel, and rule keys normalised against a catalogue built at ingestion.
