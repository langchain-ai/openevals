# Domain and dataset design

**Client scenario.** CompanyScope is a fictionalized AI company-research assistant that receives snippets from commercial registers, accounts and online news, then creates short, evidence-backed answers for analysts. A good output answers the precise entity-specific question, distinguishes missing/old/unverified evidence from facts, shows its source IDs, and does not obey adversarial instructions embedded in retrieved documents.

**Sources.** Five original synthetic documents under `documents/` are the complete evidence corpus for this lab. They deliberately resemble registry, finance, news and security material while describing **invented entities and events**. No claims about real companies or live data are implied.

**Dataset name:** `companyscope-evidence-v1`. **Size:** 18 cases spanning entity identity, name collisions, financial math, missing information, source freshness, unsupported certifications and prompt injections. Sources are versioned with the repository.

**Schema** (`cases.json`): `id`, `category`, `difficulty`, `source_ids`, `inputs` `{case_id,question,evidence}`, and `reference_outputs` `{answer,status,evidence_ids}`. `status` is `supported` if the cited evidence establishes a bounded answer, or `insufficient_evidence` if the requested conclusion is not supportable. Each input contains the relevant excerpts with `[SOURCE_ID]` labels; the target cannot retrieve additional information. A human-curated reference answer defines the expected semantics, not mandatory exact wording.

**Design notes.** The same sources can support multiple independent questions; this tests generalization only within a tiny, public-on-GitHub teaching set. Correctness is judged on factual alignment, numerical reasoning, uncertainty and instruction hierarchy. Additional deterministic checks validate required JSON fields, status class and citation validity. This is not a statistically representative production sample or a certification of compliance.
