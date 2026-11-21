# Research and implementation references

The dataset consists only of repository-local synthetic documents; those are the *domain facts* for grading. SDK implementation was checked against current public documentation as of 2026-11-21:

- [LangSmith evaluation guide](https://docs.langchain.com/langsmith/evaluate-llm-application) — `Client.evaluate(target, data=..., evaluators=..., max_concurrency=...)` and target/evaluator signatures.
- [LangSmith Python `create_examples` reference](https://reference.langchain.com/python/langsmith/client/Client/create_examples) — current `examples=[{inputs, reference_outputs, metadata}]` shape.
- [LangSmith regional endpoint support](https://support.langchain.com/articles/6271700631-pointing-the-langsmith-sdk-and-cli-at-a-non-us-regional-endpoint) — EU API endpoint and region-specific API key.
- [LangSmith Python SDK README](https://github.com/langchain-ai/langsmith-python) — `@traceable`, `wrap_openai`.

The referenced bootcamp notebook `content/02_langsmith.ipynb` was not supplied with this lab. This implementation follows equivalent public SDK patterns rather than claiming to reproduce that notebook.
