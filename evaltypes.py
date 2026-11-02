"""Shared shapes. Every module speaks these, so they can be built and tested apart."""

# doc_id -> {"title": str, "text": str}
Corpus = dict[str, dict[str, str]]
# query_id -> query text
Queries = dict[str, str]
# query_id -> {doc_id: relevance grade}; FiQA grades are all 1
Qrels = dict[str, dict[str, int]]
# query_id -> [(doc_id, score), ...] sorted by score, best first
Run = dict[str, list[tuple[str, float]]]
