# Universal semantic dictionary miner

This directory mines data-derived prompt context for semantic LLM judges. It
does not implement a keyword classifier and the output threshold is always
`null`.

Production example:

```bash
python mine_semantic_dict.py /path/to/full_imdb_reviews.csv \
  --embedding-backend sentence-transformers \
  --embedding-model all-MiniLM-L6-v2
```

The input must contain a movie identifier (`movie_id`, `tconst`, or `imdb_id`)
and review text (`review_text`, `review`, or `text`). Column names can also be
provided explicitly. Categories live in `categories.json`; append a new object
to mine a new category without changing the Python code.

`--embedding-backend hashing` is provided only for dependency-free offline
smoke tests. A dictionary mined that way records the fallback in provenance and
should be regenerated with SentenceTransformers before production use.

The miner writes `semantic_dict.json`, `semantic_dict_splits.json`, and
`semantic_dict_report.md`. The holdout IDs are persisted but their review text
is never read by any mining stage after the split.
