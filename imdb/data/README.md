# Central data directory

All runtime datasets are centralized here. The original project corpus has
been reconstructed using the preprocessing notebooks recovered from Git commit
`63438c90362fe5fa9b93f0292beb716987d8aec5`.

- `canonical/imdb_reviews.csv`: the 25,000-row IMDb-ID/Stanford training split
  (`tconst`, `review`, `score`, `label`), preserved without deduplication.
- `canonical/imdb_structured.csv`: structured IMDb movie database built from
  `title.basics`, `title.crew`, and `name.basics`.
- `canonical/imdb_structured_joined.csv`: compatibility alias of the structured database.
- `canonical/imdb_joined.csv`: SUQL table created by joining reviews to structured movies on `tconst = movie_id`.
- `canonical/provenance.json`: source URLs, checksums, snapshot caveat, and row counts.
- `sources/`: downloaded source snapshots used for reproducible reconstruction.
- `benchmark_union/`: deduplicated union of all benchmark records (1,000 rows).
- `subdatasets/<suite>/<question>/`: exact per-question benchmark datasets, annotations, and ground truth.

Each subdataset contains an `annotations.csv` data dictionary in row form. Its
`structured_match`, `semantic_label`, and `ground_truth` columns expose the
label derivation; `annotation_source`, `evidence_excerpt`, and
`annotation_rationale` make every label traceable to its policy and source
review. These annotation columns are metadata only: implementations receive
the movie/review inputs and never read labels during execution.

Run `python3 data/restore_original_data.py` to reconstruct `canonical/` from
the downloaded source snapshots. Run `python3 data/build_data.py` to refresh
`benchmark_union/` and centralize the benchmark subdatasets; it intentionally
does not overwrite the restored canonical corpus. Benchmark execution reads
the subdatasets from this directory rather than historical per-question copies.
