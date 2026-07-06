# Data Directory

This directory is the local workspace for IMDb-derived datasets used by the Master 2 final project.

Large CSV and TSV files are intentionally ignored by git.

To run the SUQL engines and benchmarks locally, place or generate these joined CSV files here:

```text
imdb_joined.csv
imdb_reviews.csv
imdb_structured_joined.csv
```

The raw IMDb TSV files are only needed if you want to rerun the preprocessing notebooks and rebuild the joined CSVs from scratch:

```text
name.basics.tsv
title.basics.tsv
title.crew.tsv
```

The notebooks in this directory document the preprocessing steps used to build the joined datasets.
