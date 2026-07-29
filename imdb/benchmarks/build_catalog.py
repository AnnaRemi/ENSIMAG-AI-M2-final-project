#!/usr/bin/env python3
"""Build a diverse, deterministic ten-question robustness benchmark."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "10q"
LAB_ROOT = ROOT.parents[1]
SOURCE = LAB_ROOT / "data" / "canonical" / "imdb_joined.csv"
QUESTION_ROOT = ROOT / "per_question"
QUESTION_DATA_ROOT = LAB_ROOT / "data" / "subdatasets" / "10q"
ROWS_PER_QUESTION = 100
OUTPUT_COLUMNS = ["movie_id", "title", "director", "year", "runtime", "genres", "review"]


@dataclass(frozen=True)
class QuestionSpec:
    number: int
    slug: str
    difficulty: int
    difficulty_label: str
    question: str
    semantic_question: str
    semantic_task: str
    semantic_patterns: tuple[str, ...]
    suql_query: str
    structured_filters: tuple[dict[str, str], ...]
    genres_any: tuple[str, ...] = ()
    genres_all: tuple[str, ...] = ()
    year_eq: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None
    title_contains: str | None = None
    director_contains: str | None = None


QUESTIONS = [
    QuestionSpec(1, "question_01_2001_recommended", 1, "easy",
        "Which movies released in 2001 have reviews that explicitly recommend the movie?",
        "Does the review explicitly recommend the movie or say it is worth watching?",
        "explicit recommendation", (r"\brecommend(?:ed|ing|s)?\b", r"\bworth (?:a )?(?:watch|watching|seeing)\b", r"\bmust[- ]see\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE year = 2001 AND answer(review, 'Does the review explicitly recommend the movie or say it is worth watching?') = 'Yes';",
        ({"column":"year","op":"eq","value":"2001"},), year_eq=2001),
    QuestionSpec(2, "question_02_short_comedy_funny", 2, "medium",
        "Which Comedy movies under 95 minutes have reviews describing them as funny?",
        "Does the review say the movie is funny, hilarious, or made the reviewer laugh?",
        "humor recognition", (r"\bfunny\b", r"\bhilarious\b", r"\blaugh(?:ed|ing|s)?\b", r"\bhumou?r\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Comedy%' AND runtime < 95 AND answer(review, 'Does the review say the movie is funny, hilarious, or made the reviewer laugh?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Comedy"},{"column":"runtime","op":"lt","value":"95"}), genres_all=("Comedy",), runtime_max=94),
    QuestionSpec(3, "question_03_short_horror_scary", 2, "medium",
        "Which Horror movies under 100 minutes have reviews describing them as scary or frightening?",
        "Does the review describe the movie as scary, frightening, terrifying, or creepy?",
        "fear and atmosphere", (r"\bscary\b", r"\bfrighten(?:ing|ed)?\b", r"\bterrifying\b", r"\bcreepy\b", r"\bchilling\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Horror%' AND runtime < 100 AND answer(review, 'Does the review describe the movie as scary, frightening, terrifying, or creepy?') = 'Yes';",
        ({"column":"runtime","op":"lt","value":"100"},{"column":"genres","op":"contains","value":"Horror"}), genres_all=("Horror",), runtime_max=99),
    QuestionSpec(4, "question_04_drama_100_140_acting", 3, "medium_hard",
        "Which Drama movies with runtime >= 100 and runtime <= 140 have reviews praising the acting or performances?",
        "Does the review praise the acting, cast, or a performance?",
        "acting praise", (r"\bgreat acting\b", r"\bexcellent (?:acting|performance)\b", r"\bstrong performance", r"\bsuperb (?:acting|performance|cast)\b", r"\bbrilliant (?:acting|performance)\b", r"\bwell[- ]acted\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Drama%' AND runtime >= 100 AND runtime <= 140 AND answer(review, 'Does the review praise the acting, cast, or a performance?') = 'Yes';",
        ({"column":"runtime","op":"ge","value":"100"},{"column":"runtime","op":"le","value":"140"},{"column":"genres","op":"contains","value":"Drama"}), genres_all=("Drama",), runtime_min=100, runtime_max=140),
    QuestionSpec(5, "question_05_fantasy_visuals", 3, "medium_hard",
        "Which Fantasy movies have reviews praising the visuals or special effects?",
        "Does the review praise the visuals, cinematography, imagery, or special effects?",
        "visual-quality praise", (r"\bstunning visual", r"\bbeautiful(?:ly)? (?:shot|filmed|photograph|visual|cinematograph)", r"\bgreat special effects\b", r"\bexcellent special effects\b", r"\bvisual(?:ly)? (?:impressive|stunning|spectacular)\b", r"\bimpressive (?:visuals|effects|cinematography)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Fantasy%' AND answer(review, 'Does the review praise the visuals, cinematography, imagery, or special effects?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Fantasy"},), genres_all=("Fantasy",)),
    QuestionSpec(6, "question_06_romantic_comedy_chemistry", 4, "hard",
        "Which movies that are both Comedy and Romance have reviews praising romantic chemistry or the central relationship?",
        "Does the review praise the romantic chemistry or central relationship between characters?",
        "relationship and chemistry praise", (r"\bgreat chemistry\b", r"\bwonderful chemistry\b", r"\bromantic chemistry\b", r"\bbelievable relationship\b", r"\bchemistry between\b", r"\blove story (?:works|is touching|is convincing)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Comedy%' AND genres LIKE '%Romance%' AND answer(review, 'Does the review praise the romantic chemistry or central relationship between characters?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Comedy"},{"column":"genres","op":"contains","value":"Romance"}), genres_all=("Comedy","Romance")),
    QuestionSpec(7, "question_07_director_john_slow", 4, "hard",
        "Which movies directed by John with reviews criticizing the pacing as slow or boring?",
        "Does the review criticize the movie as slow, boring, dragging, or badly paced?",
        "negative pacing assessment", (r"\btoo slow\b", r"\bslow[- ]moving\b", r"\bboring\b", r"\bdrag(?:s|ged|ging)\b", r"\bpoor(?:ly)? paced\b", r"\btedious\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE director LIKE '%John%' AND answer(review, 'Does the review criticize the movie as slow, boring, dragging, or badly paced?') = 'Yes';",
        ({"column":"director","op":"contains","value":"john"},), director_contains="John"),
    QuestionSpec(8, "question_08_director_michael_exciting", 4, "hard",
        "Which movies directed by Michael with reviews calling them exciting, thrilling, or suspenseful?",
        "Does the review describe the movie as exciting, thrilling, gripping, or suspenseful?",
        "excitement and suspense", (r"\bexciting\b", r"\bthrilling\b", r"\bgripping\b", r"\bsuspenseful\b", r"\bedge of (?:my|your|the) seat\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE director LIKE '%Michael%' AND answer(review, 'Does the review describe the movie as exciting, thrilling, gripping, or suspenseful?') = 'Yes';",
        ({"column":"director","op":"contains","value":"michael"},), director_contains="Michael"),
    QuestionSpec(9, "question_09_the_title_original", 5, "very_hard",
        "Which movies with title contains The, with reviews praising the story as original, inventive, or unpredictable?",
        "Does the review praise the story or premise as original, inventive, fresh, or unpredictable?",
        "originality assessment", (r"\boriginal (?:story|plot|idea|premise|concept)\b", r"\binventive\b", r"\brefreshingly original\b", r"\bfresh (?:idea|take|approach|story)\b", r"\bunpredictable\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE title LIKE '%The%' AND answer(review, 'Does the review praise the story or premise as original, inventive, fresh, or unpredictable?') = 'Yes';",
        ({"column":"title","op":"contains","value":"The"},), title_contains="The"),
    QuestionSpec(10, "question_10_crime_ending", 5, "very_hard",
        "Which Crime movies have reviews praising the ending or plot twist?",
        "Does the review praise the ending, finale, resolution, or plot twist?",
        "ending and twist praise", (r"\bgreat ending\b", r"\bexcellent ending\b", r"\bbrilliant ending\b", r"\bsatisfying ending\b", r"\bpowerful ending\b", r"\bperfect ending\b", r"\bgreat twist\b", r"\bclever twist\b", r"\bsurprise ending\b", r"\bending (?:was|is) (?:great|excellent|brilliant|satisfying|powerful|perfect|effective)\b", r"\btwist (?:was|is) (?:great|excellent|brilliant|clever|effective)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Crime%' AND answer(review, 'Does the review praise the ending, finale, resolution, or plot twist?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Crime"},), genres_all=("Crime",)),
]


# This catalog is intentionally independent of the historical 1q/3q/5q
# questions above. Keeping the old declarations in the file documents the
# held-out prompts; only this new list is used to build 10q.
QUESTIONS = [
    QuestionSpec(1, "new_01_animation_humor", 2, "medium",
        "Which animated movies have reviews saying that the film is funny or hilarious?",
        "Does the review describe the film as funny, hilarious, humorous, or laugh-inducing?",
        "animated-film humor", (r"\bfunny\b", r"\bhilarious\b", r"\blaugh(?:ed|ing|s)?\b", r"\bhumou?r\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Animation%' AND answer(review, 'Does the review describe the film as funny, hilarious, humorous, or laugh-inducing?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Animation"},), genres_all=("Animation",)),
    QuestionSpec(2, "new_02_modern_drama_acting", 3, "medium_hard",
        "Which Drama movies released since 1990 have reviews commending the acting or cast performances?",
        "Does the review commend the acting, cast, or an individual performance?",
        "modern-drama performance praise", (r"\bgreat acting\b", r"\bexcellent (?:acting|performance)\b", r"\bstrong performance", r"\bsuperb (?:acting|performance|cast)\b", r"\bbrilliant (?:acting|performance)\b", r"\bwell[- ]acted\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Drama%' AND year >= 1990 AND answer(review, 'Does the review commend the acting, cast, or an individual performance?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Drama"},{"column":"year","op":"ge","value":"1990"}), genres_all=("Drama",), year_min=1990),
    QuestionSpec(3, "new_03_romance_chemistry", 4, "hard",
        "Which Romance movies between 90 and 130 minutes have reviews praising the characters' chemistry or relationship?",
        "Does the review praise the chemistry or relationship between the central characters?",
        "feature-length romantic chemistry", (r"\b(?:great|wonderful|romantic) chemistry\b", r"\bbelievable relationship\b", r"\bchemistry between\b", r"\blove story (?:works|is touching|is convincing)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Romance%' AND runtime >= 90 AND runtime <= 130 AND answer(review, 'Does the review praise the chemistry or relationship between the central characters?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Romance"},{"column":"runtime","op":"ge","value":"90"},{"column":"runtime","op":"le","value":"130"}), genres_all=("Romance",), runtime_min=90, runtime_max=130),
    QuestionSpec(4, "new_04_long_action_excitement", 3, "medium_hard",
        "Which Action movies lasting at least 100 minutes have reviews calling them exciting, gripping, or suspenseful?",
        "Does the review call the movie exciting, thrilling, gripping, or suspenseful?",
        "long-action excitement", (r"\bexciting\b", r"\bthrilling\b", r"\bgripping\b", r"\bsuspenseful\b", r"\bedge of (?:my|your|the) seat\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Action%' AND runtime >= 100 AND answer(review, 'Does the review call the movie exciting, thrilling, gripping, or suspenseful?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Action"},{"column":"runtime","op":"ge","value":"100"}), genres_all=("Action",), runtime_min=100),
    QuestionSpec(5, "new_05_mystery_ending", 5, "very_hard",
        "Which Mystery movies have reviews approving of the ending or a plot twist?",
        "Does the review approve of the ending, finale, resolution, or plot twist?",
        "mystery ending approval", (r"\b(?:great|excellent|brilliant|satisfying|powerful|perfect) ending\b", r"\b(?:great|clever) twist\b", r"\bsurprise ending\b", r"\bending (?:was|is) (?:great|excellent|brilliant|satisfying|powerful|perfect|effective)\b", r"\btwist (?:was|is) (?:great|excellent|brilliant|clever|effective)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Mystery%' AND answer(review, 'Does the review approve of the ending, finale, resolution, or plot twist?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Mystery"},), genres_all=("Mystery",)),
    QuestionSpec(6, "new_06_music_soundtrack", 3, "medium_hard",
        "Which Music or Musical movies have reviews praising the soundtrack, score, music, or songs?",
        "Does the review praise the soundtrack, musical score, music, or songs?",
        "soundtrack and score praise", (r"\b(?:great|excellent|beautiful|amazing|wonderful|memorable) (?:soundtrack|score|music|songs?)\b", r"\bsoundtrack (?:is|was) (?:great|excellent|amazing|wonderful)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE (genres LIKE '%Music%' OR genres LIKE '%Musical%') AND answer(review, 'Does the review praise the soundtrack, musical score, music, or songs?') = 'Yes';",
        ({"column":"genres","op":"contains_any","value":"Music|Musical"},), genres_any=("Music","Musical")),
    QuestionSpec(7, "new_07_classic_drama_emotion", 3, "medium_hard",
        "Which Drama movies released before 2000 have reviews describing them as emotional, moving, or heartbreaking?",
        "Does the review describe the movie as emotional, moving, touching, or heartbreaking?",
        "pre-2000 emotional impact", (r"\bemotional\b", r"\bmoving\b", r"\btouching\b", r"\bheartbreaking\b", r"\btear[- ]jerker\b", r"\bmade me cry\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Drama%' AND year < 2000 AND answer(review, 'Does the review describe the movie as emotional, moving, touching, or heartbreaking?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Drama"},{"column":"year","op":"lt","value":"2000"}), genres_all=("Drama",), year_max=1999),
    QuestionSpec(8, "new_08_scifi_thought_provoking", 4, "hard",
        "Which Science-Fiction movies have reviews describing them as intelligent, philosophical, or thought-provoking?",
        "Does the review describe the movie as intelligent, philosophical, or thought-provoking?",
        "science-fiction intellectual depth", (r"\bthought[- ]provoking\b", r"\bintelligent\b", r"\bphilosophical\b", r"\bmakes? you think\b", r"\bfood for thought\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Sci-Fi%' AND answer(review, 'Does the review describe the movie as intelligent, philosophical, or thought-provoking?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Sci-Fi"},), genres_all=("Sci-Fi",)),
    QuestionSpec(9, "new_09_documentary_informative", 3, "medium_hard",
        "Which Documentary movies have reviews calling them informative, educational, insightful, or enlightening?",
        "Does the review call the documentary informative, educational, insightful, or enlightening?",
        "documentary informational value", (r"\binformative\b", r"\beducational\b", r"\binsightful\b", r"\benlightening\b", r"\blearn(?:ed|t)\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Documentary%' AND answer(review, 'Does the review call the documentary informative, educational, insightful, or enlightening?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Documentary"},), genres_all=("Documentary",)),
    QuestionSpec(10, "new_10_family_entertaining", 2, "medium",
        "Which Family movies under 121 minutes have reviews describing them as entertaining, enjoyable, or great fun?",
        "Does the review describe the movie as entertaining, enjoyable, or great fun?",
        "family-film entertainment value", (r"\bentertaining\b", r"\benjoyable\b", r"\bfun (?:movie|film|adventure|ride)\b", r"\bgreat fun\b"),
        "SELECT movie_id, title, year, runtime, director, genres FROM movies WHERE genres LIKE '%Family%' AND runtime <= 120 AND answer(review, 'Does the review describe the movie as entertaining, enjoyable, or great fun?') = 'Yes';",
        ({"column":"genres","op":"contains","value":"Family"},{"column":"runtime","op":"le","value":"120"}), genres_all=("Family",), runtime_max=120),
]


def load_source() -> pd.DataFrame:
    source = pd.read_csv(SOURCE).reset_index(names="source_row")
    source = source.dropna(subset=OUTPUT_COLUMNS).copy()
    source["year"] = pd.to_numeric(source["year"], errors="coerce")
    source["runtime"] = pd.to_numeric(source["runtime"], errors="coerce")
    source = source.dropna(subset=["year", "runtime"]).copy()
    source[["year", "runtime"]] = source[["year", "runtime"]].astype(int)
    return source


def structured_mask(frame: pd.DataFrame, spec: QuestionSpec) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    genres = frame["genres"].fillna("")
    if spec.genres_any:
        mask &= pd.concat([genres.str.contains(rf"\b{re.escape(v)}\b", case=False) for v in spec.genres_any], axis=1).any(axis=1)
    for value in spec.genres_all:
        mask &= genres.str.contains(rf"\b{re.escape(value)}\b", case=False)
    if spec.year_eq is not None: mask &= frame["year"].eq(spec.year_eq)
    if spec.year_min is not None: mask &= frame["year"].ge(spec.year_min)
    if spec.year_max is not None: mask &= frame["year"].le(spec.year_max)
    if spec.runtime_min is not None: mask &= frame["runtime"].ge(spec.runtime_min)
    if spec.runtime_max is not None: mask &= frame["runtime"].le(spec.runtime_max)
    if spec.title_contains: mask &= frame["title"].str.contains(re.escape(spec.title_contains), case=False, na=False)
    if spec.director_contains: mask &= frame["director"].str.contains(re.escape(spec.director_contains), case=False, na=False)
    return mask


def semantic_mask(frame: pd.DataFrame, spec: QuestionSpec) -> pd.Series:
    combined = "(?:" + "|".join(spec.semantic_patterns) + ")"
    return frame["review"].str.contains(combined, case=False, regex=True, na=False)


def take_unique(frame: pd.DataFrame, count: int, used: set[str]) -> pd.DataFrame:
    rows = []
    for row in frame.sort_values("source_row").to_dict("records"):
        movie_id = str(row["movie_id"])
        if movie_id in used: continue
        rows.append(row); used.add(movie_id)
        if len(rows) == count: break
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} unique rows, found {len(rows)} from pool of {len(frame)}")
    return pd.DataFrame(rows)


def write_question(spec: QuestionSpec, source: pd.DataFrame, used: set[str]) -> dict:
    structured = structured_mask(source, spec)
    semantic = semantic_mask(source, spec)
    # 100 unique candidates: 40 pass the structured filter and 60 are
    # structured distractors. Ground truth remains the 12 structured+semantic rows.
    selected = pd.concat([
        take_unique(source[structured & semantic], 12, used),
        take_unique(source[structured & ~semantic], 28, used),
        take_unique(source[~structured & semantic], 30, used),
        take_unique(source[~structured & ~semantic], 30, used),
    ], ignore_index=True)
    selected["structured_match"] = structured_mask(selected, spec)
    selected["semantic_label"] = semantic_mask(selected, spec)
    selected["ground_truth"] = (selected["structured_match"] & selected["semantic_label"]).astype(int)
    selected = selected.sort_values(["structured_match", "movie_id"], ascending=[False, True])
    question_id = f"q_{spec.number:02d}"
    data_dir = QUESTION_DATA_ROOT / question_id; data_dir.mkdir(parents=True, exist_ok=True)
    selected[OUTPUT_COLUMNS].to_csv(data_dir / "imdb_joined.csv", index=False)
    selected[["movie_id","title","director","year","runtime","genres"]].to_csv(data_dir / "imdb_structured_joined.csv", index=False)
    selected[["movie_id","review"]].rename(columns={"movie_id":"tconst"}).to_csv(data_dir / "imdb_reviews.csv", index=False)
    annotations = selected[["movie_id","source_row","title","year","runtime","genres","structured_match","semantic_label","ground_truth"]].copy()
    annotations["annotation_source"] = "deterministic case-insensitive lexical policy over IMDb review text"
    annotations["evidence_excerpt"] = selected["review"].str.replace(r"<br\s*/?>", " ", regex=True).str.slice(0, 240)
    annotations["annotation_rationale"] = annotations.apply(lambda row: "Included: structured and semantic conditions match." if row.ground_truth else ("Excluded by structured conditions." if not row.structured_match else "Excluded by semantic condition."), axis=1)
    annotations.to_csv(data_dir / "annotations.csv", index=False)
    truth = annotations[annotations.ground_truth.eq(1)]
    truth[["movie_id","title","ground_truth","annotation_rationale","evidence_excerpt"]].to_csv(data_dir / "ground_truth.csv", index=False)
    truth_ids = sorted(truth.movie_id.astype(str))
    benchmark = {
        "benchmark_id": f"canonical_10q_q{spec.number:02d}", "difficulty": spec.difficulty,
        "difficulty_label": spec.difficulty_label, "question": spec.question,
        "semantic_question": spec.semantic_question, "suql_query": spec.suql_query,
        "structured_filters": list(spec.structured_filters), "semantic_task": spec.semantic_task,
        "semantic_annotation_patterns": list(spec.semantic_patterns),
        "annotation_policy": "Deterministic lexical annotation; no evaluated LLM used",
        "row_count": ROWS_PER_QUESTION, "structured_candidate_count": int(selected.structured_match.sum()),
        "semantic_positive_count": int(selected.semantic_label.sum()), "ground_truth_movie_ids": truth_ids,
    }
    metadata_dir = QUESTION_ROOT / question_id
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    return benchmark


def main() -> None:
    source = load_source(); used: set[str] = set(); built = {}
    # Allocate rare true-positive pools first so broader questions cannot consume
    # the few rows needed by the most selective semantic/structured intersections.
    for spec in sorted(
        QUESTIONS,
        key=lambda item: (structured_mask(source, item) & semantic_mask(source, item)).sum(),
    ):
        built[spec.number] = write_question(spec, source, used)
    manifest = {
        "suite": "10q", "question_count": 10, "source": str(SOURCE.relative_to(LAB_ROOT)),
        "questions": [{"id": f"q_{q.number:02d}", "directory": f"q_{q.number:02d}", "catalog_directory": q.slug, "benchmark_id": built[q.number]["benchmark_id"], "difficulty": q.difficulty, "difficulty_label": q.difficulty_label, "question": q.question, "semantic_task": q.semantic_task, "structured_filters": list(q.structured_filters), "row_count": ROWS_PER_QUESTION, "structured_candidate_count": built[q.number]["structured_candidate_count"], "ground_truth_count": len(built[q.number]["ground_truth_movie_ids"])} for q in QUESTIONS],
        "dataset_policy": "Ten held-out questions independent of 1q/3q/5q, with disjoint balanced 100-movie datasets. Each has 12 true positives, 28 structured semantic negatives, 30 semantic-only distractors, and 30 double negatives. Semantic labels use declared deterministic lexical rules over review text.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    question_lines: list[str] = []
    truth_lines: list[str] = []
    for spec in QUESTIONS:
        benchmark = built[spec.number]
        question_lines.extend([
            f"Q{spec.number}: {spec.question}",
            f"Semantic task: {spec.semantic_task}",
            f"Structured filters: {json.dumps(list(spec.structured_filters))}",
            "",
        ])
        truth_lines.append(f"Q{spec.number}: {spec.question}")
        rows = pd.read_csv(QUESTION_DATA_ROOT / f"q_{spec.number:02d}" / "imdb_structured_joined.csv")
        by_id = rows.set_index("movie_id").to_dict("index")
        for movie_id in benchmark["ground_truth_movie_ids"]:
            row = by_id[movie_id]
            truth_lines.append(f"- {row['title']} ({int(row['year'])})")
        truth_lines.append("")
    (ROOT / "questions.txt").write_text("\n".join(question_lines).rstrip() + "\n")
    (ROOT / "ground_truth_movies.txt").write_text("\n".join(truth_lines).rstrip() + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
