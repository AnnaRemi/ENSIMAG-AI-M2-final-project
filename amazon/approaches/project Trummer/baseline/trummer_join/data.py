from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _default_data_root() -> Path:
    configured = os.environ.get("LAB_DATA_ROOT")
    if configured:
        return Path(configured) / "canonical"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "canonical"
        if (candidate / "amazon_reviews.csv").exists():
            return candidate
    raise FileNotFoundError("Cannot find data/canonical")


DEFAULT_SUQL_DATA = _default_data_root()


def load_products_and_reviews(
    data_dir: str | Path | None = None,
    max_products: int | None = None,
    max_reviews: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir) if data_dir else DEFAULT_SUQL_DATA
    products_path = data_path / "amazon_structured_joined.csv"
    reviews_path = data_path / "amazon_reviews.csv"

    products = pd.read_csv(products_path)
    reviews = pd.read_csv(reviews_path)

    products = products.dropna(subset=["product_id", "title"]).copy()
    reviews = reviews.dropna(subset=["product_id", "review"]).copy()

    if max_products is not None:
        products = products.head(max_products)
    if max_reviews is not None:
        reviews = reviews.head(max_reviews)

    products["join_id"] = products["product_id"].astype(str)
    reviews["join_id"] = reviews["product_id"].astype(str)

    products["text"] = products.apply(_product_text, axis=1)
    reviews["text"] = reviews.apply(_review_text, axis=1)

    return products.reset_index(drop=True), reviews.reset_index(drop=True)


def _product_text(row: pd.Series) -> str:
    return (
        f"product_id={row['product_id']}; "
        f"title={row['title']}; "
        f"brand={row.get('brand', '')}; "
        f"category={row.get('category', '')}; "
        f"price={row.get('price', '')}"
    )


def _review_text(row: pd.Series) -> str:
    review = str(row["review"]).replace("<br />", " ")
    review = " ".join(review.split())
    return f"product_id={row['product_id']}; review={review[:1400]}"
