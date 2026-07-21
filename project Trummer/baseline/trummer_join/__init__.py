"""Trummer-style semantic join operators adapted for local IMDb data."""

__all__ = ["adaptive_join", "block_join"]


def __getattr__(name: str):
    if name in __all__:
        from .operators import adaptive_join, block_join

        return {"adaptive_join": adaptive_join, "block_join": block_join}[name]
    raise AttributeError(name)
