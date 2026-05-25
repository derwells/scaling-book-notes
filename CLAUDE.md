# CLAUDE.md — Scaling Book Notes

## What This Is

Personal exercise notebooks and notes for "How to Scale Your Model" (jax-ml/scaling-book). The book lives in `scaling-book/` as a git submodule — don't modify it directly.

## Structure

```
scaling-book/     # submodule (read-only reference) — the original book repo
notebooks/        # Marimo notebooks with exercises and explorations
```

## The Book

The submodule tracks `git@github.com:jax-ml/scaling-book.git`. Chapter markdown files are at the submodule root — read them before answering questions about book content. See `scaling-book/CLAUDE.md` for the full chapter map and content conventions.

## Notebooks

- Format: Marimo (`.py` files, not `.ipynb`)
- Name notebooks by chapter: e.g., `ch01_roofline.py`, `ch05_training.py`
- Keep exercises self-contained — each notebook should run independently

## Tech

- **Python env:** managed via `pyproject.toml`
- **Key deps:** marimo, jax, numpy
- **Run a notebook:** `marimo edit notebooks/<file>.py`
