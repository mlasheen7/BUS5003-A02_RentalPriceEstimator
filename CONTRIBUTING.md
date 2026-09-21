# Contributing

We are four people working in one repository. These rules exist so we do not
overwrite each other's work.

## The rules

1. **Never commit to `main`.** Always branch.
2. **Never merge your own PR.** A teammate reviews and merges it.
3. **Never commit secrets.** No `.env`, no API keys, no credentials — ever.
4. **Pull before you branch.** `main` moves while you work.

Everyone on the team is an equal reviewer. There is no default approver and no
gatekeeper — request whoever is closest to the work, or whoever is free.

## Workflow

### 1. Start from an up-to-date `main`

```bash
git checkout main
git pull origin main
```

### 2. Create a branch

```bash
git checkout -b <type>/<short-description>
```

Branch name prefixes:

| Prefix | Use for |
|--------|---------|
| `feature/` | New functionality (`feature/xgboost-baseline`) |
| `fix/` | Bug fixes (`fix/suburb-name-matching`) |
| `chore/` | Config, tooling, dependencies (`chore/pin-dependencies`) |
| `docs/` | Documentation only (`docs/data-dictionary`) |

### 3. Work, and keep tests passing

```bash
pytest
```

If you add a module, add a test for it. If you change the pipeline, say so in the
PR — the committed contents of `data/processed/` may need regenerating.

### 4. Commit

Write messages in the imperative mood, explaining *why* rather than *what*:

```
Add SHAP feature importance to the training pipeline

The fairness audit needs per-feature contributions, not just the
built-in gain scores, so we can compare error across SEIFA groups.
```

### 5. Push and open a PR

```bash
git push -u origin <your-branch>
gh pr create --fill
```

Or open it from the GitHub web UI. Fill in the PR template, then **request a review
from at least one teammate**.

### 6. Review

- Reviews are not a formality — read the diff and actually run the branch if it
  touches code you depend on.
- Ask questions in the PR rather than fixing things yourself, so the author learns
  the codebase too.
- Approve when you are happy, then **merge it for them**.

### 7. Clean up

```bash
git checkout main
git pull origin main
git branch -d <your-branch>
```

## Reviewing someone else's PR

Check that:

- [ ] Tests pass (CI shows green)
- [ ] No secrets, `.env` files, or large binaries in the diff
- [ ] New code has a test
- [ ] File paths match what the guides document
- [ ] You understand what it does — if you do not, ask

## Who owns what

Work is split by role, but **anyone may review anything**. The four role guides live
in the repository root:

| Area | Guide | Code lives in |
|------|-------|---------------|
| Data pipeline | `Complete Implementation Guide_Data Pipeline Structure and Code` | `src/data/`, `src/features/` |
| Model training | `MODEL_TRAINING_TUNING_GUIDE.md` | `src/models/` |
| Claude API | `CLAUDE_API_SETUP_GUIDE.md` | `app/utils/api_client.py` |
| Streamlit UI | `UI_INTERFACE_SETUP_GUIDE.md` | `app/` |

If your change touches someone else's area, tag them on the PR. That is a courtesy,
not a permission requirement.

## Getting help

Post in the team channel or open an issue. If you are blocked on someone's branch,
say so early rather than duplicating their work.
