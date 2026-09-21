# Contributing

This project uses a **branch → pull request → review → merge** workflow.
Nobody pushes to `main` directly, including the repo owner.

---

## The rule

1. **Branch off `main`.** Never commit on `main`.
2. **Open a Pull Request** when the work is ready.
3. **Request a reviewer.** A PR with no reviewer will not be merged.
4. **The reviewer merges** — not the author. If you opened it, you don't merge it.

---

## Step by step

### 1. Sync and branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-thing
```

Branch naming:

| Prefix | Use for | Example |
|---|---|---|
| `feature/` | New functionality | `feature/xgboost-baseline` |
| `fix/` | Bug fixes | `fix/suburb-name-matching` |
| `chore/` | Config, tooling, deps | `chore/add-ci-workflow` |
| `docs/` | Documentation only | `docs/data-dictionary` |

### 2. Work and test

```bash
source venv/bin/activate
pytest
```

All tests must pass before you open the PR.

### 3. Commit

Write commit messages in the imperative mood, explaining *why* where it isn't obvious:

```bash
git add <specific files>
git commit -m "Add distance-to-CBD feature to pipeline"
```

Do **not** `git add .` blindly — check `git status` first so you don't commit
`.env`, raw data or model binaries.

### 4. Push and open the PR

```bash
git push -u origin feature/your-thing
gh pr create --base main --fill --reviewer <teammate>
```

Or open it in the GitHub UI and add a reviewer from the right-hand sidebar.

### 5. Review

- The **reviewer** reads the diff, pulls the branch if needed, and either
  approves or requests changes.
- The **author** pushes follow-up commits to the same branch to address feedback.
- Once approved, the **reviewer clicks Merge** and deletes the branch.

---

## Working on several things at once

Use a git worktree so you don't have to stash:

```bash
git worktree add -b feature/other-thing ../wt-other-thing origin/main
cd ../wt-other-thing
```

Remove it when the PR is merged:

```bash
git worktree remove ../wt-other-thing
```

---

## What never gets committed

| Never commit | Why |
|---|---|
| `.env` | Contains real API keys |
| `data/raw/*` | Large source files; download them yourself |
| `models/*.pkl` | Large binaries |
| `venv/` | Machine-specific |
| `.coverage`, `htmlcov/`, `*.log` | Build artefacts |

`data/processed/` **is** committed on purpose — see `data/README.md`.

If you commit a secret by accident: rotate the key immediately, then tell the team.
Removing the file in a later commit does not remove it from history.

---

## Team

| Role | GitHub |
|---|---|
| Repo owner / reviewer | @mlasheen7 |
| Contributor | @pratiksjbrana01 |
| Contributor | @raihanhossaingalib1-a11y |
| Contributor | @thisissiddharthbisht |
