## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `clemsgrs/beetle-solution`, using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Experiments

- **Research journal**: read `docs/research-journal.md` before proposing or designing an experiment, and add the result and verdict when one concludes.
- **Shared GPU**: the H200 is shared with other users. Launch training only when `nvidia-smi --query-compute-apps=pid --format=csv,noheader` prints nothing, then arm `/maindisk/clement/beetle-attempt-03-20260908/gpu_guard.py <pgid>` once our run holds the GPU; it stops our run as soon as another process appears. After a yield, wait for the user to ask before relaunching.
- **Interrupted folds restart from epoch 0**: Soma writes `best_model.pt` but never resumes from it. On relaunch into the same run id, folds that already have `metrics.json` are skipped.
