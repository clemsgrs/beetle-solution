## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `clemsgrs/beetle-solution`, using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Experiments

- **Research journal**: read `docs/research-journal.md` before proposing or designing an experiment, and add the result and verdict when one concludes.
- **Shared GPU**: the H200 is shared with other users. Start a GPU job only when `nvidia-smi --query-compute-apps=pid --format=csv,noheader` prints nothing, then arm `/maindisk/clement/tools/gpu_guard.py <pgid> --log <job-dir>/gpu_guard.log` once the job holds the GPU. The guard counts every GPU process present at arming as ours and stops the job when any new one appears, so keep a job's GPU work in one process. After a yield, wait for the user to ask before relaunching.
- **Interrupted folds restart from epoch 0**: Soma writes `best_model.pt` but never resumes from it. On relaunch into the same run id, folds that already have `metrics.json` are skipped.
