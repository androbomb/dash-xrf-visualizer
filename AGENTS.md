# Project agent rules — XRF-only Dash / Jupyter / JAX refactor

These rules apply to Hermes and any other agent working in the XRF visualizer repository. Prioritize scientific correctness, readable maintainable Python, resource limits, reproducibility and user-authorized scope over speed. **M0–M28 completed (2026-11-10/13):** data readers, JAX float64 processing, Dash UI and callbacks, calibration/line-window selection, full-field multi-line map/spectrum gallery with ZIP export, scale and ROI controls, square-pixel rectangular selection, matrix-exact PNG export, and user-confirmed notebook fixes for calibration and log-scale behavior. WSLC was user-tested; Docker and Apptainer recipes remain NOT TESTED. `operational_plan.md` is the single source of truth for exact feature decisions and the proposed layout. The legacy repository was audited read-only during M0; findings are in `scientific_notes.md` and `timeline.md`.

## Project snapshot — original user requirements (implementation status in README/timeline)

- Repository root as seen by Hermes: `/workspace/exchange/dash-xrf-visualizer`; verify first.
- Goal: **visualizer only**. Retain HDF5 XRF reading, left 2-D intensity map, right spectrum, spectral interval selection → map recomputation, spatial ROI selection → spectrum recomputation; **add** `.npz` / `.npy` support.
- **Remove AIRES** and its model/training/inference UI/dependencies after collecting baseline evidence and receiving implementation authorization. This feature-removal decision has **already** been made by the user; do not ask whether to keep AIRES. Postpone legacy custom-file ingestion.
- Preferred execution: JupyterLab notebook with a Python `path_to_file` variable, Dash built-in Jupyter integration, JAX numerical kernels with correct actual CPU/GPU backend detection.
- Target layout: minimal root; `src/xrf_visualizer/` for importable package, **separate** `data/`, `processing/`, `ui/layout.py`, `ui/callbacks.py` and `ui/figures.py` where justified; `tests/` (conventional plural), and `deploy/docker/`, `deploy/wslc/`, `deploy/apptainer/`, `deploy/uv/`. `pyproject.toml` and tracked `uv.lock` belong to repository root.
- Root-relative `dash-xrf-analyzer/` is *legacy bootstrap/reference only*: inspect read-only if useful, never depend on it in new code, and ignore in `.gitignore`. Never remove it until the user explicitly requests deletion; ignoring tracked files may require a separately approved Git index action.
- Container recipes for Docker, Microsoft's WSLC and Apptainer are documentation/build deliverables, **not execution/test claims**. Hermes can use `uv` **after explicit M0 approval** to create the local environment and Jupyter kernel, but may not install anything in M0.

## XRF-specific invariants and engineering rules

1. **M0 audit only:** verify project path and `git status`, map actual entry points/callbacks/file schema and AIRES connections, inspect only small file metadata/slices, report evidenced findings and M1 plan. NO modifications, package installs, downloads, servers, commits or pushes. STOP for approval.
2. **Scope is already decided:** don't ask the user to reapprove the basic visualizer features above. Escalate only unverified file layouts, ambiguous algorithms/units, ROI interaction semantics or other truly new functionality.
3. **Scientific fidelity:** distinguish discrete channel sums versus energy-weighted integration, axis order, calibration metadata, normalization, dtype, NaN/Inf behavior, empty selections and spatial mask semantics. Preserve read-only raw data and establish deterministic synthetic/golden tests.
4. **Real separation of concerns:** file readers and schema validation should not import Dash; numerical kernels should not import UI; layout should not perform numerical processing; callbacks bind UI state to pure transforms; Plotly figure construction is separate where it improves clarity. Do not create boilerplate submodules just to mirror a diagram.
5. **Notebook user experience:** public API accepts a Python string/Path; no file reads, server start or JAX compilation on package import. Prefer modern built-in Dash notebook integration (when supported) and account for kernel reruns, port conflicts and proxy/containers.
6. **JAX where appropriate:** use JAX for retained numerical reductions; NumPy/h5py may remain for loading, conversion and plotting. Remove PyTorch if only used by AIRES. Inspect actual JAX devices in notebook/kernel, prefer usable GPU, otherwise CPU; log backend explicitly, never promise untested acceleration. Control copies/compilations and precision.
7. **uv / deploy:** `pyproject.toml` and committed `uv.lock` in root, `.venv/` ignored. Document `uv sync`, `ipykernel`, kernel registration and notebook use once authorized; isolate Docker, Compose, WSLC `Containerfile` and Apptainer definition. No claim of tested runtime where tools are absent.
8. **Document roles:** `AGENTS.md` = stable rules; `operational_plan.md` = detailed agreed intent; `to_do.md` = near-term actions; `timeline.md` = historical evidence; `README.md` = human guide (bootstrap version to be replaced only when authorized). Synchronize meaningfully without duplicating huge text.

## 0. Bootstrap, context and document ownership

Start by identifying which of these evidence paths applies:

- **Code-first:** the repository already contains code, tests or data contracts. Inspect
  it read-only first; infer only what is supported by observable behavior and docs.
- **Plan-first:** the user supplies a research question, proposed methodology or a
  populated `operational_plan.md`. Treat it as *intended work*, not completed results.
  Critically review assumptions, literature, baseline parity and feasibility.
- **Hybrid:** both exist; compare the plan to the code and flag conflicts explicitly.

Read `operational_plan.md`, `to_do.md`, `timeline.md` and applicable project docs.
Use `scientific_notes.md` only when scientific notes are relevant. The optional
`first_prompt_general_rules.md` summarizes this file; it never supersedes it.

**Document roles and synchronization:** `AGENTS.md` defines enduring agent rules;
`operational_plan.md` defines scientific/technical intent, requirements, evaluation
and milestones; `to_do.md` tracks near-term prospective actions; `timeline.md` is an
evidence-backed retrospective; `README.md` explains the actual project to humans.
When scope changes, update the **affected** Markdown files coherently, rather than
copying long text into all of them. If `README.md` contains the marker
`BOOTSTRAP_TEMPLATE_README`, it is a temporary template instruction guide; once
initialization edits are authorized, **replace** it with a project-specific README.
If a real README already exists, preserve it and update only what is justified.

During the first interaction, inspect in read-only mode and propose a small
initialization plan. Do not rewrite docs, code or data until the user authorizes
that scope, unless the user has already explicitly requested those edits. Never
invent project facts to fill template placeholders; label unknowns as `TBD`.

## 1. Start with evidence, not implementation

- Inspect the relevant project files, `git status` (if applicable), and existing scripts before proposing changes.
- Identify the objective, available data, current workflow, constraints, and what is already implemented.
- Inspect data incrementally: filenames, schemas, shapes, types, missing values, and small samples first.
  Do not dump entire datasets into the conversation or into external services.
- For nontrivial tasks, briefly state the intended approach and an explicit definition of done.
- Reuse the project's tools and conventions; do not introduce a new architecture to solve a local problem.
- Ask for a decision when requirements, scientific methodology, or data permissions are genuinely ambiguous.

## 2. Lean, human-readable code

**Write the least code that clearly solves the problem.** Code must be easy for a researcher
or maintainer to read and change, not impressive or over-engineered.

- Use descriptive English identifiers; avoid cryptic abbreviations (standard scientific terms are fine).
  Name booleans as predicates (`is_valid`), and functions with clear verbs (`load_data`, `fit_model`).
- Follow the project language/framework conventions; for Python, follow PEP 8.
  Prefer straightforward control flow.
- Favor simple functions and standard data structures. Introduce a class only when it captures
  meaningful state, lifecycle, or behavior. Prefer composition over deep inheritance.
- A function should have one coherent responsibility; **do not** split simple logic into
  gratuitous wrappers or single-use microfunctions.
- Use Python's standard library and already-installed project dependencies first.
  Do **not** add, upgrade, or install dependencies unless necessary and explicitly approved.
  Do not reimplement reliable functionality already available in the existing stack.
- Avoid premature generalization, speculative features, plugin systems, elaborate configuration,
  excess defensive scaffolding, and compatibility layers with no demonstrated need.
- Reduce duplication where the common abstraction actually simplifies the code; a little
  repetition is preferable to a complicated abstraction.
- Keep related code close together; do not create files or packages solely to appear modular.
- Use explicit inputs/outputs, limited side effects, and no unnecessary global mutable state.
  Avoid surprising boolean mode flags when clearer interfaces are available.
- Use exceptions idiomatically: fail early on invalid inputs; catch only errors you can handle
  meaningfully; never silently swallow a failure or use broad `except` without justification.
- Comments should explain **why**, assumptions, non-obvious formulas, and scientific decisions,
  not restate the code. Use useful docstrings for public or non-obvious interfaces.
- Optimize for readability first. Optimize performance only after identifying a bottleneck.

### Mandatory simplification pass

After each meaningful implementation milestone, perform **one focused cleanup pass**:

1. Remove dead code, unused imports, duplicate logic, unnecessary helpers, and obsolete comments.
2. Simplify complex control flow and misleading names without changing validated behavior.
3. Review the diff: can any new file, abstraction, dependency, or line be avoided?
4. Run the smallest relevant verification after cleanup, then stop. Do not endlessly refactor.

### Optional Ponytail-assisted review

- When Ponytail is **already installed and available in the active agent**, use its
  lean-code principles during implementation and **prefer one focused Ponytail review**
  (`/ponytail-review`, or the equivalent available skill) after a meaningful code milestone.
- For Hermes, use Ponytail **directly only if its skill is available**; otherwise, when
  delegating coding to an OpenCode instance that has the plugin, request the review
  through OpenCode. Installing Ponytail in OpenCode does **not** automatically install
  the skill in Hermes. Do not assume the same slash commands work in both agents.
- Use `/ponytail-audit` only for an explicitly requested repository-wide audit, not
  as a default step. Never trigger repeated reviews, recursive delegation, or extra
  refactoring loops. The mandatory simplification pass and Ponytail review should be
  **the same bounded activity**, not two separate cleanup cycles.
- Treat Ponytail as a reviewer, not as an authority that overrides scientific validity,
  reproducibility, safety, existing conventions, or the agreed scope. Prefer its
  actionable findings; reject cosmetic or speculative changes that add work.
- If Ponytail is absent or unavailable, continue normally with the rules above.
  **Do not install plugins, add dependencies, or change configuration without approval.**

## 3. Scientific rigor and reproducibility

- Distinguish observations, hypotheses, implementation choices, and conclusions.
- Never invent measurements, citations, experimental results, or claims of successful execution.
- Record essential provenance: data origin/version, preprocessing, experimental conditions,
  models, hyperparameters, random seeds, and software versions when relevant.
- Avoid data leakage; keep training, validation, and test roles clear. Never use the test set
  for iterative model selection. Compare against a meaningful baseline when appropriate.
- Report metrics with units, sample sizes, uncertainty or variability where warranted,
  and limitations. Document negative or inconclusive findings rather than hiding them.
- Keep raw inputs unchanged. Write derived data and results to separate, clearly named paths.
  Do not overwrite existing outputs or human work without checking and obtaining approval.
- Prefer scripts or notebooks that can be rerun with documented inputs and commands.
  Do not create a new framework or packaging layer just for reproducibility.

## 4. Bounded testing and autonomous work

Tests and scientific checks are important, but **testing is not an end in itself**.

- Before executing a task, choose the smallest checks that demonstrate the requested behavior:
  usually a smoke test, one or two targeted checks, and a relevant scientific sanity check.
- Prefer existing tests and fixtures. Add a test only for an actual failure mode, scientific
  invariant, or meaningful regression risk; do not chase arbitrary coverage percentages.
- Run each relevant check once after a change. On failure, investigate and allow at most
  **two focused fix-and-retest cycles for the same failure**. If still unresolved, stop,
  record the evidence and blocker, and ask the user what to do next.
- Do not repeatedly run almost-identical experiments hoping for a better outcome.
  Before expensive experiments, establish a finite budget (runs, seeds, time/compute) and
  clear stopping criteria with the user if not already specified.
- Never launch unattended long-running tasks, large hyperparameter sweeps, recursive
  subagents, or indefinite loops without explicit authorization and resource limits.
- A completed milestone with documented limitations is better than uncontrolled retries.
  If the agreed acceptance criteria pass, **stop** instead of inventing more work.

## 5. Resource awareness and execution budget

**Inspect before launching; budget before scaling; monitor while running.**
Available hardware is not automatically authorized capacity. Apply these rules to local machines,
containers, WSL, shared servers, schedulers, and cloud environments. Keep checks proportional:
small inspections and bounded smoke tests need not become an elaborate profiling exercise.

### Preflight: effective resource limits

Before any nontrivial computation, data movement, or parallel job:

- **Environment and ownership:** Identify the execution context, applicable quotas or scheduler
  allocation, current resource consumers, and whether an equivalent job is already running.
  Respect shared workloads and existing jobs; never start duplicate experiments blindly.
- **CPU:** Check effective CPU availability (affinity/cpuset, cgroup quota, scheduler allocation,
  and current load), not `nproc` or host logical-core count alone. Set explicit limits for
  processes, MPI ranks, data-loader workers, and numerical-library threads. Avoid CPU
  oversubscription and nested MPI/OpenMP/BLAS parallelism unless measured to be beneficial.
- **GPU and VRAM:** For GPU workloads, inspect visible/allocated devices, current utilization,
  free VRAM, competing processes, and applicable GPU partition or scheduler limits. Estimate
  model, batch, activation, and optimizer memory before training; do not assume a visible GPU
  is idle or exclusively available.
- **RAM:** Check available system memory, process/container/job memory limits, and swap pressure
  where relevant. Estimate peak memory, including duplicated worker state, caching, and temporary
  arrays. Leave headroom; avoid configurations likely to trigger swapping or OOM termination.
- **Storage and I/O:** Check free space, quotas, and inode availability on the *actual* input,
  output, cache, temporary, and checkpoint filesystems. Estimate download, intermediate, and
  output sizes. Consider disk throughput, concurrent I/O, and shared/network-filesystem load;
  prevent unbounded logs, temporary files, and excessive small or HDF5 writes.
- **Uncertain capacity:** If effective limits or resource ownership cannot be established,
  restrict work to lightweight diagnostics or a conservative smoke test; do not launch a heavy job.

### Authorization and bounded execution

- **Resource gate:** Do not start bulk downloads, dependency/environment installations, GPU
  training, large CPU/MPI simulations, extensive sweeps, cloud uploads, or unattended long-running
  jobs without explicit user authorization of the scope and resource budget. An existing,
  sufficiently specific authorization need not be requested again. Small, bounded inspections
  and local synthetic-data checks are allowed within the current task.
- Before expensive work, establish finite bounds for relevant dimensions: runtime, concurrency,
  CPU/GPU usage, memory, disk space/I/O, transfer size, runs/seeds, and stopping criteria.
  If the estimate exceeds the approved budget, stop and request a decision; do not silently scale.
- Start with the smallest representative smoke test. Increase dataset size, batch size, worker
  count, MPI ranks, or concurrency only after checking correctness, throughput, and resource use.
  For MPI jobs, start with one thread per rank (`OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`) unless the workload has an established, measured configuration.
- When a scheduler or orchestrator is present, request and stay within its allocation; do not
  bypass quotas by launching unmanaged workers, subprocess trees, or parallel agents.

### Monitoring, recovery, and cleanup

- For substantial jobs, monitor useful progress (not CPU utilization alone), elapsed time,
  CPU/GPU load, RAM/VRAM, swapping, disk activity, output growth, and error logs. Detect stalls,
  runaway subprocesses, resource contention, and unexpected I/O amplification.
- Stop or downscale promptly on sustained stalls, resource exhaustion risk, or exceeded budgets.
  Prefer resumable outputs and appropriately spaced checkpoints; avoid checkpoint/write storms.
- Before restarting, inspect existing processes, partial outputs, logs, and checkpoints.
  Never launch a second copy or repeatedly retry expensive failures without renewed approval.
  Clean up only temporary resources owned by the task; do not kill unrelated processes.
- Record material resource settings, observed limits or bottlenecks, and the actual outcome when
  relevant to reproducibility. Report when an execution was skipped or halted for resource reasons.

## 6. Documentation: concise, scientific, maintained

- Document important decisions, assumptions, methods, results, limitations, and reproducible
  commands. Keep explanations compact but sufficient to audit the work months later.
- Prefer updating an existing genuine README or relevant document to creating another
  overlapping file. Replace a bootstrap README with a project-specific one after
  initialization is authorized; do not retain template prose in the real README.
- After a meaningful milestone, update `timeline.md` with a short, dated, factual entry.
- Revisit `operational_plan.md` when the scientific scope, method, baseline, evaluation
  or major milestones change; do not rewrite it for every small coding task.
- Maintain `to_do.md` only when a multi-step plan is useful; keep it actionable and current.
  Do not turn it into a second activity log.
- Periodically (e.g., every 3 milestones, or before changing direction) reread `timeline.md`,
  check the outstanding tasks, summarize the key findings, and prune obsolete plans.
- Ensure documentation agrees with the code and actual experiments. Never describe planned
  results as completed results.

### `timeline.md`: retrospective record

Append entries in chronological order. Each entry should include, as applicable:
- Date and milestone / question addressed.
- Changes made and relevant paths or commit references.
- Validation or experiment performed (commands, data, seeds, outcome).
- Scientific observations, decisions, limitations, and unresolved blockers.
- A concise next step, only if useful.

Do not fabricate entries, retroactively rewrite outcomes, or log trivial edits.

### `to_do.md`: prospective plan (optional)

- Keep a short list of **Now / Next / Later / Blocked** items.
- Every active task should have a clear deliverable or criterion for completion.
- Mark completed tasks and move important outcomes to `timeline.md` rather than accumulating
  an exhaustive history in `to_do.md`.
- Revise the plan when evidence changes; avoid detailed schedules without a concrete need.

## 7. Safety and scope

- Treat original datasets as read-only unless the user expressly authorizes modifications.
- Never delete, overwrite, migrate, publish, or upload files outside the agreed scope.
  Confirm before destructive commands, changing permissions, `sudo`, or network installs.
- Do not expose API keys, passwords, private identifiers, or sensitive dataset contents.
  Verify authorization before sending restricted material to external services or models.
- Do not overwrite unrelated Git changes. Inspect the diff before reporting completion.
- Do not `git push`, merge to protected branches, publish releases or open public PRs
  unless the user has authorized that external action; never commit secrets.
- Delegate to OpenCode or other agents only for a clearly bounded subtask; require a concise
  summary and review the resulting changes. Avoid uncontrolled chains of delegation.

## 8. End-of-task report

At the end of each milestone, report briefly:

1. **Done:** what changed and where.
2. **Evidence:** tests/experiments run and their actual outcomes.
3. **Caveats:** uncertainties, failures, or unverified claims.
4. **Next:** the single most useful follow-up, or say that the task is complete.

Do not claim tests ran unless they actually ran. Do not continue autonomously after the
agreed task is complete.

## References (adapted, not copied verbatim)

- https://gist.github.com/evaera/fee751d4e228dd262fe1174ba142a719
- https://www.pyopensci.org/lessons/write-better-code/clean-modular-code/index.html
- https://peps.python.org/pep-0008/
