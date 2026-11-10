<!-- BOOTSTRAP_TEMPLATE_README: replace with the real project README after initialization. -->
# [Human read] XRF-only visualizer — Dash / Jupyter / JAX bootstrap templates

**This README describes the XRF refactoring bootstrap kit, NOT the actual legacy app.**
Copy these files into a project root (or use this repository as a template). They give
an AI coding/research agent a restrained, evidence-based way to start and maintain a
project: read first, formulate the scientific or engineering plan, agree on a small
milestone, and update documentation as real work progresses.

The labels in headings identify the intended reader:

- **`[Human read]`**: instructions for the person setting up the repository. In
  particular, first-prompt drafts are examples for the human to select and edit.
- **`[AI AGENT read]`**: bootstrap conventions for the agent. These are secondary
  to `AGENTS.md` and should not be retained in the final project README.

## [AI AGENT read] Bootstrap status — read this, not the human setup guide

This file is **not** the project's real README, and is **not** an alternative to
`AGENTS.md`. The human instructions and example first prompts below are not
independent tasks or permission to act. Read `AGENTS.md` first; use
`operational_plan.md`, `to_do.md`, and `timeline.md` for project-specific context.

During initialization, identify the applicable path: (A) existing code (document
verified reality), (B) supplied scientific/technical plan (treat as intended
scope), or (C) both (reconcile discrepancies explicitly). First inspect in
read-only mode and propose coordinated document changes. Do not edit, install,
download, start expensive jobs, set up SSH credentials, or push to GitHub unless
specifically authorized under the project's rules.

After the human authorizes bootstrap documentation changes, replace this entire
file with a genuine project README, or preserve and update an existing genuine
README. See the **`[AI AGENT read]`** sections at the end for the replacement
contract. Do not copy this guide, its marker, or its prompt examples into the
finished README.

## [Human read] What each file does

| File | Role |
| --- | --- |
| `AGENTS.md` | Persistent agent rules: scientific rigor, lean code, security, bounded testing, resources and documentation ownership. |
| `operational_plan.md` | Fixed XRF-only feature scope, proposed layout, format contract, ROI/JAX validation, deployment targets and milestones. |
| `to_do.md` | Short, prospective next steps; not a completed-work journal. |
| `timeline.md` | Brief, dated record of evidence, decisions and completed milestones. |
| `scientific_notes.md` | How to prepare mathematical/research notes under `notes/` when appropriate. |
| `first_prompt_general_rules.md` | Compact fallback instructions, with XRF-specific rules. |
| `.gitignore` | Ignores `dash-xrf-analyzer/`, caches and `.venv/`; preserves `uv.lock` and small validated fixtures. Merge with existing rules. |
| `first_prompt_telegram.md` / `telegram_message.txt` | Ready-to-copy Telegram M0 prompt with the project path currently set. |
| `README.md` | **Temporary bootstrap guide**, to be replaced with the actual project README. |

This distribution is specialized for a **code-first XRF-only application refactor**.
The actual source code was not supplied here: the agent must confirm every claim
about the legacy app from its repository. `operational_plan.md` contains proposed
requirements and milestones, not tested software.


## [Human read] Target functionality and project structure

**Already decided by the user:** the final app is **only an XRF visualizer**.
It reads HDF5 and should also read `.npy` / `.npz` under a documented schema,
shows a 2-D map on the left and an XRF spectrum on the right, recomputes the map
when a spectral interval is selected and recomputes the spectrum when a spatial
ROI is selected. **AIRES must not survive the refactor**; custom file formats
are deferred. The concrete HDF5 layout and numerical conventions are not yet verified.

The intended root is small: Markdown/project documentation, `.gitignore`,
`pyproject.toml`, committed `uv.lock`, optional `.python-version`, and one
`example.ipynb`. Source lives under `src/xrf_visualizer/` (data readers,
processing/JAX, UI layout/callbacks/figures), tests in `tests/`, deployment
recipes and instructions under `deploy/{docker,wslc,apptainer,uv}/`. The original
`dash-xrf-analyzer/` is read-only bootstrap reference; ignore in Git, do not
import in the new runtime and do not delete without permission.

**M0 remains read-only.** Later, and only after explicit approval, Hermes may
create a uv environment and register a Jupyter kernel; Docker/Compose, WSLC
and Apptainer cannot be runtime-tested by this agent and must remain labeled
`NOT TESTED` pending human validation. Reproducible project dependencies/lock
files belong in root, not duplicated under `deploy/uv/`.

## [Human read] Fast start for this legacy XRF repository

1. **Back up or branch before importing templates.** Inspect existing `README.md`,
   `AGENTS.md` and `.gitignore`; **merge** their content rather than overwriting
   project-specific information. If the repository has a real README already,
   preserve it; this bootstrap guide need not be copied over it.
2. Copy the relevant templates to the **actual repository root** (not into a
   `templates/` subdirectory); alternatively pass the files to the agent and
   explicitly approve their placement after the read-only audit.
3. Open `telegram_message.txt`, replace exactly
   `PROJECT_ROOT=/workspace/exchange/dash-xrf-visualizer` with the absolute repository path
   **as seen by Hermes**, and send the full text through Telegram. The message
   must fit in one Telegram text message (less than 4096 characters).
4. Hermes should finish **M0 with a feature mapping and stop**. The core features
   are already fixed (keep XRF visualizer, remove AIRES, defer custom reader).
   Review only outstanding technical decisions and authorize M1 separately.
5. After authorizing doc initialization, the agent should replace this bootstrap
   README with the **real README for the XRF app**; `AGENTS.md`, the plan, to-do
   and timeline should then reflect only verified facts and approved work.

**Distinguish two paths:** `PROJECT_ROOT` is the source-code directory Hermes
inspects; `path_to_file` is the *per-user XRF data file* later supplied in the
Python notebook. The paths may be on different filesystems, but the XRF data path must be accessible to the Jupyter kernel, not merely to the browser. The example
import below is a **proposed API**, not confirmed working legacy code:

```python
path_to_file = "/absolute/path/to/xrf_measurement"
from xrf_visualizer import create_app  # planned API, not yet implemented
app = create_app(path_to_file)
app.run(jupyter_mode="inline")  # built-in in Dash >= 2.11
```

JAX GPU use requires a compatible backend and drivers *in the notebook kernel's
runtime*. A machine-visible GPU by itself is not enough. Use CPU when no usable
JAX GPU backend is available; explicitly disclose device and exceptions. No
package installation is automatically authorized by these instructions.

## [Human read] Choose an initialization path

### [Human read] A. Existing codebase — documentation from evidence

1. Place the templates at the repository root **without overwriting existing
   documentation**. Where names conflict, inspect and merge deliberately.
2. Start the agent from that root. Ask for a **read-only inventory** of the code,
   dependencies, documentation, tests, data contracts, provenance and actual status.
3. Ask the agent to propose a populated `operational_plan.md`, a short `to_do.md`,
   the minimal project-specific additions to `AGENTS.md`, and a replacement README.
4. Review and approve the proposed scope before the agent writes substantive
   documentation or runs experiments. Preserve user-authored content and Git changes.
5. After approval, have the agent **synchronize** the Markdown files and replace
   this bootstrap README. Record only actual work in `timeline.md`.

**General alternative first prompt (for this XRF project prefer `telegram_message.txt`):**

```text
We are initializing documentation for the project in the current repository.

First read AGENTS.md, operational_plan.md, to_do.md, timeline.md and (if relevant)
scientific_notes.md. The current README.md is a temporary template guide, not the
project README. Inspect the repository, existing docs, git status, entry points,
tests and data interfaces in READ-ONLY mode. Do not install, download, train, edit
code/data or run expensive jobs.

From verified repository evidence, explain: (1) what exists and works, (2) what is
missing or uncertain, (3) the intended scientific/engineering purpose, and (4)
the smallest justified next milestone. Distinguish implemented behavior from
inference and proposals. Flag contradictions between code and documentation.

Propose coordinated updates to operational_plan.md, AGENTS.md, to_do.md,
timeline.md (actual events only), scientific_notes.md (if relevant), and a real
project-specific README.md replacing the bootstrap guide. Preserve valid content;
do not invent results, citations or chronology. State deliverables and acceptance
criteria, then STOP for my approval before applying changes or running jobs.
```

### [Human read] B. New project — from a research plan or scientific concept

1. Supply a **populated `operational_plan.md`** or provide the research question,
   hypothesis, references, constraints and intended methods in your first prompt.
   It is fine if the plan is incomplete; missing decisions should remain explicit.
2. Start the agent from the project root and request a critical scientific review,
   not immediate implementation. Have it distinguish prior work from the proposed
   contribution; examine assumptions, falsifiability, baselines, metrics and costs.
3. Agree on a small first milestone with a finite resource budget and stopping
   criteria before downloads, installations, training or other costly work.
4. After approving the documentation initialization, ask the agent to align
   `AGENTS.md`, `operational_plan.md`, `to_do.md`, `timeline.md` and, where useful,
   `scientific_notes.md`, and replace this README with the real project README.

**First prompt draft — plan-first (copy and edit):**

```text
We are starting [PROJECT TITLE] in the current repository.

Research concept / scientific question:
[Describe the idea, motivation, expected contribution, theoretical analogy if any,
references or existing methods to reuse, and any non-negotiable constraints.]

Read AGENTS.md, operational_plan.md, to_do.md, timeline.md and scientific_notes.md.
Treat a populated operational_plan.md and my research description as the starting
scientific specification, not as verified results. This README.md is a temporary
bootstrap guide and must eventually be replaced with the project README.

For this FIRST interaction: discuss and critique the theory and methodology.
Identify assumptions, mathematical or engineering uncertainties, relevant
literature/baselines that require verification, falsifiable hypotheses, measures of
success, resource implications and the smallest proof of concept. Challenge weak
claims instead of agreeing automatically. Clearly separate established results,
proposals, unknowns, and observed evidence.

Propose the minimal coordinated updates needed in operational_plan.md, AGENTS.md,
to_do.md, timeline.md (no fabricated history), scientific_notes.md (if useful), and
a new human-facing README.md. Do not modify files, install dependencies, download
assets or run training yet. End with one proposed milestone, acceptance criteria,
resource/authorization needs and open questions. STOP for my approval.
```

### [Human read] C. Existing code + supplied plan

Use both tracks: inspect code as evidence and treat the plan as **intended scope**.
Never silently assume they match. Report the differences and request decisions
before changing scientific objectives, existing artifacts or nontrivial code.

## [Human read] Optional: give an agent repository-scoped GitHub access (Deploy Key)

If an AI agent needs to fetch or push to **one specific GitHub repository**,
consider a dedicated SSH *Deploy Key* instead of removing the passphrase from a
personal SSH key. A deploy key belongs to **one repository**; it is read-only by
default. Enable write access only if automated pushes are genuinely required,
and retain human approval for pushes to important branches. GitHub deploy keys do
**not** expire automatically, so plan to revoke unused keys.

**This is a human setup reminder, not an agent instruction.** The agent must not
create keys, modify `~/.ssh/config`, change Git remotes, change GitHub settings,
or push simply because this section exists.

### [Human read] 1. Generate a separate key on the machine running the agent

Run the following as the **normal OS user** that will run the agent (not root).
Choose a descriptive filename for this repository; do not overwrite existing
keys. A passphrase-free key allows unattended Git access, but makes protecting
the file and limiting machine access especially important. For interactive
workflows, a passphrase plus `ssh-agent` is preferable.

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# Replace REPO_SLUG with a short, filesystem-safe project name.
KEY="$HOME/.ssh/REPO_SLUG_deploy_ed25519"
if [ -e "$KEY" ] || [ -e "$KEY.pub" ]; then
    echo "Key already exists; choose another filename." >&2
else
    (umask 077; ssh-keygen -t ed25519 -f "$KEY" -C "REPO_SLUG agent deploy key" -N "")
fi
```

Print **only the public key** to copy into GitHub:

```bash
cat "$HOME/.ssh/REPO_SLUG_deploy_ed25519.pub"
```

Never paste, commit, share, or transmit the private key (the file without `.pub`).
It must remain outside the project repository, under `~/.ssh/`.

### [Human read] 2. Register the public key in GitHub

Open **your repository → Settings → Deploy keys → Add deploy key**, supply a
recognizable title, and paste the `.pub` file contents. Leave **Allow write
access** unchecked for read-only use; check it **only** when you want the agent
to be able to push. A write-enabled deploy key is a powerful repository
credential, not a fine-grained branch permission. Repository branch protection
and the agent's push-approval policy remain separate controls.

### [Human read] 3. Assign an SSH host alias to the repository

Add this block to `~/.ssh/config` (create it if needed). Replace `REPO_SLUG` in
both the alias and key filename, and keep the config private:

```sshconfig
Host github-REPO_SLUG
    HostName github.com
    User git
    IdentityFile ~/.ssh/REPO_SLUG_deploy_ed25519
    IdentitiesOnly yes
```

```bash
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/REPO_SLUG_deploy_ed25519
```

If connecting to `github.com` for the first time, verify its SSH host-key
fingerprint against GitHub's published fingerprints **before** accepting the
host-key prompt. Do not disable SSH host-key checking.

### [Human read] 4. Point the Git remote at the alias and test read access

From the **actual project repository** (not this template repository), inspect
the existing remote first:

```bash
git remote -v
# Only after verifying OWNER/REPO and that 'origin' is the right remote:
git remote set-url origin git@github-REPO_SLUG:OWNER/REPO.git
ssh -T git@github-REPO_SLUG
git ls-remote origin HEAD
```

Replace `OWNER/REPO` with the GitHub owner and repository. The `ssh -T` command
may report successful authentication while returning exit status 1, because
GitHub does not provide interactive shell access. `git ls-remote` verifies
read access without pushing. **Do not test write access with a push unless the
user has approved the target branch and contents.** If there is no `origin`
remote yet, configure it deliberately rather than assuming one exists.

### [Human read] 5. Maintain and revoke access

A separate Deploy Key is needed for each GitHub repository; the same key cannot
be attached to multiple repositories. Remove keys you no longer need from
**Settings → Deploy keys**, and securely remove the corresponding private key
from the agent machine when appropriate. Review what the agent is allowed to
commit/push independently of its SSH access. For complex or fine-grained
permissions, consider a GitHub App instead.

Official reference: [GitHub Docs — Managing deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys).

## [AI AGENT read] What the project README must become

After approved initialization, the replacement `README.md` should serve **humans**. The `[Human read]` / `[AI AGENT read]` tags belong to this template and are **not** required in the new project README:

- Project name and short motivation; current verified status and scope.
- What is implemented versus planned, with links to `operational_plan.md`.
- Requirements, setup and reproduction/usage commands that were actually verified
  (or clearly label commands not yet tested).
- Data/asset access and licenses, if relevant; never publish secrets or credentials.
- Repository map only where it helps navigation.
- Tests or evaluation entry points, results with provenance and known limitations.
- Links to research notes, bibliography and contribution guidance as applicable.

**Remove this entire bootstrap guide, including the example first prompts and
marker comment.** If an existing project already has a genuine README, preserve it
and update only what is needed; do **not** overwrite it with a generic template.

## [AI AGENT read] Maintenance contract

- `AGENTS.md` owns **rules**; `operational_plan.md` owns **scientific and technical
  intent**; `to_do.md` owns **next steps**; `timeline.md` owns **what happened**;
  the actual README owns **human-oriented project introduction and usage**.
- On meaningful scope changes, reconcile the affected Markdown files **together**,
  with evidence and without copying the same long explanation into each.
- Never fill blank fields with plausible guesses, invent experiment outcomes,
  write fake timeline entries, or claim a README command was tested when it was not.
- Resource limits and authorization gates in `AGENTS.md` remain in force, regardless
  of which bootstrap path was chosen.

For a first discussion, optionally append the compact fallback in
`first_prompt_general_rules.md` to one of the example prompts above. It is not
necessary if the agent already loaded `AGENTS.md` correctly.
