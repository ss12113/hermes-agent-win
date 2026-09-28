# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.
This root file holds only what applies everywhere. Each area has its own `AGENTS.md` (aim for
~8k chars; `agent/subdirectory_hints.py` delivers up to 32k and truncates head/tail with a warning
past that); see the **routing table** at the end and read the area file before editing in that area.

**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a messaging
gateway (Telegram, Discord, Slack, ~20 platforms), a TUI, and an Electron desktop app. It
learns across sessions (memory + skills), delegates to subagents, runs scheduled jobs, and
drives a real terminal and browser. It is extended primarily through **plugins and skills**,
not by growing the core.

Two invariants shape almost every design decision and are the lens for reviewing any change:

- **Per-conversation prompt caching is sacred.** A long-lived conversation reuses a cached
  prefix every turn. Anything that mutates past context, swaps toolsets, reloads memories, or
  rebuilds the system prompt mid-conversation invalidates that cache and multiplies the user's
  cost. We do not do it; the ONE exception is context compression. Slash commands that mutate
  system-prompt state (skills, tools, memory) must be **cache-aware**: default to deferred
  invalidation (takes effect next session) with an opt-in `--now` flag (`/skills install --now`
  is the canonical pattern).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on
  every API call, so the bar for a new *core* tool is high. New capability should arrive as a
  CLI command + skill, a service-gated tool, or a plugin — not as core surface.

## Development Environment

```bash
source ./activate   # provisions/syncs PM tools + dependencies, then activates
```
Select an isolated development `HERMES_HOME` and `HERMES_RUNTIME_DIR` first;
see `website/docs/reference/package-management.md#developer-workflow`.
PowerShell: `. .\activate.ps1`. `deactivate` restores the prior environment.
For tests, use the independent test environment in `CONTRIBUTING.md` (or Nix);
PM activation's `PYTHONPATH` does not survive the test runner's environment scrub.
`scripts/run_tests.sh` probes `.venv`, then `venv`, then `$HOME/.hermes/hermes-agent/venv`
(worktrees sharing the main checkout's venv).

## Project Structure

Counts shift constantly; the filesystem is canonical. Load-bearing entry points:

```
hermes-agent/
├── run_agent.py          # AIAgent facade; the turn loop lives in agent/turn_*.py
├── model_tools.py        # Tool orchestration, discover_builtin_tools(), handle_function_call()
├── toolsets.py           # TOOLSETS dict, _HERMES_CORE_TOOLS
├── cli.py                # HermesCLI (REPL, slash dispatch) + hermes_cli/cli_*_mixin.py
├── hermes_state.py       # SessionDB facade; hermes_state_*.py siblings
├── hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
├── hermes_logging.py     # agent.log / errors.log / gateway.log (profile-aware)
├── batch_runner.py       # Parallel batch processing
├── agent/                # turn_*.py loop phases, providers, memory, compression, prompt builder
├── hermes_cli/           # CLI subcommands, setup, config, plugins loader, skins, updater
│   └── web_routers/      # Dashboard FastAPI routers (one per surface); web_server.py mounts them
├── tools/                # Tool implementations, auto-discovered via tools/registry.py
│   └── environments/     # Terminal backends (local, docker, ssh, modal, daytona, singularity)
├── gateway/              # run.py facade + run_*.py phases + session*.py + platforms/
│   ├── platforms/        # One adapter per platform; see platforms/ADDING_A_PLATFORM.md
│   └── builtin_hooks/    # Always-registered gateway hooks (extension point; none shipped)
├── plugins/              # memory/, context_engine/, model-providers/, kanban/, image_gen/, ...
├── skills/               # Built-in skills (by category)   optional-skills/: shipped, not active
├── ui-tui/               # Ink (React) terminal UI — `hermes --tui`
├── tui_gateway/          # Python JSON-RPC backend for TUI + Desktop — server.py + methods_*.py
├── apps/desktop/         # Electron desktop app (+ apps/shared JSON-RPC client)   web/: dashboard SPA
├── acp_adapter/          # ACP server (VS Code / Zed / JetBrains)
├── cron/                 # jobs.py + scheduler.py (+ scheduler_*.py)
├── evals/                # Offline benchmarks (codebase_navigability/, compaction/, ...)
├── scripts/              # run_tests.sh, release.py, check_compat_pointers.py, ci/
├── website/              # Docusaurus docs (developer-guide/ holds the long-form area docs)
└── tests/                # Pytest suite (~39k tests / ~3.7k files, Sep 2026)
```

**User state:** `~/.hermes/config.yaml` (settings), `~/.hermes/.env` (secrets only),
`~/.hermes/logs/` (`agent.log` INFO+, `errors.log` WARNING+, `gateway.log`); all
profile-aware via `get_hermes_home()`. Browse logs with `hermes logs [--follow] [--level] [--session]`.

**Dependency chain:** `tools/registry.py` (no deps) ← `tools/*.py` (register at import) ←
`model_tools.py` (discovery) ← `run_agent.py`, `cli.py`, `batch_runner.py`, `environments/`.


Facade + siblings conventions (`<stem>_<topic>.py` decomposition, patch-where-production-reads, god-file limits, doc-sync on symbol moves) live in `.claude/rules/agent-core.md`, attached when an `agent/` path is first touched.

## Scoped Subsystem Instructions

Detailed subsystem guidance lives in `.claude/rules/*.md` (with `.hermes/rules/*.md` also supported for local projects). Hermes injects the matching rule file when a tool first touches a path covered by its `paths:` frontmatter — the text arrives in that tool's result, so it costs nothing on the startup prompt.

**Before modifying a subsystem, inspect at least one relevant implementation or test file first.** This claims its scoped rule before the edit; do not make a write/patch call as the first operation in an unfamiliar subsystem. Root `AGENTS.md`, nested `AGENTS.md`/`CLAUDE.md`, and matching scoped rules compose; none replaces the others.

## Code Shape Rules (all languages)

- No "defense-in-depth" wrappers, `try/except: pass` around code that cannot fail, or flags
  nobody sets. Docstrings/comments keep the WHY, cut the WHAT.
- **Never infer process identity from argv substrings** (`"serve" in cmdline`) — the bug class
  behind ~10 fleet-update issues (#90778, #87594, #78089, #76129, #91964). Use the canonical
  matchers `gateway.status.looks_like_gateway_command_line` and
  `hermes_cli.update_cmd._hermes_holder_subcommand`; flag sets are DERIVED from the parser
  (`_holder_value_flags()`), never hand-written; match FULL cmdlines and truncate only for
  display. Details: `hermes_cli/AGENTS.md`.
- **Never hardcode `~/.hermes`.** `get_hermes_home()` for code paths, `display_hermes_home()`
  for user-facing text (both from `hermes_constants`). Hardcoding breaks profiles (5 bugs in
  PR #3575). Profile operations themselves are HOME-anchored
  (`_get_profiles_root()` = `Path.home()/.hermes/profiles`) so `hermes -p x profile list`
  sees all profiles — intentional, not a bug.
- **One process may serve many profiles; code that runs outside a turn binds the owning
  profile scope explicitly.** A profile = home + secret scope + terminal scope, bound by
  `gateway/run.py::_profile_runtime_scope` (turn), `tui_gateway/server.py::@_profile_scoped` +
  `model_switch.py::_session_profile_runtime_scope` (RPC, teardown), `cron/scheduler_provider.py::
  _profile_cron_scope` (ticker), `gateway/run_agent_cache.py::_run_release_in_profile_scope`
  (eviction). `os.environ`, module globals and import-time values hold the *launch* profile's, so
  an unbound read is a silent default-profile leak, never an error: home/config/`.env`-derived
  module constants are a bug class — key slots by `hermes_home_key()` or resolve at call time.
  Needs a binding: boot probes (`check_fn`, MCP discovery, hooks), session end/eviction, tickers,
  deferred callbacks, RPC methods, config readers, thread hops (`spawn_context_thread`), child
  spawns (`served_profile_child_env`, never `os.environ.copy()`). Fail-closed reads exist only after
  `set_multiplex_active(True)`. Prove live with two homes (A→B→A) under multiplex, not one temp
  `HERMES_HOME`. Advisory lint: `scripts/check_profile_scope_patterns.py`.
- **Machine facts and resource lookup go through `hermes_platform`.** `hermes_platform.host` is the
  one answer for OS family, native architecture (`IsWow64Process2` → `platform.machine()`; never
  `PROCESSOR_ARCHITECTURE` alone, it reads AMD64 under x64-on-ARM64 emulation), CPU identity, and
  WSL/container/Termux. Facts are cached per process and take **no environment-variable input**, so
  a hardware recognizer (`host/products.py`) cannot be set from a shell. Distinguish the control
  host (where this Python runs) from the terminal execution target (SSH/container) and the Desktop
  client (another machine): `host.*` answers only the first. A new bare `shutil.which` or a
  hand-written known-path table outside `hermes_platform/` fails
  `tests/test_managed_runtime_resolution.py` unless allowlisted with a reason; resolvers land in
  `hermes_platform/resolver/`. Lookup never installs, downloads, or starts anything.
- **Argparse alias dispatch:** `add_parser("list", aliases=["ls"])` sets `dest` to the literal
  the user typed (`"ls"`). Dispatch must accept both (caught PTY-testing `hermes webhook ls`).
- **Don't wire in dead code without E2E validation.** Unshipped code was dead for a reason;
  E2E the real resolution chain with real imports against a temp `HERMES_HOME` first.

## Dependency Pinning Policy

All dependencies carry upper bounds (litellm compromise #2796/#2810; Mini Shai-Hulud worm,
May 2026). PyPI: `>=floor,<next_major` (`"httpx>=0.28.1,<1"`); pre-1.0: `<0.(minor+2)`
(`>=0.29,<0.32`). Git URLs: 40-char commit SHA. GitHub Actions: SHA + `# vN` comment. CI-only
Python requirements: `==exact`. A bare `>=X.Y.Z` is rejected by CI and reviewers.
After changing `pyproject.toml`, run `hermes pm lock`, re-source `./activate`, and commit
`pyproject.toml` with `uv.lock`. Reference: #2810 (bounds), #9801 (SHA pinning + audit CI).

PM owns Hermes Python dependency changes. Use `pm.sync_venv(['extra'], explicit=True)`
for declared runtime extras, `hermes pm install` for setup/sync, and `hermes pm repair`
for damaged dependencies. Do not mutate Hermes environments with raw pip or uv.
Use `pm.build_environment` for fresh build outputs and `pm.ensure_environment` for
isolated tool environments. Callers receive an interpreter or tool path, not uv.
Nix's declarative uv2nix builds and unrelated user projects remain independently owned.

The `[tool.uv] exclude-newer = "14 days"` quarantine covers **Hermes's own dependencies only**
(every registry package in core's `uv.lock`). Plugin `python_dependencies` follow the plugin's own
policy: when PM generates the plugin workspace (`pm/workspace.py::_core_release_quarantine`) the
global cutoff moves onto each core-locked package, so plugin-only packages are not filtered and a
plugin still cannot drag a core package past the window. Teknium's ruling: "plugins dont have to
abide by our 14 day rule … Only hermes' dependencies themselves have to." We recommend (not require)
plugin authors adopt their own quarantine — the developer guide and `plugin-catalog/README.md` carry
that guidance.

## Commits, Merges, PRs

- **Squash merges from stale branches silently revert recent fixes.** Before squash-merging,
  bring the branch to `main` (`git fetch origin main && git reset --hard origin/main`, re-apply
  the PR's commits). Verify with `git diff HEAD~1..HEAD` after merging — unexpected deletions
  are a red flag.
- Salvage by cherry-pick so contributor authorship survives (see rubric).
- Tests per fix: 1–2 INVARIANT tests (behaviour contract, proven red on base), never
  change-detectors; ≤ 2 tests is the salvage bar too. Reject/rewrite in salvaged diffs:
  appendages to facades, new god helpers, compat aliases, wrappers.

## Testing (applies everywhere)

**ALWAYS use `scripts/run_tests.sh`**, never bare `pytest`. It enforces CI parity: credential
vars unset, `TZ=UTC`, `LANG=C.UTF-8`, `HERMES_HOME` → temp dir, and per-file subprocess
isolation via `scripts/run_tests_parallel.py` (no xdist; workers scale with CPU count) so
module-level dicts/ContextVars cannot leak between files. Direct `pytest` on a big machine
with API keys set has caused repeated "works locally, fails in CI" incidents (and the reverse).

Prepare a test interpreter with the checkout's bootstrapped Python:

```bash
python -m pm.build_env --source . --out .venv --group dev --group test
```

This is a fresh build, not an in-place sync. If the disposable output exists,
stop its processes and intentionally remove it before regeneration. The runner
clears `PYTHONPATH`, so PM shell activation alone does not supply pytest. For a
fresh output outside the checkout, set `HERMES_PYTHON` to its interpreter.

```bash
scripts/run_tests.sh                                    # full suite
scripts/run_tests.sh tests/gateway/                     # one directory
scripts/run_tests.sh tests/agent/test_foo.py -k test_x  # runner is file-granular; -k narrows
scripts/run_tests.sh -v --tb=long                       # pytest flags pass through
```

- **Flake policy:** a failing FILE is retried once in a fresh subprocess (`--file-retries`;
  `HERMES_TEST_FILE_RETRIES=0` disables); a worker killed by signal or the file timeout is never
  retried (relaunching a runaway doubles the damage). Pass-on-retry is green but printed under `⚠ FLAKY`
  with both outputs — a bug to fix, not noise. Timing tests must not assume a quiet runner:
  wall-clock bounds ≥ 2s, event-based sync, no `assert not _wait_until(...)` races.
- **Placement mirrors the source tree.** A test lives in `tests/<top-level source dir>/` (`tests/hermes_cli/`,
  `tests/agent/`, `tests/hermes_state/`, `tests/gateway/relay/`, ...); installer/updater script tests
  under `tests/scripts/{install,desktop_update}/`. Only tests of root-level modules (`batch_runner`,
  `utils`, `hermes_constants`, packaging) sit directly in `tests/`. No issue numbers in filenames —
  cite the issue in the module docstring (`test_89315_x.py` → `test_x.py`, "Regression for #89315").
- **Placement (CI lanes):** `scripts/ci/classify_changes.py` picks jobs by changed files. A Python test
  asserting about `package.json`, `package-lock.json`, `tsconfig.json`, or `.ts/.tsx/.js/
  .mjs/.cjs` sources will not run on a JS-only PR (green on PR, red on `main` where the
  classifier fails open). Such tests belong in the vitest suite, not `tests/*.py`.
- **Tests must not write to `~/.hermes/`.** The autouse `_isolate_hermes_home` fixture in
  `tests/conftest.py` redirects `HERMES_HOME`; never hardcode `~/.hermes/` in tests. Profile
  tests also mock `Path.home()` so `_get_profiles_root()` / `_get_default_hermes_home()` stay
  in the temp dir (pattern: `tests/hermes_cli/test_profiles.py`):
  ```python
  @pytest.fixture
  def profile_env(tmp_path, monkeypatch):
      home = tmp_path / ".hermes"; home.mkdir()
      monkeypatch.setattr(Path, "home", lambda: tmp_path)
      monkeypatch.setenv("HERMES_HOME", str(home))
      return home
  ```
  Tests that `patch.object(Path, "home", ...)` must ALSO set `HERMES_HOME` — code reads the
  env var, not `Path.home()/.hermes`.

### Don't fake the host OS

Behaviour that genuinely differs per host is tested ON that host with `@pytest.mark.platforms("linux")`
/ `platforms("macos")` / `platforms("windows")`, never by patching `sys.platform`. Host-independent things stay
unmarked: pure functions that take the platform as data (`hidden_windows_child_options(opts,
is_windows=True)`) and declaration/packaging invariants ("pyproject declares `tzdata` with a
`sys_platform == 'win32'` marker"). Setting a module-level `IS_WINDOWS` flag and calling
`windows_detach_flags()` IS a fake. The line: **if the test needs the interpreter to believe it
is on another OS to pass, it belongs on that OS.** A test that walks several platforms in
sequence is split — host-native arm on Linux, other arms as their own marked tests.

One marker per test, with any number of spec strings (any-of semantics) plus
optional arch filters. To gate on several OSes, pass several specs to ONE
marker — never stack several `platforms()` decorators on one test (the
conftest rejects that at collection):

```python
@pytest.mark.platforms("linux", "macos")  # ONE marker, two specs: runs on either
def test_posix_signal_path(): ...
```

Other single-marker forms (each is a complete marker on its own):
`platforms("windows")` (native Windows only), `platforms("not macos")`
(anywhere except macOS), `platforms("windows", arch="arm64")` (native Windows
on arm64), `platforms("posix")` (Linux or macOS).

Specs: `linux`, `macos`, `windows`, `posix`, `any`, and `not <spec>`.
The historic `linux_only` / `macos_only` / `windows_only` markers have been
fully replaced — `platforms` is the only host-gating marker in the tree.

**Live Windows process-topology E2E: the `wine2e` lane.** For claims about
real Windows process behavior that mocks cannot reproduce (venv-holder
scans, process-tree parentage, launcher/worker chains, detach semantics),
there is an on-demand workflow `windows-venv-e2e.yml` that runs
`tests/hermes_cli/test_venv_holder_windows_live.py` on a real
`windows-latest` runner — spawning actual processes and driving the real
detection code, no mocked psutil. It fires ONLY on pushes to `wine2e/**`
branches (inert on PRs and main; costs nothing on normal work). The proven
workflow: write probes that pin CORRECT behavior, push to a `wine2e/`
branch to reproduce the bugs live on unfixed code, build the fix, iterate
until the lane is green, then open the PR — the live receipt on the exact
head is the Windows proof reviewers ask for. Extend the live suite when
touching that subsystem; assert against the gateway ANCESTOR found by
argv, not the direct parent (the venv shim makes every spawn a
launcher/worker chain).

**Use the marker, never a bare `skipif`.** `scripts/ci/list_os_marked_tests.py`
decides which files an OS lane imports by resolving the quoted specs inside
`platforms(...)` (`"posix"` reaches the macOS lane, `"not linux"` reaches
both others), and the lane then selects with `-m platforms` while the
conftest's per-test host skips do the actual gating. A test gated with
`@pytest.mark.skipif(sys.platform != "win32")` therefore runs on no host at
all, silently — it is never imported by the lane that would run it, and the
full-suite lanes skip it. `skipif(sys.platform == "win32")` becomes
`platforms("posix")`; a non-host condition (`os.geteuid() == 0`) stays a
separate `skipif` beside the marker. A misspelt spec is a collection error,
not a skip. Don't stack a module-level `pytestmark =
platforms(...)` on a file whose tests carry their own host marker — the
conftest hard-rejects tests carrying two `platforms()` markers (a test
skipped on every host, reported green everywhere).
Equally, don't `pytest.skip()` the non-host rows of a `@parametrize` over
platforms — split it into one marked test per OS, or only the host's row ever
executes.

### Don't write change-detector tests

A change-detector fails whenever data *expected to change* is updated — model catalogs,
`_config_version`, enumeration counts, hardcoded model lists. It adds no coverage and taxes
every routine update. Don't: `assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]`,
`assert DEFAULT_CONFIG["_config_version"] == 21`, `assert len(models) == 8`. Do: `assert
"gemini" in _PROVIDER_MODELS and len(_PROVIDER_MODELS["gemini"]) >= 1` (plumbing works);
`assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]` (migration reaches
latest); `assert not (set(moonshot_models) & coding_plan_only_models)` (no leak); every
catalog model has a context-length entry (relationship). If it reads like a snapshot, delete
it; if it reads like a contract between two pieces of data, keep it. Reviewers reject new
change-detectors; authors convert them before re-review.

### Never read source code in tests

A test that reads a `.py`/`.ts`/`.tsx` file's text tests the *shape of the source*, not
behavior — banned outright. It passes when the implementation is subtly broken (regex matches
a mis-wired call site) and fails on correct refactors; it can't run against bundled/minified
artifacts; it blocks structural cleanup; it gives false confidence. Don't
`fs.readFileSync('main.ts')` + `assert.match(source, /spawn\(...hiddenWindowsChildOptions/)`.
Do extract the logic into a pure/DI-testable function and call it:
```ts
export function hiddenWindowsChildOptions(options = {}, isWindows = process.platform === 'win32') {
  if (!isWindows || 'windowsHide' in options) return options
  return { ...options, windowsHide: true }
}
```
If the logic lives inline in a god-file and extraction feels disruptive, that is the signal to
extract, not to regex around it.

## Routing Table — working in X → read X/AGENTS.md

| Area | Read | Covers |
|---|---|---|
| `run_agent.py`, `agent/` | `agent/AGENTS.md` | AIAgent + mixins, turn phases, caching integrity, message-flow invariants, compression, model/aux resolution |
| `cli.py`, `hermes_cli/`, `main.py` | `hermes_cli/AGENTS.md` | CLI mixins, `_SLASH_DISPATCH`, slash registry, config system + loaders, skins, `hermes update` pipeline, profiles / multiplex |
| `gateway/` | `gateway/AGENTS.md` | Adapters, two message guards, streaming contract, background notifications, gateway vs desktop lifecycle, token locks, scoped secrets |
| `tools/`, `toolsets.py`, `model_tools.py` | `tools/AGENTS.md` | Adding tools, registry, toolsets, delegation, cross-tool references, backends |
| `plugins/`, `hermes_cli/plugins*.py` | `plugins/AGENTS.md` | Plugin kinds, native compat contract, in-tree policy, Sep-2026 compat window |
| `tui_gateway/`, `ui-tui/` | `tui_gateway/AGENTS.md` | Process model, JSON-RPC transport, key surfaces, slash flow, dev commands |
| `web/`, `hermes_cli/web_routers/` | `web/AGENTS.md` | Dashboard embeds the real TUI; what React may and may not rebuild |
| `apps/desktop/` | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` | Desktop judgment guide; `serve` backend, slash palette curation, Bot Mode canonical chat |
| `skills/`, `optional-skills/`, `agent/curator*.py` | `skills/AGENTS.md` | Frontmatter, HARDLINE authoring standards, curator |
| `cron/`, kanban (`hermes_cli/kanban*.py`, `tools/kanban_tools.py`, `plugins/kanban/`) | `cron/AGENTS.md` | Scheduler invariants, job fields, kanban board/dispatcher |
| `gateway/platforms/` new adapter | `gateway/platforms/ADDING_A_PLATFORM.md` | Step-by-step adapter guide |
| profiles / multiplex / secret scope (any area) | `gateway/AGENTS.md` § Profile scope, `website/docs/user-guide/multi-profile-gateways.md` § What is isolated per profile | which execution points bind scope, what is isolated per profile |

Long-form background lives in `website/docs/developer-guide/` (agent-loop, prompt-assembly,
context-compression-and-caching, gateway-internals, tools-runtime, plugins/, cron-internals,
session-storage, ...). Workflow rules (PR/issue/review/salvage process) live in the
`hermes-agent-dev` skill, not here.
