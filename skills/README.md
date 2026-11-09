# Skills

A skill is a reusable recipe for one small job: a folder
`skills/<name>/` holding a file called `SKILL.md`. Both Claude Code and
Codex load a skill from that same folder, so one file serves both tools.
This folder currently holds no real skills, only this guide and the
installer described below. Each real skill arrives with its own change.

## Format

`SKILL.md` starts with a short front matter block: a first line `---`,
then exactly two lines, `name: <name>` and `description: <one line>`
in either order, then `---`. The value may sit inside double quotes.
The name equals the folder name, uses lowercase letters, digits and
dashes, and stays short. The description is one line.

The rest of the file holds plain steps, stop rules and what to report.
Commands sit in fenced code blocks tagged `sh`, `bash`, `shell`, or with
no tag. Every command line there is a pipeline call of the form
`python3 -m fcp_pipeline <step> --trip <trip> ...`, where `<trip>` is a
placeholder for the trip folder. Fenced blocks with any other tag are
examples and are never run.

Because one file loads in both tools, a skill uses only what both
support: front matter with just `name` and `description`, plain steps,
and pipeline commands. Anything only one tool understands is out: extra
front-matter keys, slash commands, argument placeholders, the shell
injection prefix, tool variables, and any command that is not a
pipeline call.

## How Codex finds skills

The repository root holds `AGENTS.md`, a short pointer that tells Codex
(and other agents) that skills live in `skills/<name>/SKILL.md`. To use
a skill, read its `SKILL.md` and follow its steps.

## Installing skills for both tools

`install.sh` links every skill into the skill folders named on its
command line, so both tools follow the same files:

```
sh skills/install.sh ~/.claude/skills ~/.codex/skills
```

Without `--apply` (the default) it only prints its plan and changes
nothing at all. When the plan looks right, run the same command with
`--apply` to create the links:

```
sh skills/install.sh --apply ~/.claude/skills ~/.codex/skills
```

The installer only adds new links and reports `ok` for links that are
already right. When something else already sits where a link would go,
it refuses that entry, leaves it alone, and goes on with the rest. It
never removes, moves, copies, overwrites or edits anything.

## Checking skills

This command checks every skill and dry-runs its commands on a fresh
synthetic trip, offline, writing nothing:

```
PYTHONDONTWRITEBYTECODE=1 python3 -m tests.skill_harness skills
```

It prints one line per skill and exits nonzero when any skill fails.
The synthetic sample under `tests/fixtures/skills/` exercises the
checker; the synthetic trip fixtures live under
`tests/fixtures/trips/synthetic/`.
