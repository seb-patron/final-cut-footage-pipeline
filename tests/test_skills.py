"""Tests for the skill dry-run harness and installer.

Every home, repository copy, target and trip is a temporary folder;
``HOME`` is set to a temporary folder for every subprocess. The real
home skill folders are never read or written.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import skill_harness
from tests.skill_harness import (
    Skill,
    SkillError,
    build_synthetic_trip,
    check_commands,
    check_skill,
    dry_run,
    load_claude,
    load_codex,
    parse_skill,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = (REPO_ROOT / "tests" / "fixtures"
              / "skills" / "sample-dry-run")
SAMPLE_SKILL = SAMPLE_DIR / "SKILL.md"
INSTALLER = REPO_ROOT / "skills" / "install.sh"
AGENTS_MD = REPO_ROOT / "AGENTS.md"

SAMPLE_STEPS = ["scan", "tag", "organize"]


def _valid_skill_text(name, extra=""):
    return ("---\n"
            "name: %s\n"
            "description: \"A temporary probe skill.\"\n"
            "---\n"
            "\n"
            "Probe body.\n"
            "\n"
            "```sh\n"
            "python3 -m fcp_pipeline scan --trip <trip> --dry-run\n"
            "```\n"
            "%s\n" % (name, extra))


class TestSkills(unittest.TestCase):
    """End-to-end checks for shared skills, the harness and installer."""

    def setUp(self):
        scratch = os.environ.get("DELEGATE_SCRATCH_DIR", None)
        try:
            if scratch is None:
                raise OSError("no scratch dir configured")
            self.base = Path(tempfile.mkdtemp(
                prefix="test-skills-", dir=scratch))
        except OSError:
            self.base = Path(tempfile.mkdtemp(prefix="test-skills-"))
        self.home = self.base / "home"
        self.home.mkdir()
        self.old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)
        self.old_dont_write = os.environ.get("PYTHONDONTWRITEBYTECODE")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    def tearDown(self):
        if self.old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self.old_home
        if self.old_dont_write is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = self.old_dont_write
        shutil.rmtree(self.base, ignore_errors=True)

    # Helpers.

    def _write_skill(self, parent, name, text):
        skill_dir = Path(parent) / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
        return skill_dir / "SKILL.md"

    def _run_installer(self, *args, installer=INSTALLER):
        env = dict(os.environ)
        env["HOME"] = str(self.home)
        return subprocess.run(
            ["sh", str(installer)] + list(args),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, timeout=300, text=True)

    def _make_repo_copy(self, dest, extra_skills=()):
        skills = Path(dest) / "skills"
        skills.mkdir(parents=True)
        shutil.copyfile(INSTALLER, skills / "install.sh")
        shutil.copyfile(AGENTS_MD, Path(dest) / "AGENTS.md")
        shutil.copytree(SAMPLE_DIR, skills / "sample-dry-run")
        for name, text in extra_skills:
            self._write_skill(skills, name, text)
        return Path(dest)

    def _snapshot_tree(self, root):
        """Map each relative path to (kind, size, mtime, link target)."""
        root = Path(root)
        snapshot = {}
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            for leaf in dirnames + filenames:
                full = Path(current) / leaf
                rel = full.relative_to(root).as_posix()
                if os.path.islink(full):
                    snapshot[rel] = ("link", os.readlink(full))
                elif full.is_dir():
                    snapshot[rel] = ("dir",)
                else:
                    stat = full.stat()
                    snapshot[rel] = ("file", stat.st_size, stat.st_mtime_ns)
        return snapshot

    # Skill format.

    def test_front_matter_valid(self):
        for path in sorted((REPO_ROOT / "skills").glob("*/SKILL.md")):
            with self.subTest(skill=path.parent.name):
                skill = parse_skill(path)
                self.assertEqual(skill.name, path.parent.name)
                self.assertTrue(skill.description.strip())
        sample = parse_skill(SAMPLE_SKILL)
        self.assertEqual(sample.name, "sample-dry-run")
        self.assertTrue(sample.description.strip())
        cases = [
            ("missing name",
             "bad-missing-name",
             "---\n"
             "description: \"Has no name line.\"\n"
             "model: something\n"
             "---\n"
             "Body.\n",
             "name"),
            ("missing description",
             "bad-missing-description",
             "---\n"
             "name: bad-missing-description\n"
             "model: something\n"
             "---\n"
             "Body.\n",
             "description"),
            ("extra key",
             "bad-extra-key",
             "---\n"
             "name: bad-extra-key\n"
             "description: \"Has an extra key.\"\n"
             "allowed-tools: Read\n"
             "---\n"
             "Body.\n",
             "allowed-tools"),

            ("name differs from folder",
             "bad-wrong-name",
             "---\n"
             "name: some-other-name\n"
             "description: \"Name differs from folder.\"\n"
             "---\n"
             "Body.\n",
             "folder"),
            ("no front matter",
             "bad-no-front-matter",
             "Just a body with no front matter.\n",
             "front matter"),
        ]
        for label, folder, text, rule in cases:
            with self.subTest(label=label):
                path = self._write_skill(self.base / "skills", folder, text)
                with self.assertRaises(SkillError) as caught:
                    parse_skill(path)
                self.assertIn("SKILL.md", str(caught.exception))
                self.assertIn(rule, str(caught.exception))

    # Shared-features rule.

    def test_no_tool_specific_features(self):
        for path in sorted((REPO_ROOT / "skills").glob("*/SKILL.md")):
            with self.subTest(skill=path.parent.name):
                self.assertEqual(check_skill(path), [])
        self.assertEqual(check_skill(SAMPLE_SKILL), [])
        allowed_tools = ("---\n"
                         "name: probe-allowed-tools\n"
                         "allowed-tools: Read\n"
                         "description: \"Probe.\"\n"
                         "---\n"
                         "Body.\n")
        cases = [
            ("allowed-tools front matter",
             "probe-allowed-tools", allowed_tools, "allowed-tools"),
            ("slash command line",
             "probe-slash-line",
             _valid_skill_text("probe-slash-line",
                               "/review the clips\n"),
             "/review"),
            ("slash command in backticks",
             "probe-slash-ticks",
             _valid_skill_text("probe-slash-ticks",
                               "Run `/compact` now.\n"),
             "/compact"),
            ("argument placeholder",
             "probe-args",
             _valid_skill_text("probe-args",
                               "Fill in $ARGUMENTS here.\n"),
             "$ARGUMENTS"),
            ("shell-injection prefix",
             "probe-inject",
             _valid_skill_text("probe-inject",
                               "Run !`date` now.\n"),
             "!`"),
            ("non-pipeline command",
             "probe-command",
             ("---\n"
              "name: probe-command\n"
              "description: \"A temporary probe skill.\"\n"
              "---\n"
              "\n"
              "```sh\n"
              "ls -la\n"
              "```\n"),
             "ls -la"),
        ]
        for label, folder, text, feature in cases:
            with self.subTest(label=label):
                path = self._write_skill(self.base / "skills", folder, text)
                findings = check_skill(path)
                self.assertTrue(findings, label)
                self.assertIn(feature, "\n".join(findings))
        with self.subTest(label="path with second slash"):
            path = self._write_skill(
                self.base / "skills", "probe-devnull",
                _valid_skill_text("probe-devnull",
                                  "The fixed path /dev/null is kept.\n"))
            self.assertEqual(check_skill(path), [])

    def test_command_block_markers_flagged(self):
        def command_text(name, command):
            return ("---\n"
                    "name: %s\n"
                    "description: \"A temporary probe skill.\"\n"
                    "---\n"
                    "\n"
                    "```sh\n"
                    "%s\n"
                    "```\n" % (name, command))

        cases = [
            ("argument placeholder",
             "probe-command-args",
             "python3 -m fcp_pipeline scan --trip $ARGUMENTS",
             "$ARGUMENTS"),
            ("numbered placeholder",
             "probe-command-dollar1",
             "python3 -m fcp_pipeline scan --trip $1",
             "$1"),
            ("claude variable",
             "probe-command-claude",
             "python3 -m fcp_pipeline scan --trip $CLAUDE_HOME",
             "$CLAUDE_"),
            ("codex variable",
             "probe-command-codex",
             "python3 -m fcp_pipeline scan --trip $CODEX_HOME",
             "$CODEX_"),
            ("shell-injection prefix",
             "probe-command-inject",
             "python3 -m fcp_pipeline scan --trip <trip> !`date`",
             "!`"),
            ("slash command in backticks",
             "probe-command-slash",
             "python3 -m fcp_pipeline scan --trip <trip> `/compact`",
             "/compact"),
        ]
        for label, folder, command, marker in cases:
            with self.subTest(label=label):
                path = self._write_skill(
                    self.base / "skills", folder,
                    command_text(folder, command))
                findings = check_skill(path)
                self.assertTrue(findings, label)
                self.assertIn(marker, "\n".join(findings))
        with self.subTest(label="clean valid command"):
            path = self._write_skill(
                self.base / "skills", "probe-command-clean",
                command_text("probe-command-clean",
                             "python3 -m fcp_pipeline scan "
                             "--trip <trip> --dry-run"))
            self.assertEqual(check_skill(path), [])

    def test_documented_invocation_writes_no_bytecode(self):
        documented = (REPO_ROOT / "skills" / "README.md").read_text(
            encoding="utf-8")
        guard = "PYTHONDONTWRITEBYTECODE=1" in documented
        tree = self.base / "bytecode-tree"
        tree.mkdir()
        ignore = shutil.ignore_patterns("__pycache__")
        for leaf in ("fcp_pipeline", "tests", "skills"):
            shutil.copytree(REPO_ROOT / leaf, tree / leaf, ignore=ignore)
        shutil.copyfile(AGENTS_MD, tree / "AGENTS.md")
        shutil.copytree(SAMPLE_DIR, tree / "skills" / "sample-dry-run")
        before = self._snapshot_tree(tree)
        self.assertFalse([rel for rel in before if "__pycache__" in rel])
        # Start the documented invocation without inheriting the guard
        # (or a redirecting PYTHONPYCACHEPREFIX) from this process, so
        # any bytecode freedom comes from the invocation itself.
        env = dict(os.environ)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        env.pop("PYTHONPYCACHEPREFIX", None)
        if guard:
            env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "tests.skill_harness", "skills"],
            cwd=tree, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok sample-dry-run", completed.stdout)
        self.assertEqual(self._snapshot_tree(tree), before)

    # Known commands.

    def test_named_commands_exist(self):
        for path in sorted((REPO_ROOT / "skills").glob("*/SKILL.md")):
            with self.subTest(skill=path.parent.name):
                self.assertEqual(check_commands(parse_skill(path)), [])
        self.assertEqual(check_commands(parse_skill(SAMPLE_SKILL)), [])
        path = self._write_skill(
            self.base / "skills", "probe-unknown-step",
            ("---\n"
             "name: probe-unknown-step\n"
             "description: \"A temporary probe skill.\"\n"
             "---\n"
             "\n"
             "```sh\n"
             "python3 -m fcp_pipeline nosuchstep --trip <trip>\n"
             "```\n"))
        findings = check_commands(parse_skill(path))
        self.assertTrue(findings)
        self.assertIn("nosuchstep", "\n".join(findings))

    # Dry run through both installed layouts.

    def test_dry_run_writes_nothing(self):
        repo = self._make_repo_copy(self.base / "repo")
        claude_skills = self.home / ".claude" / "skills"
        codex_skills = self.home / ".codex" / "skills"
        installed = self._run_installer(
            "--apply", str(claude_skills), str(codex_skills),
            installer=repo / "skills" / "install.sh")
        self.assertEqual(installed.returncode, 0, installed.stderr)
        skill_claude = load_claude(self.home, "sample-dry-run")
        skill_codex = load_codex(self.home, repo, "sample-dry-run")
        self.assertIsInstance(skill_claude, Skill)
        self.assertEqual(skill_claude, skill_codex)
        trip = build_synthetic_trip(self.base / "trip")
        before = {name: self._snapshot_tree(part)
                  for name, part in (("trip", trip),
                                     ("home", self.home),
                                     ("repo", repo))}
        steps_claude = dry_run(skill_claude, trip, home=self.home)
        steps_codex = dry_run(skill_codex, trip, home=self.home)
        self.assertEqual(steps_claude, SAMPLE_STEPS)
        self.assertEqual(steps_codex, SAMPLE_STEPS)
        for name, part in (("trip", trip),
                           ("home", self.home),
                           ("repo", repo)):
            with self.subTest(tree=name):
                self.assertEqual(self._snapshot_tree(part), before[name])

    # Installer behavior.

    def test_install_idempotent(self):
        src = self._make_repo_copy(self.base / "src")
        installer = src / "skills" / "install.sh"
        first_target = self.home / "first"
        second_target = self.home / "second"
        first = self._run_installer(
            "--apply", str(first_target), str(second_target),
            installer=installer)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = self._snapshot_tree(self.home)
        links_before = {
            str(target / "sample-dry-run"): os.readlink(target / "sample-dry-run")
            for target in (first_target, second_target)}
        second = self._run_installer(
            "--apply", str(first_target), str(second_target),
            installer=installer)
        self.assertEqual(second.returncode, 0, second.stderr)
        ok_lines = [line for line in second.stdout.splitlines()
                    if line.startswith("ok ")]
        self.assertEqual(len(ok_lines), 2)
        self.assertEqual(self._snapshot_tree(self.home), before)
        for target in (first_target, second_target):
            with self.subTest(target=str(target)):
                self.assertEqual(
                    os.readlink(target / "sample-dry-run"),
                    links_before[str(target / "sample-dry-run")])

    def test_install_refuses_other_skill(self):
        second_text = _valid_skill_text("second-skill")
        cases = [
            ("folder holding another skill",
             lambda entry: (
                 entry.mkdir(parents=True),
                 (entry / "SKILL.md").write_text(
                     "Another skill entirely.\n", encoding="utf-8")),
             lambda entry: (entry / "SKILL.md").read_text(encoding="utf-8")),
            ("symlink to another folder",
             lambda entry: os.symlink(str(self.base / "elsewhere"), entry),
             lambda entry: os.readlink(entry)),
            ("broken symlink",
             lambda entry: os.symlink(str(self.base / "nowhere"), entry),
             lambda entry: os.readlink(entry)),
        ]
        (self.base / "elsewhere").mkdir()
        for label, setup, read_back in cases:
            with self.subTest(label=label):
                work = self.base / "refuse"
                if work.exists():
                    shutil.rmtree(work)
                src = self._make_repo_copy(
                    work / "src", extra_skills=[("second-skill", second_text)])
                target = self.home / ("target-%s" % label.split()[0])
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
                entry = target / "sample-dry-run"
                setup(entry)
                before = read_back(entry)
                run = self._run_installer(
                    "--apply", str(target),
                    installer=src / "skills" / "install.sh")
                self.assertEqual(run.returncode, 2, run.stdout)
                self.assertIn("refuse", run.stdout)
                self.assertEqual(read_back(entry), before)
                second = target / "second-skill"
                self.assertTrue(second.is_symlink())
                self.assertEqual(
                    os.readlink(second),
                    str((src / "skills").resolve() / "second-skill"))

    def test_install_dry_run_default(self):
        src = self._make_repo_copy(self.base / "src")
        installer = src / "skills" / "install.sh"
        missing = self.base / "missing-target"
        planned = self._run_installer(str(missing), installer=installer)
        self.assertEqual(planned.returncode, 0, planned.stderr)
        self.assertIn("link ", planned.stdout)
        self.assertIn("sample-dry-run", planned.stdout)
        self.assertFalse(os.path.lexists(missing))
        empty = self.base / "empty-target"
        empty.mkdir()
        rerun = self._run_installer(str(empty), installer=installer)
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        self.assertIn("link ", rerun.stdout)
        self.assertEqual(list(empty.iterdir()), [])

    # Codex pointer.

    def test_agents_md_points_to_skills(self):
        text = AGENTS_MD.read_text(encoding="utf-8")
        self.assertIn("skills/<name>/SKILL.md", text)
        repo = self._make_repo_copy(self.base / "repo")
        claude_skills = self.home / ".claude" / "skills"
        codex_skills = self.home / ".codex" / "skills"
        installed = self._run_installer(
            "--apply", str(claude_skills), str(codex_skills),
            installer=repo / "skills" / "install.sh")
        self.assertEqual(installed.returncode, 0, installed.stderr)
        skill = load_codex(self.home, repo, "sample-dry-run")
        self.assertEqual(skill.name, "sample-dry-run")
        self.assertEqual(skill, load_claude(self.home, "sample-dry-run"))


if __name__ == "__main__":
    unittest.main()
