---
name: sample-dry-run
description: "A synthetic sample that exists only for the harness tests."
---

# Sample dry run

A synthetic sample skill that exercises the harness. It runs three
pipeline steps on a synthetic trip. Replace <trip> with the trip folder
before running each command.

## Steps

Run these in order:

```sh
python3 -m fcp_pipeline scan --trip <trip> --dry-run
python3 -m fcp_pipeline tag --trip <trip> --dry-run
python3 -m fcp_pipeline organize --trip <trip> --dry-run
```

## Stop rule

Stop after the organize step. Do not run any other step, and do not
rerun a step that already finished.

## Report

Report the step names in the order they ran.
