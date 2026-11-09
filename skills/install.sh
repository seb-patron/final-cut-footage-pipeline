#!/bin/sh
# Link pipeline skills into the skill folders named on the command line.
#
# Usage: sh skills/install.sh [--apply] <target-folder>...
#
# Each skill is a folder next to this script holding SKILL.md. For every
# skill and every target folder the script plans one link:
# link <target>/<name> -> <skills>/<name>.
#
# Without --apply (the default) it only prints the plan and changes
# nothing at all, not even a missing target folder. With --apply it
# creates each planned link with ln -s, making the target folder first
# with mkdir -p only when a link is about to be made.
#
# The script only ever adds new links. When <target>/<name> already
# holds a link to the same skill it reports ok and changes nothing.
# When anything else is there instead (a folder, a file, or a link to
# anywhere else, broken or not) it reports a refusal, leaves that entry
# alone, and goes on with the other skills. It never deletes, replaces,
# moves, or edits anything; it only reads the folders it is given and
# uses no network.
#
# Exit 0 when nothing was refused, 2 when any skill was refused, and 1
# with a usage line on stderr for bad arguments. With no skills beside
# this script it prints "no skills found" and exits 0.

apply=0
if [ "${1:-}" = "--apply" ]; then
    apply=1
    shift
fi
if [ "$#" -lt 1 ]; then
    echo "Usage: sh skills/install.sh [--apply] <target-folder>..." >&2
    exit 1
fi
for arg in "$@"; do
    case "$arg" in
        -*)
            echo "Usage: sh skills/install.sh [--apply] <target-folder>..." >&2
            exit 1
            ;;
    esac
done

script_dir=$(cd "$(dirname "$0")" && pwd -P)

found=0
refused=0
for skill_file in "$script_dir"/*/SKILL.md; do
    if [ ! -e "$skill_file" ]; then
        continue
    fi
    found=$((found + 1))
    name=$(basename "$(dirname "$skill_file")")
    want="$script_dir/$name"
    for target in "$@"; do
        entry="$target/$name"
        if [ -L "$entry" ]; then
            current=$(readlink "$entry")
            if [ "$current" = "$want" ]; then
                echo "ok $entry"
            else
                echo "refuse $entry: a different skill is there; leaving it alone"
                refused=$((refused + 1))
            fi
        elif [ -e "$entry" ]; then
            echo "refuse $entry: a different skill is there; leaving it alone"
            refused=$((refused + 1))
        else
            echo "link $entry -> $want"
            if [ "$apply" -eq 1 ]; then
                mkdir -p "$target"
                ln -s "$want" "$entry"
            fi
        fi
    done
done

if [ "$found" -eq 0 ]; then
    echo "no skills found"
    exit 0
fi
if [ "$refused" -gt 0 ]; then
    exit 2
fi
exit 0
