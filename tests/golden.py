"""Golden-file checks for FCPXML documents.

This module implements one behavior: comparing an FCPXML document against a
stored golden file. No exporter uses it yet.

The public helpers are:

- :func:`canonicalize`: render FCPXML bytes into a stable text form that
  ignores attribute order, insignificant whitespace, generated ids and
  uids, and per-test temporary-folder file URIs.
- :func:`assert_golden`: compare a document with its golden file, which
  lives under ``tests/golden/``.
- :func:`assert_dtd_valid`: validate a file through the offline check in
  ``fcp_pipeline/bridge/check.py``, which reads xmllint's verdict against
  the installed copy of the Final Cut Pro DTD for the file's own version.
  The Apple DTD is never committed; only the installed copy is read, by
  xmllint.
"""

import difflib
import os
import re
import tempfile
import unittest
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from fcp_pipeline.bridge.check import CANNOT_VERIFY, VALID, check_fcpxml

DEFAULT_DTD_PATH = (
    "/Applications/Final Cut Pro.app/Contents/Frameworks/"
    "Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd"
)

NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

# IDREF attributes declared by the Final Cut Pro 1.14 DTD. Every other
# attribute value is compared exactly (apart from temp-folder URIs).
IDREF_ATTRIBUTE_NAMES = frozenset({"format", "ref", "projectRef", "tracking", "dataLocator"})

# Elements whose uid is generated per export get "UID". An effect's uid
# names its Motion template, so it is compared exactly.
NORMALIZED_UID_TAGS = frozenset({"event", "project", "media", "asset"})


def _parse(xml_bytes):
    builder = ET.TreeBuilder(insert_comments=True)
    parser = ET.XMLParser(target=builder)
    return ET.fromstring(xml_bytes, parser=parser)


def _blank(text):
    return text is None or text.strip() == ""


def _renumber_ids(root):
    """Map every id value to r1, r2, ... in document order."""
    mapping = {}
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        old = element.attrib.get("id")
        if old is not None and old not in mapping:
            mapping[old] = "r%d" % (len(mapping) + 1)
    return mapping


def _rewrite_idref(value, mapping):
    if value in mapping:
        return mapping[value]
    parts = value.split()
    if any(part in mapping for part in parts):
        return " ".join(mapping.get(part, part) for part in parts)
    return value


def _normalize_tmp_uri(value):
    """Collapse a temp-folder file URI to file:///GOLDEN_TMP/<rest>.

    Only the temp folder plus the first path component below it (the
    random per-test folder) is replaced; the rest of the path is kept.
    """
    if not value.startswith("file://"):
        return value
    split = urllib.parse.urlsplit(value)
    if split.scheme != "file":
        return value
    try:
        decoded = urllib.parse.unquote(split.path)
    except Exception:
        return value
    tmp = tempfile.gettempdir()
    try:
        real_tmp = os.path.realpath(tmp)
    except Exception:
        real_tmp = tmp
    try:
        real_path = os.path.realpath(decoded)
    except Exception:
        real_path = decoded
    remainder = None
    for base, candidate in ((tmp, decoded), (real_tmp, decoded),
                            (tmp, real_path), (real_tmp, real_path)):
        stripped_base = base.rstrip("/") or "/"
        if candidate == stripped_base:
            remainder = ""
            break
        if candidate.startswith(stripped_base + "/"):
            remainder = candidate[len(stripped_base) + 1:]
            break
    if remainder is None:
        return value
    components = [part for part in remainder.split("/") if part != ""]
    if len(components) <= 1:
        new_path = "/GOLDEN_TMP"
    else:
        new_path = "/GOLDEN_TMP/" + "/".join(components[1:])
    rebuilt = "file://" + split.netloc + urllib.parse.quote(new_path, safe="/")
    if split.query:
        rebuilt += "?" + split.query
    if split.fragment:
        rebuilt += "#" + split.fragment
    return rebuilt


def _normalize_tree(root):
    mapping = _renumber_ids(root)
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        attrib = element.attrib
        if attrib.get("id") in mapping:
            attrib["id"] = mapping[attrib["id"]]
        for attr_name in list(attrib):
            if attr_name == "id":
                continue
            value = attrib[attr_name]
            if attr_name in IDREF_ATTRIBUTE_NAMES:
                value = _rewrite_idref(value, mapping)
                attrib[attr_name] = value
            if attr_name == "uid" and element.tag in NORMALIZED_UID_TAGS:
                attrib[attr_name] = "UID"
                value = "UID"
            if value.startswith("file://"):
                attrib[attr_name] = _normalize_tmp_uri(value)
    return root


def _append_element(element, level, lines):
    indent = "  " * level
    if element.tag is ET.Comment:
        lines.append("%s<!--%s-->" % (indent, element.text or ""))
        return
    attributes = "".join(
        " %s=%s" % (name, quoteattr(value))
        for name, value in sorted(element.attrib.items())
    )
    children = list(element)
    if _blank(element.text):
        text = element.text if (not children and element.text) else None
    else:
        text = element.text
    if not children and text is None:
        lines.append("%s<%s%s/>" % (indent, element.tag, attributes))
        return
    if not children:
        lines.append("%s<%s%s>%s</%s>" % (indent, element.tag, attributes,
                                         escape(text), element.tag))
        return
    lines.append("%s<%s%s>" % (indent, element.tag, attributes))
    if text is not None:
        lines.append("%s%s" % ("  " * (level + 1), escape(text)))
    for child in children:
        _append_element(child, level + 1, lines)
        tail = None if _blank(child.tail) else child.tail
        if tail is not None:
            lines.append("%s%s" % ("  " * (level + 1), escape(tail)))
    lines.append("%s</%s>" % (indent, element.tag))


def canonicalize(xml_bytes):
    """Return the stable text form of an FCPXML document.

    Attribute order, whitespace-only text between elements, generated
    ids/uids, and temp-folder file URIs are normalized away; the XML
    declaration and DOCTYPE are not compared. Element order, element
    text, comments, and every other attribute value are compared.
    """
    root = _normalize_tree(_parse(xml_bytes))
    lines = []
    _append_element(root, 0, lines)
    return "\n".join(lines) + "\n"


def _describe(element):
    if element.tag is ET.Comment:
        return "#comment"
    if isinstance(element.tag, str):
        return element.tag
    return str(element.tag)


def _effective_text(element):
    if _blank(element.text):
        if len(element) == 0 and element.text:
            return element.text
        return None
    return element.text


def _effective_tail(element):
    return None if _blank(element.tail) else element.tail


def _walk_difference(actual, expected, path):
    actual_tag = _describe(actual)
    if actual_tag != _describe(expected):
        return (actual_tag, path)
    if sorted(actual.attrib.items()) != sorted(expected.attrib.items()):
        return (actual_tag, path)
    if _effective_text(actual) != _effective_text(expected):
        return (actual_tag, path)
    actual_children = list(actual)
    expected_children = list(expected)
    if len(actual_children) != len(expected_children):
        return (actual_tag, path)
    for actual_child, expected_child in zip(actual_children, expected_children):
        child_path = "%s/%s" % (path, _describe(expected_child))
        if _effective_tail(actual_child) != _effective_tail(expected_child):
            return (_describe(actual_child), child_path)
        found = _walk_difference(actual_child, expected_child, child_path)
        if found is not None:
            return found
    return None


def _first_difference(xml_bytes, expected_text):
    actual_root = _normalize_tree(_parse(xml_bytes))
    expected_root = _parse(expected_text.encode("utf-8"))
    found = _walk_difference(actual_root, expected_root, _describe(expected_root))
    if found is not None:
        return found
    return (_describe(actual_root), _describe(actual_root))


def assert_golden(xml_bytes, name, *, golden_dir=None):
    """Compare a document against its golden file.

    The golden file is ``<golden_dir>/<name>.fcpxml``; when golden_dir is
    None, ``tests/golden/`` next to this module is used. Equal canonical
    text returns quietly. With GOLDEN_UPDATE=1 the golden is rewritten
    instead of compared.
    """
    if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("invalid golden name %r; use only [A-Za-z0-9_-]+" % (name,))
    if golden_dir is None:
        directory = Path(__file__).resolve().parent / "golden"
    else:
        directory = Path(golden_dir)
    golden_path = directory / ("%s.fcpxml" % name)
    # One broad handler for the whole read-decode-parse step for both the
    # candidate and the golden, so no variant (OSError, undecodable bytes,
    # malformed XML, unknown encoding) escapes raw. Catches Exception, not
    # BaseException, and chains the original error. diff holds whatever
    # comparison was available before the failure.
    diff = ""
    try:
        canonical = canonicalize(xml_bytes)
        if os.environ.get("GOLDEN_UPDATE") == "1":
            directory.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(canonical, encoding="utf-8")
            return None
        expected = golden_path.read_text(encoding="utf-8")
        if canonical == expected:
            return None
        diff = "".join(difflib.unified_diff(
            expected.splitlines(keepends=True),
            canonical.splitlines(keepends=True),
            fromfile=str(golden_path),
            tofile="actual",
        ))
        tag, path = _first_difference(xml_bytes, expected)
    except Exception as exc:
        if isinstance(exc, AssertionError):
            raise
        hint = ""
        if isinstance(exc, FileNotFoundError):
            hint = "; rerun with GOLDEN_UPDATE=1 to create it"
        raise AssertionError(
            "golden check failed for %s%s: %s\n%s" % (golden_path, hint, exc, diff)
        ) from exc
    raise AssertionError(
        "golden mismatch for %s: first difference at <%s> (path %s)\n%s"
        % (golden_path, tag, path, diff)
    )


def _format_check_problem(problem):
    """Render one check problem with its line and element when known."""
    location = ""
    if problem.line is not None:
        location += "line %s" % (problem.line,)
    if problem.element is not None:
        if location:
            location += " "
        location += "element %s" % (problem.element,)
    if location:
        return "%s: %s" % (location, problem.message)
    return problem.message


def assert_dtd_valid(path):
    """Validate a file through the offline FCPXML check.

    Calls :func:`fcp_pipeline.bridge.check.check_fcpxml` with media
    checking off, so a sample whose media ``src`` is a placeholder still
    passes. The DTD path comes from FCP_DTD and overrides the DTD file;
    otherwise the DTD for the file's own version comes from FCP_DTD_DIR
    or the installed copy. Returns None when the file validates. Raises
    unittest.SkipTest naming the missing DTD file, folder or xmllint when
    the check cannot verify, and AssertionError naming the path and every
    problem (element and line where known) when it is invalid or refused.
    """
    override = os.environ.get("FCP_DTD")
    dtd_path = Path(override) if override else None
    if dtd_path is not None and not dtd_path.is_file():
        raise unittest.SkipTest("Final Cut DTD not found: %s" % dtd_path)
    result = check_fcpxml(path, check_media=False, dtd_path=dtd_path)
    if result.status == VALID:
        return None
    if result.status == CANNOT_VERIFY:
        reason = "; ".join(problem.message for problem in result.problems)
        if not reason:
            reason = "no reason recorded"
        raise unittest.SkipTest(
            "cannot verify %s: %s" % (path, reason))
    details = "; ".join(
        _format_check_problem(problem) for problem in result.problems)
    if not details:
        details = result.status
    raise AssertionError(
        "%s failed the offline FCPXML check: %s" % (path, details))
