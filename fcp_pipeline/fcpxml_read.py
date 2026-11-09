"""Hardened FCPXML reader and duplicate-annotation detection.

Moved unchanged in behavior from the ``footage_keywords`` prototype so both
prototype commands keep working exactly as before.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.parsers import expat

from fcp_pipeline.fcptime import seconds, xtime


def xml_bytes(path):
    path = Path(path)
    for parent in path.parents:
        if parent.suffix == '.fcpxmld':
            if path != parent / 'Info.fcpxml' or set(parent.iterdir()) != {path}:
                raise ValueError('bundled sidecars unsupported')
            break
    return path.read_bytes()


def parse_xml_bytes(raw):
    # Deliberately narrow encoding contract; reject UTF-16/32 before XML parsing.
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise ValueError('only UTF-8 XML supported') from error
    if '\x00' in text:
        raise ValueError('only UTF-8 XML supported')
    declaration = re.match(r"\s*<\?xml\s+[^?]*encoding=['\"]([^'\"]+)['\"]", text)
    if declaration and declaration[1].lower() not in ('utf-8', 'utf8'):
        raise ValueError('only UTF-8 XML supported')
    if '<!ENTITY' in text:
        raise ValueError('XML entities unsupported')
    # Allow only the bare FCPXML doctype, never an external or internal subset.
    stripped = text.replace('<!DOCTYPE fcpxml>', '')
    if '<!DOCTYPE' in stripped:
        raise ValueError('external/internal DTD declarations unsupported')
    # Parser-level guards are the security boundary, not byte-pattern matching.
    guard = expat.ParserCreate()
    def reject(*args):
        raise ValueError('XML entities/external resources unsupported')
    def doctype(name, system_id, public_id, internal_subset):
        if name != 'fcpxml' or system_id or public_id or internal_subset:
            reject()
    guard.StartDoctypeDeclHandler = doctype
    guard.EntityDeclHandler = reject
    guard.ExternalEntityRefHandler = reject
    guard.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    guard.Parse(text, True)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.fromstring(text, parser=parser)
    if tree.tag != 'fcpxml':
        raise ValueError('FCPXML root required')
    return tree


def parse_xml(path):
    return parse_xml_bytes(xml_bytes(path))


def same_annotation(element, kind, attrs, clip_start, clip_duration):
    if element.tag != kind:
        return False
    labels = [v.strip() for v in element.get('value', '').split(',')] if kind == 'keyword' else [element.get('value')]
    return (attrs['value'] in labels
            and seconds(element.get('start', xtime(clip_start))) == seconds(attrs['start'])
            and seconds(element.get('duration', xtime(clip_duration))) == seconds(attrs['duration']))
