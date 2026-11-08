"""Offline, review-gated source-footage annotations. No import or network operations."""
import argparse, csv, hashlib, json, math, os, re, tempfile, zipfile
from urllib.parse import urlsplit, unquote
from pathlib import Path
from fractions import Fraction
import xml.etree.ElementTree as ET
from xml.parsers import expat


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds(value):
    value = str(value)
    return Fraction(value[:-1] if value.endswith('s') else value)


def xtime(value):
    v = Fraction(value)
    return f'{v.numerator}/{v.denominator}s'


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def catalog(inventory, root):
    rows = list(csv.DictReader(Path(inventory).open()))
    ids, paths = set(), set()
    result = {}
    root = Path(root).resolve()
    for r in rows:
        ident = r['asset_id']
        rel = Path(r['library_relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe source path')
        source = root / rel
        if ident in ids or str(source) in paths:
            raise ValueError('duplicate identity or source path')
        ids.add(ident); paths.add(str(source))
        result[ident] = dict(r, source_uri=source.as_uri())
    return result


def propose(inventory, segments, root, visual=None):
    assets = catalog(inventory, root)
    proposals = []
    for i, r in enumerate(csv.DictReader(Path(segments).open())):
        a = assets[r['asset_id']]
        if r['source_relative_path'] != a['library_relative_path'] or r['filename'] != a['filename']:
            raise ValueError('transcript/source mapping mismatch')
        start, end = seconds(r['start_seconds']), seconds(r['end_seconds'])
        valid = 0 <= start < end <= seconds(a['duration_seconds'])
        if not r['text'].strip():
            continue
        proposals.append(dict(id=f'asr-{i}', asset_id=r['asset_id'], kind='marker',
            start_seconds=str(start), end_seconds=str(end), label=r['text'].strip(),
            evidence='local Whisper ASR; NOT auditioned', flags=r['review_flags'],
            valid_range=valid, reviewed=False, approved=False))
    if visual:
        for i, r in enumerate(json.loads(Path(visual).read_text())):
            a = assets[r['asset_id']]
            start, end = seconds(r['start_seconds']), seconds(r['end_seconds'])
            if not r.get('evidence') or not r.get('label'):
                raise ValueError('visual evidence and label required')
            proposals.append(dict(r, id=f'visual-{i}', kind='keyword',
                valid_range=0 <= start < end <= seconds(a['duration_seconds']),
                reviewed=False, approved=False))
    return dict(schema=1, inventory_sha256=digest(inventory), assets=assets,
                timestamp_basis='seconds relative to source file; not timeline time', proposals=proposals)


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


def source_path(uri):
    u = urlsplit(uri)
    if u.scheme != 'file' or u.netloc or u.query or u.fragment:
        raise ValueError('local absolute file URI required')
    path = Path(unquote(u.path))
    if not path.is_absolute():
        raise ValueError('absolute source path required')
    return path


def stat_identity(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def source_state(uri):
    path = source_path(uri)
    return dict(resolved_path=str(path.resolve(strict=True)), stat=stat_identity(path.stat()))


def source_fingerprint(uri):
    path = source_path(uri)
    resolved = path.resolve(strict=True)
    with resolved.open('rb') as f:
        before = os.fstat(f.fileno())
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
        after = os.fstat(f.fileno())
    if stat_identity(before) != stat_identity(after) or stat_identity(after) != stat_identity(path.stat()):
        raise ValueError('source changed while fingerprinting')
    return dict(resolved_path=str(resolved), stat=stat_identity(after), sha256=h.hexdigest())


def target(tree, event_name, asset):
    # Whole-export rejection: never silently preserve a sidecar dependency.
    if any(e.tag in ('timeMap', 'conform-rate') for e in tree.iter()):
        raise ValueError('retiming and conform-rate unsupported')
    if any(e.tag == 'bookmark' and e.get('src') for e in tree.iter()):
        raise ValueError('external bookmark sidecar unsupported')
    if any(e.get('src') for e in tree.iter() if e.tag != 'media-rep'):
        raise ValueError('external sidecar/resource reference unsupported')
    for media in tree.findall('.//media-rep'):
        # Unselected dependencies also move with the candidate XML.
        source_path(media.get('src', ''))  # reject relative, remote or malformed URIs
    if any(e.tag not in ('asset', 'format') for e in tree.findall('./resources/*') if isinstance(e.tag, str)):
        raise ValueError('non-asset/format resources unsupported')
    for library in tree.findall('library'):
        if library.get('location'):
            source_path(library.get('location'))
    events = [e for e in tree.findall('.//event') if e.get('name') == event_name]
    if len(events) != 1 or not events[0].get('uid'):
        raise ValueError('one event with native UID required')
    assets = tree.findall('./resources/asset')
    resources = {a.get('id'): a for a in assets}
    if len(resources) != len(assets):
        raise ValueError('duplicate XML asset IDs')
    targets = []
    for clip in events[0].findall('asset-clip'):
        resource = resources.get(clip.get('ref'))
        if resource is None:
            continue
        media = resource.findall('media-rep')
        if any(m.get('src') == asset['source_uri'] and m.get('kind') == 'original-media' for m in media):
            targets.append((clip, resource))
    if len(targets) != 1:
        raise ValueError('source URI must map to exactly one browser asset-clip')
    clip, resource = targets[0]
    if not resource.get('uid') or len(resource.findall('media-rep')) != 1:
        raise ValueError('native asset UID and one original media representation required')
    # Nested media/connected clips require richer source mapping than this prototype.
    if any(e.tag in ('asset-clip','clip','ref-clip','video','audio','sync-clip','mc-clip') for e in list(clip)):
        raise ValueError('nested media unsupported')
    frame = tree.find(f"./resources/format[@id='{resource.get('format')}']")
    if frame is None or not frame.get('frameDuration') or seconds(frame.get('frameDuration')) <= 0:
        raise ValueError('positive explicit frame duration required')
    if clip.get('format', resource.get('format')) != resource.get('format'):
        raise ValueError('clip/resource format mismatch unsupported')
    if asset.get('frame_rate') and 1 / seconds(asset['frame_rate']) != seconds(frame.get('frameDuration')):
        raise ValueError('inventory/native frame rate mismatch')
    identity = dict(event_uid=events[0].get('uid'), asset_uid=resource.get('uid'),
                    source_uri=asset['source_uri'], resource_start=resource.get('start','0s'),
                    resource_duration=resource.get('duration'), clip_attributes=clip.attrib,
                    format_attributes=frame.attrib)
    return clip, resource, frame, identity


def bind(xml, proposal, event_name, asset_ids):
    """Bind selected proposals to the exact export and source bytes BEFORE review."""
    p = json.loads(Path(proposal).read_text())
    raw = xml_bytes(xml)
    tree = parse_xml_bytes(raw)
    selected = set(asset_ids)
    if not selected or not selected <= p['assets'].keys():
        raise ValueError('select known asset IDs explicitly')
    p['proposals'] = [dict(r, reviewed=False, approved=False) for r in p['proposals'] if r['asset_id'] in selected]
    p['binding'] = dict(xml_sha256=hashlib.sha256(raw).hexdigest(), event=event_name, assets={})
    for aid in sorted(selected):
        _, _, _, identity = target(tree, event_name, p['assets'][aid])
        p['binding']['assets'][aid] = dict(identity=identity, source=source_fingerprint(p['assets'][aid]['source_uri']))
    p['schema'] = 2
    return p


def same_annotation(element, kind, attrs, clip_start, clip_duration):
    if element.tag != kind:
        return False
    labels = [v.strip() for v in element.get('value','').split(',')] if kind == 'keyword' else [element.get('value')]
    return (attrs['value'] in labels
            and seconds(element.get('start', xtime(clip_start))) == seconds(attrs['start'])
            and seconds(element.get('duration', xtime(clip_duration))) == seconds(attrs['duration']))


def publish_packet(output, data, audit):
    """Publish XML+audit as ONE atomic, exclusive ZIP; never a naked partial XML."""
    output = Path(output)
    if output.suffix != '.zip':
        raise ValueError('output must be a new .zip review packet')
    if os.path.lexists(output):
        raise ValueError('output exists')
    fd, temporary = tempfile.mkstemp(prefix='.keyword-packet-', dir=output.parent)
    try:
        with os.fdopen(fd, 'w+b') as f:
            with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED) as packet:
                packet.writestr('candidate.fcpxml', data)
                packet.writestr('audit.json', json.dumps(audit, indent=2, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        # Same-filesystem hard link is atomic and fails if destination already exists.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)


def patch(xml, proposal, event_name, output):
    """Only direct event asset-clips. Reject ambiguous or unsupported source shapes."""
    p = json.loads(Path(proposal).read_text())
    raw = xml_bytes(xml)
    tree = parse_xml_bytes(raw)
    binding = p.get('binding', {})
    if binding.get('xml_sha256') != hashlib.sha256(raw).hexdigest() or binding.get('event') != event_name:
        raise ValueError('missing or stale XML binding; bind before reviewing')
    changes = []
    checked_sources = {}
    for r in p['proposals']:
        if r.get('approved') is not True:
            continue
        if r.get('reviewed') is not True or not r.get('evidence'):
            raise ValueError('approved annotation needs explicit review and evidence')
        a = p['assets'][r['asset_id']]
        start, end = seconds(r['start_seconds']), seconds(r['end_seconds'])
        if not 0 <= start < end <= seconds(a['duration_seconds']):
            raise ValueError('out-of-bounds source range')
        c, resource, frame, identity = target(tree, event_name, a)
        bound = binding.get('assets', {}).get(r['asset_id'], {})
        if a['source_uri'] not in checked_sources:
            checked_sources[a['source_uri']] = source_fingerprint(a['source_uri'])
        if bound.get('identity') != identity or bound.get('source') != checked_sources[a['source_uri']]:
            raise ValueError('stale source/native identity; bind and review again')
        fd = seconds(frame.get('frameDuration'))
        origin = seconds(resource.get('start','0s'))
        # Frame enclosure is explicit; never silently truncate to a trimmed clip.
        lo = origin + math.floor(start / fd) * fd
        hi = origin + math.ceil(end / fd) * fd
        clip_start = seconds(c.get('start', resource.get('start','0s')))
        clip_duration = seconds(c.get('duration', resource.get('duration')))
        if not clip_start <= lo < hi <= clip_start + clip_duration:
            raise ValueError('frame-aligned range exceeds browser clip; revise and review')
        if not origin <= lo < hi <= origin + seconds(resource.get('duration')):
            raise ValueError('frame-aligned range exceeds native source')
        kind = r['kind']
        if kind not in ('keyword','marker') or not r['label'].strip():
            raise ValueError('invalid annotation')
        if kind == 'keyword' and ',' in r['label']:
            raise ValueError('one keyword per proposal; commas disallowed')
        attrs = dict(start=xtime(lo), duration=xtime(hi-lo), value=r['label'])
        if any(same_annotation(e, kind, attrs, clip_start, clip_duration) for e in c):
            continue
        node = ET.Element(kind, attrs)
        # Marker items precede channel sources, filters and metadata in official DTD.
        stop = next((i for i, e in enumerate(c) if e.tag in ('audio-channel-source','filter-video','filter-video-mask','filter-audio','metadata')), len(c))
        c.insert(stop, node)
        changes.append(dict(proposal_id=r['id'], source_uri=a['source_uri'], attributes=attrs))
    if not changes:
        raise ValueError('no new reviewed, approved annotations')
    data = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + ET.tostring(tree, encoding='utf-8')
    audit = dict(input_sha256=hashlib.sha256(raw).hexdigest(), output_sha256=hashlib.sha256(data).hexdigest(), event=event_name,
                 binding=binding, annotations=changes, native_import='NOT RUN', duplicate_behavior='UNKNOWN')
    if hashlib.sha256(xml_bytes(xml)).hexdigest() != binding['xml_sha256']:
        raise ValueError('XML changed during patch')
    for uri, fingerprint in checked_sources.items():
        if source_state(uri) != {k: fingerprint[k] for k in ('resolved_path', 'stat')}:
            raise ValueError('source changed during patch')
    publish_packet(output, data, audit)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('propose')
    for name in ('inventory','segments','root','output'): q.add_argument('--'+name, required=True)
    q.add_argument('--visual')
    q = sub.add_parser('bind')
    for name in ('xml','proposal','event','output'): q.add_argument('--'+name, required=True)
    q.add_argument('--asset-id', action='append', required=True)
    q = sub.add_parser('patch')
    for name in ('xml','proposal','event','output'): q.add_argument('--'+name, required=True)
    a = p.parse_args()
    if a.command == 'propose': save(a.output, propose(a.inventory,a.segments,a.root,a.visual))
    elif a.command == 'bind': save(a.output, bind(a.xml,a.proposal,a.event,a.asset_id))
    else:
        report = patch(a.xml,a.proposal,a.event,a.output)
        print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
