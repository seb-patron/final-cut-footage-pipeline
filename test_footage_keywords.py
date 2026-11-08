import copy, json, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch as mock
from footage_keywords import patch, parse_xml, parse_xml_bytes, bind, publish_packet
XML='''<fcpxml version="1.13"><resources><format id="r1" frameDuration="1001/30000s" width="1920" height="1080"/><asset id="r2" uid="native-source-id" name="sample" start="3600s" duration="10s" hasVideo="1" format="r1"><media-rep kind="original-media" src="SOURCE"/></asset></resources><library><event name="Keyword Validation" uid="native-event-id"><asset-clip name="sample" ref="r2" start="3600s" duration="10s"><note>preserve</note><keyword start="3600s" duration="1s" value="Existing"/><metadata><md key="test" value="keep"/></metadata></asset-clip></event></library></fcpxml>'''
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(); self.d=Path(self.t.name)
  self.media=self.d/'sample.mov'; self.media.write_bytes(b'synthetic fixture, not video')
  self.xml=self.d/'source.fcpxml'; self.original=XML.replace('SOURCE',self.media.as_uri()); self.xml.write_text(self.original)
  self.output=self.d/'out.zip'; self.path=self.d/'p.json'
  self.p={'assets':{'a':{'source_uri':self.media.as_uri(),'duration_seconds':'10'}},'proposals':[{'id':'x','asset_id':'a','kind':'keyword','start_seconds':'1','end_seconds':'2','label':'Scene - Water','evidence':'reviewed frames','reviewed':True,'approved':True}]}
 def tearDown(self): self.t.cleanup()
 def bindp(self):
  self.path.write_text(json.dumps(self.p))
  bound=bind(self.xml,self.path,'Keyword Validation',['a'])
  bound['proposals']=self.p['proposals'] # Simulate explicit subsequent review in tests.
  return bound
 def runpatch(self, bound=None):
  self.path.write_text(json.dumps(bound if bound is not None else self.bindp()))
  return patch(self.xml,self.path,'Keyword Validation',self.output)
 def write_tree(self, tree):
  import xml.etree.ElementTree as E
  self.xml.write_text(E.tostring(tree,encoding='unicode'))
 def test_origin_preservation_and_order(self):
  r=self.runpatch()
  with zipfile.ZipFile(self.output) as z:
   root=parse_xml_bytes(z.read('candidate.fcpxml')); audit=json.loads(z.read('audit.json'))
   self.assertEqual(r,audit)
  c=root.find('.//asset-clip')
  self.assertEqual(c.find('note').text,'preserve'); self.assertEqual(c.find('metadata/md').get('value'),'keep')
  self.assertEqual([x.tag for x in c],['note','keyword','keyword','metadata'])
  self.assertEqual(self.xml.read_text(),self.original)
  self.assertEqual(r['annotations'][0]['attributes']['start'],'108029029/30000s')
 def test_no_approval(self):
  self.p['proposals'][0]['approved']=False
  with self.assertRaises(ValueError): self.runpatch()
 def test_unreviewed(self):
  self.p['proposals'][0]['reviewed']=False
  with self.assertRaises(ValueError): self.runpatch()
 def test_bounds(self):
  self.p['proposals'][0]['end_seconds']='11'
  with self.assertRaises(ValueError): self.runpatch()
 def test_wrong_source(self):
  self.p['assets']['a']['source_uri']='file:///other/sample.mov'
  with self.assertRaises(ValueError): self.runpatch()
 def test_duplicate_browser_clip(self):
  root=parse_xml(self.xml);root.find('.//event').append(copy.deepcopy(root.find('.//asset-clip')));self.write_tree(root)
  with self.assertRaises(ValueError): self.runpatch()
 def test_no_overwrite(self):
  self.output.write_bytes(b'keep')
  with self.assertRaises(ValueError): self.runpatch()
  self.assertEqual(self.output.read_bytes(),b'keep')
 def test_comma_rejected(self):
  self.p['proposals'][0]['label']='A,B'
  with self.assertRaises(ValueError): self.runpatch()
 def test_semantic_keyword_dedupe_rational_and_note(self):
  root=parse_xml(self.xml);k=root.find('.//keyword');k.attrib.update(start='108029029/30000s',duration='31031/30000s',value='Other, Scene - Water',note='keep note')
  self.write_tree(root)
  with self.assertRaisesRegex(ValueError,'no new'): self.runpatch()
  self.assertFalse(self.output.exists())
 def test_whole_clip_implicit_keyword_dedupe(self):
  root=parse_xml(self.xml);root.find('./resources/format').set('frameDuration','1s')
  k=root.find('.//keyword'); k.attrib={'value':'Scene - Water','note':'preserved'};self.write_tree(root)
  self.p['proposals'][0].update(start_seconds='0',end_seconds='10')
  with self.assertRaisesRegex(ValueError,'no new'): self.runpatch()
 def test_trimmed_clip_refuses_truncation(self):
  root=parse_xml(self.xml);root.find('.//asset-clip').set('start','3604s');root.find('.//asset-clip').set('duration','2s');self.write_tree(root)
  self.p['proposals'][0].update(start_seconds='1',end_seconds='8')
  with self.assertRaisesRegex(ValueError,'exceeds browser'): self.runpatch()
 def test_utf16_entity_rejected(self):
  malicious='<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE fcpxml [<!ENTITY x "expanded">]><fcpxml>&x;</fcpxml>'
  self.xml.write_bytes(malicious.encode('utf-16'))
  with self.assertRaises(ValueError): parse_xml(self.xml)
 def test_utf32_and_utf8_entity_rejected(self):
  text='<!DOCTYPE fcpxml [<!ENTITY x "expanded">]><fcpxml>&x;</fcpxml>'
  for encoding in ('utf-32','utf-8'):
   with self.subTest(encoding=encoding),self.assertRaises(ValueError):parse_xml_bytes(text.encode(encoding))
 def test_external_dtd_rejected(self):
  with self.assertRaises(ValueError):parse_xml_bytes(b'<!DOCTYPE fcpxml SYSTEM "file:///tmp/example"><fcpxml/>')
 def test_stale_xml_rejected(self):
  p=self.bindp();self.xml.write_text(self.original+'\n')
  with self.assertRaisesRegex(ValueError,'stale XML'):self.runpatch(p)
 def test_stale_source_rejected(self):
  p=self.bindp();self.media.write_bytes(b'replaced source')
  with self.assertRaisesRegex(ValueError,'stale source'):self.runpatch(p)
 def test_missing_binding_rejected(self):
  with self.assertRaisesRegex(ValueError,'binding'):self.runpatch(self.p)
 def test_bind_clears_approval(self):
  self.path.write_text(json.dumps(self.p));p=bind(self.xml,self.path,'Keyword Validation',['a'])
  self.assertFalse(p['proposals'][0]['approved']);self.assertFalse(p['proposals'][0]['reviewed'])
 def test_retime_and_conform_rejected(self):
  import xml.etree.ElementTree as E
  for tag in ('timeMap','conform-rate'):
   root=parse_xml_bytes(self.original.encode());E.SubElement(root.find('.//asset-clip'),tag);self.write_tree(root)
   with self.subTest(tag=tag),self.assertRaisesRegex(ValueError,'retiming'):self.runpatch()
 def test_bundle_sidecar_rejected(self):
  bundle=self.d/'export.fcpxmld';bundle.mkdir();self.xml=bundle/'Info.fcpxml';self.xml.write_text(self.original)
  (bundle/'sidecar.bin').write_bytes(b'auxiliary')
  with self.assertRaisesRegex(ValueError,'sidecars'):self.runpatch()
 def test_external_bookmark_rejected(self):
  import xml.etree.ElementTree as E
  root=parse_xml(self.xml);E.SubElement(root.find('.//media-rep'),'bookmark',{'src':'sidecar.bin'});self.write_tree(root)
  with self.assertRaisesRegex(ValueError,'sidecar'):self.runpatch()
 def test_atomic_audit_serialization_failure(self):
  with self.assertRaises(TypeError):publish_packet(self.output,b'<fcpxml/>',{'bad':object()})
  self.assertFalse(self.output.exists());self.assertFalse(list(self.d.glob('.keyword-packet-*')))
 def test_atomic_publication_failure(self):
  with mock('footage_keywords.os.link',side_effect=OSError('disk failure')):
   with self.assertRaises(OSError):self.runpatch()
  self.assertFalse(self.output.exists());self.assertFalse(list(self.d.glob('.keyword-packet-*')))
 def test_atomic_concurrent_destination_not_overwritten(self):
  import os
  real_link=os.link
  def race(src,dst):
   Path(dst).write_bytes(b'other writer');return real_link(src,dst)
  with mock('footage_keywords.os.link',side_effect=race):
   with self.assertRaises(FileExistsError):self.runpatch()
  self.assertEqual(self.output.read_bytes(),b'other writer')
 def test_packet_only(self):
  self.output=self.d/'unsafe.fcpxml'
  with self.assertRaisesRegex(ValueError,'review packet'):self.runpatch()
 def test_unselected_relative_media_rejected(self):
  import xml.etree.ElementTree as E
  root=parse_xml(self.xml)
  other=E.SubElement(root.find('resources'),'asset',id='r3',uid='other',format='r1',start='0s',duration='10s')
  E.SubElement(other,'media-rep',kind='original-media',src='media/other.mov')
  self.write_tree(root)
  with self.assertRaisesRegex(ValueError,'file URI'):self.runpatch()
 def test_fingerprint_once_per_source(self):
  from footage_keywords import source_fingerprint
  bound=self.bindp()
  bound['proposals'].append(dict(bound['proposals'][0],id='second',label='Second label'))
  with mock('footage_keywords.source_fingerprint',wraps=source_fingerprint) as fp:
   self.runpatch(bound)
   self.assertEqual(fp.call_count,1)
 def test_source_changed_before_publish_rejected(self):
  from footage_keywords import source_fingerprint
  bound=self.bindp()
  def changing(uri):
   fingerprint=source_fingerprint(uri)
   self.media.write_bytes(b'changed after hash')
   return fingerprint
  with mock('footage_keywords.source_fingerprint',side_effect=changing):
   with self.assertRaisesRegex(ValueError,'changed during patch'):self.runpatch(bound)
  self.assertFalse(self.output.exists())
 def test_xml_changed_before_publish_rejected(self):
  from footage_keywords import source_fingerprint
  bound=self.bindp()
  def changing(uri):
   self.xml.write_text(self.original+'\n')
   return source_fingerprint(uri)
  with mock('footage_keywords.source_fingerprint',side_effect=changing):
   with self.assertRaisesRegex(ValueError,'XML changed'):self.runpatch(bound)
  self.assertFalse(self.output.exists())
 def test_relative_library_location_rejected(self):
  root=parse_xml(self.xml);root.find('library').set('location','../other.fcpbundle');self.write_tree(root)
  with self.assertRaisesRegex(ValueError,'file URI'):self.runpatch()
if __name__=='__main__': unittest.main()
