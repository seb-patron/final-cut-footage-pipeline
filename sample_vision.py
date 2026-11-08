"""Sample one source frame locally; preserve exact source/time and classifier evidence."""
import argparse, json, subprocess
from pathlib import Path
from fractions import Fraction
from footage_keywords import catalog, save
p=argparse.ArgumentParser(description=__doc__)
for n in ('inventory','root','asset-id','output'): p.add_argument('--'+n,required=True)
p.add_argument('--seconds',required=True)
a=p.parse_args()
asset=catalog(a.inventory,a.root)[a.asset_id]
t=Fraction(a.seconds); fd=1/Fraction(asset['frame_rate'])
if not 0 <= t < Fraction(asset['duration_seconds']): raise ValueError('sample outside source')
out=Path(a.output); out.mkdir(exist_ok=False,parents=True)
frame=out/'frame.jpg'; source=Path(a.root)/asset['library_relative_path']
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss',str(float(t)),'-i',str(source),'-frames:v','1','-vf','scale=960:-2',str(frame)],check=True)
r=subprocess.run(['swift','-module-cache-path',str(out/'swift-cache'),str(Path(__file__).with_name('classify_frame.swift')),str(frame)],check=True,capture_output=True,text=True)
evidence=json.loads(r.stdout)
save(out/'classification.json',dict(evidence,asset_id=a.asset_id,source_uri=asset['source_uri'],requested_seconds=str(t)))
# Only a sampled-frame interval, never extend model claims across unseen footage.
proposals=[dict(asset_id=a.asset_id,start_seconds=str(t),end_seconds=str(min(t+fd,Fraction(asset['duration_seconds']))),label='Visual candidate - '+v['label'],confidence=v['confidence'],evidence=f"Apple Vision revision {evidence['revision']}; frame sampled at {t}s; not a human scene review",reviewed=False,approved=False) for v in evidence['results'] if v['confidence']>=0.6]
save(out/'visual-proposals.json',proposals)
print(f'{len(proposals)} unapproved visual suggestions; no Final Cut change')
