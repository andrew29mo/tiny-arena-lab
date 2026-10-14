"""Render verified allocations as portable SVGs and a dependency-free HTML viewer."""
import argparse
from html import escape
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

PALETTE=['#2563eb','#0d9488','#d97706','#9333ea','#dc2626','#0284c7','#65a30d','#db2777']


def svg(name,record):
    graph=record['graph'];plans=record['plans'];n=len(graph['nodes'])
    base=plans['separate']['arena_bytes'];used=plans['reuse']['arena_bytes']
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="1080" viewBox="0 0 1160 1080" role="img" aria-labelledby="title desc">',
        '<title id="title">'+escape(name)+' tensor lifetimes and arena layout</title>',
        '<desc id="desc">Separate and reused layouts share the first two byte axes. A third panel zooms reused storage. Lifetimes are inclusive; outputs are pinned.</desc>',
        '<rect width="1160" height="1080" fill="#f8fafc"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.tick{font-size:12px;fill:#526176}.bar:hover{fill-opacity:1;stroke:#111827;stroke-width:2}</style>',
        '<text x="40" y="38" font-size="25" font-weight="700">'+escape(name)+' · static arena layout</text>',
        '<text x="40" y="65" font-size="15">Separate '+str(base)+' B → reused '+str(used)+' B · '+
        ('%.1f%%'%(100*(1-used/base)))+' less tensor storage · live-byte lower bound '+str(plans['reuse']['live_lower_bound_bytes'])+' B</text>',
        '<text x="40" y="90" font-size="13">64-byte aligned blocks · inclusive last-read lifetimes · pinned outputs have a dark border · hover for exact values</text>']
    left,width,height=115,1000,205
    for index,(mode,scale,label) in enumerate([
        ('separate',base,'Separate buffers — full byte scale'),
        ('reuse',base,'Reused buffers — SAME byte scale as separate'),
        ('reuse',used,'Reused detail — INDEPENDENT byte scale (read axis labels)')]):
        top=145+index*295;plan=plans[mode]
        parts.append('<text x="40" y="%d" font-size="17" font-weight="700">%s</text>'%(top-20,label))
        parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="white" stroke="#cbd5e1"/>'%(left,top,width,height))
        for tick in range(5):
            y=top+height-tick*height/4;value=scale*tick/4
            parts.append('<path d="M %g %g H %g" stroke="#e2e8f0"/>'%(left,y,left+width))
            parts.append('<text class="tick" x="%g" y="%g" text-anchor="end">%g B</text>'%(left-10,y+4,value))
        step=max(1,math.ceil((n+1)/10))
        for t in sorted(set(list(range(0,n+1,step))+[n])):
            x=left+t*width/(n+1)
            parts.append('<path d="M %g %g V %g" stroke="#e2e8f0"/>'%(x,top,top+height))
            parts.append('<text class="tick" x="%g" y="%g" text-anchor="middle">%d</text>'%(x,top+height+19,t))
        parts.append('<text class="tick" x="615" y="%g" text-anchor="middle">Execution step (last cell retains final outputs)</text>'%(top+height+41))
        for i,node in enumerate(graph['nodes']):
            a,b=plan['birth'][i],plan['death'][i];offset,size=plan['offset'][i],plan['bytes'][i]
            x=left+a*width/(n+1);w=(b-a+1)*width/(n+1)
            y=top+height-(offset+size)*height/scale;h=size*height/scale
            pinned=i in graph['outputs'];color=PALETTE[i%len(PALETTE)]
            tooltip='T%d %s %dx%d | lifetime [%d,%d] | offset %d B | padded %d B | inputs %s/%s%s'%(
                i,['INPUT','DENSE','RELU','ADD'][node['op']],node['rows'],node['cols'],a,b,offset,size,
                node['a'],node['b'],' | pinned output' if pinned else '')
            parts.append('<rect class="bar" data-panel="%d" data-tensor="%d" x="%.4f" y="%.4f" width="%.4f" height="%.4f" fill="%s" fill-opacity=".78" stroke="%s" stroke-width="%s"><title>%s</title></rect>'%(
                index,i,x,y,w,h,color,'#111827' if pinned else 'white','2' if pinned else '.6',escape(tooltip)))
            if w>34 and h>13:
                parts.append('<text x="%.4f" y="%.4f" text-anchor="middle" font-size="11" style="fill:white;pointer-events:none">T%d</text>'%(x+w/2,y+h/2+4,i))
    parts.append('<text x="40" y="1040" font-size="13">Tensor arena only. Graph/weights, plan metadata, planner scratch, allocator overhead and process RSS are separate.</text>')
    parts.append('<text x="40" y="1062" font-size="13">The greedy heuristic can exceed the live-byte lower bound; this chart does not claim global optimality.</text></svg>')
    return '\n'.join(parts)+'\n'


def render(directory,output):
    directory=Path(directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((directory/'manifest.json').read_text())
    checks=json.loads((directory/'correctness.json').read_text())
    for name,record in checks.items():
        text=svg(name,record);ET.fromstring(text)
        (output/(name+'.svg')).write_text(text)
    cards=[]
    for name,record in checks.items():
        a,b=(record['plans'][k]['arena_bytes'] for k in ('separate','reuse'))
        stats=manifest['summaries'][name]
        cards.append('<section id="'+escape(name)+'"><h2>'+escape(name)+'</h2><p>'+str(a)+' → '+str(b)+
            ' bytes of tensor arena. Complete C planning: '+('%.3f'%(stats['separate']['median_ns']/1000))+' → '+
            ('%.3f'%(stats['reuse']['median_ns']/1000))+' µs/plan (median of process medians).</p>'+
            '<object type="image/svg+xml" data="'+escape(name)+'.svg" aria-label="'+escape(name)+' layout"></object></section>')
    links=' '.join('<a href="#'+escape(n)+'">'+escape(n)+'</a>' for n in checks)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tiny Arena Lab — allocation evidence</title><style>
body{margin:0;background:#eef2f7;color:#172033;font:16px/1.6 system-ui,sans-serif}main{max-width:1160px;margin:0 auto;padding:32px 20px}
h1{font-size:34px;margin:0}h2{margin:0}nav{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}a{color:#1d4ed8}
section{background:white;border:1px solid #dbe3ee;border-radius:12px;padding:20px;margin:24px 0}object{display:block;width:100%;aspect-ratio:1160/1080}
.note{border-left:4px solid #2563eb;padding:10px 18px;background:#eff6ff}code{font-size:14px}</style><main>
<h1>Tiny Arena Lab</h1><p>Actual C plans and independently checked tensor lifetimes.</p>
<div class="note">The first two panels share a byte scale; the third zooms reused storage. Hover on a block for its tensor, lifetime and offset.
Arena bytes are working tensor storage, not total process RAM. Planning cost includes validation and the C measurement loop, not model execution.</div>
<nav>LINKS</nav><p>Measured source <code>REVISION</code>. Original raw records and measurement protocol accompany these figures.</p>CARDS
<footer>Static float32 DAGs · no in-place operators · outputs pinned · greedy placement is not guaranteed optimal.</footer></main></html>'''
    (output/'index.html').write_text(html.replace('LINKS',links).replace('REVISION',manifest['source_revision'][:7]).replace('CARDS',''.join(cards)))
    print('Rendered four SVG allocation diagrams and standalone HTML viewer')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    from verify_report import verify
    verify(args.directory);render(args.directory,args.output)
