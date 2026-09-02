import json, time, urllib.request
rows=[json.loads(l) for l in open('data/scorer_tada_test.jsonl')]
out=[]; t0=time.time()
for i,r in enumerate(rows):
    req=urllib.request.Request('http://127.0.0.1:8095/score', data=json.dumps({'text':r['text']}).encode(), headers={'Content-Type':'application/json'})
    try:
        d=json.loads(urllib.request.urlopen(req, timeout=120).read())
        out.append({'id':r['id'],'y':r['y'],'pred':d['rating']})
    except Exception as e:
        out.append({'id':r['id'],'y':r['y'],'pred':None,'err':str(e)[:60]})
    if (i+1)%100==0: print(f"{i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
json.dump(out, open('out/tada_zeroshot.json','w'))
ok=[o for o in out if o['pred'] is not None]
import statistics
def rank(v):
    s=sorted(range(len(v)), key=lambda i:v[i]); r=[0]*len(v)
    for j,i in enumerate(s): r[i]=j
    return r
ys=rank([o['y'] for o in ok]); ps=rank([o['pred'] for o in ok]); n=len(ok)
my, mp = sum(ys)/n, sum(ps)/n
cov=sum((a-my)*(b-mp) for a,b in zip(ys,ps)); sy=sum((a-my)**2 for a in ys)**.5; sp=sum((b-mp)**2 for b in ps)**.5
print(f"ZERO-SHOT tada test: Spearman={cov/sy/sp:+.4f} (n={n})")
