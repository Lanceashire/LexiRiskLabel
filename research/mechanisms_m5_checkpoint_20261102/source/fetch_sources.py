from pathlib import Path
import urllib.request,hashlib,json,zipfile,time
HERE=Path(__file__).resolve().parent
scope=json.loads((HERE/'pre_label_scope.json').read_text())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}))
receipts=[]
for s in scope['sources']:
    name='LC_Loan_sample_2017.csv' if s['name']=='LendingClub2017' else 'SouthGermanCredit.zip'
    p=HERE/'private/source_raw'/name
    req=urllib.request.Request(s['url'],headers={'User-Agent':'LexiRiskLabel-research/1.0'})
    with opener.open(req,timeout=45) as response:
        b=response.read(); headers={k:response.headers.get(k) for k in ['Content-Type','Last-Modified','ETag']}
    p.write_bytes(b)
    receipts.append({'dataset':s['name'],'url':s['url'],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'headers':headers,'outcome_values_inspected':False})
    (HERE/'source_download_receipts.json').write_text(json.dumps(receipts,indent=2))
    print(s['name'],len(b),receipts[-1]['sha256'],flush=True)
    if name.endswith('.zip'):
        z=zipfile.ZipFile(p); print('MEMBERS',z.namelist(),flush=True)
        for n in z.namelist():
            if n.endswith(('.txt','.R')):
                t=z.read(n).decode('utf8',errors='replace'); (HERE/Path(n).name).write_text(t); print(t,flush=True)
    else: print('HEADER ONLY',b.splitlines()[0].decode('utf8')[:5000],flush=True)
(HERE/'fetch_complete.json').write_text(json.dumps({'passed':True,'outcomes_inspected':False}))
