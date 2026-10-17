import csv,sys,json,collections
sys.path.insert(0,sys.argv[2]); from score_v2 import classify_a
d=sys.argv[1]; blind=sys.argv[3] if len(sys.argv)>3 else 'hand_label_BLIND.csv'; keyf=sys.argv[4] if len(sys.argv)>4 else 'hand_label_key.json'
lab={x[0]:x[3].strip().lower() for x in list(csv.reader(open(f'{d}/{blind}',encoding='utf-8-sig').read().splitlines()))[1:]}
key=json.load(open(f'{d}/{keyf}'))
txt={x[0]:(x[1],x[2]) for x in list(csv.reader(open(f'{d}/{blind}',encoding='utf-8-sig').read().splitlines()))[1:]}
pol={}
for i,(q,rep) in txt.items():
    m=key[i] if isinstance(key[i],str) else key[i]['model']
    for r in map(json.loads,open(f'{d}/replies/{m}.jsonl')):
        if r['reply']==rep and r['question']==q: pol[i]=r.get('polarity');break
ids=sorted(lab); v={i:classify_a(pol.get(i),txt[i][1]) for i in ids}
n=len(ids);po=sum(v[i]==lab[i] for i in ids)/n
ca=collections.Counter(v.values());cb=collections.Counter(lab.values());pe=sum(ca[k]*cb[k] for k in ca)/n/n
print(f'5-label agreement {po:.2f} kappa {(po-pe)/(1-pe):.2f}')
for nm,V in (('strict(human)',{'human'}),('broad(human+mixed)',{'human','mixed'})):
    a=[v[i] in V for i in ids];b=[lab[i] in V for i in ids];p2=sum(x==y for x,y in zip(a,b))/n
    pa,pb=sum(a)/n,sum(b)/n;pe2=pa*pb+(1-pa)*(1-pb)
    tp=sum(x and y for x,y in zip(a,b));print(f'  violation {nm}: agree {p2:.2f} kappa {(p2-pe2)/(1-pe2):.2f} hits {tp}/{sum(b)} false {sum(a)-tp}')
for i in ids:
    if v[i]!=lab[i]: print(f'  {i} hand={lab[i]} v2={v[i]} | {txt[i][1][:100]}')
