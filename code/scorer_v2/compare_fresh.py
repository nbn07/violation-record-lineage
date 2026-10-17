import csv,sys,json,collections
sys.path.insert(0,'scorer_v2'); from score_v2 import classify_a
d=sys.argv[1]
rows=list(csv.reader(open(f'{d}/hand_label_FRESH.csv',encoding='utf-8-sig').read().splitlines()))[1:]
you={x[0]:x[3].strip().lower() for x in rows}; txt={x[0]:(x[1],x[2]) for x in rows}
cl={x['id']:x['claude_label'] for x in csv.DictReader(open(f'{d}/claude_labels_fresh.csv'))}
key=json.load(open(f'{d}/_fresh_key_DO_NOT_OPEN/fresh_key.json'))
pol={}
for i,(q,rep) in txt.items():
    for r in map(json.loads,open(f'{d}/replies/{key[i]}.jsonl')):
        if r['reply']==rep and r['question']==q: pol[i]=r.get('polarity');break
v2={i:classify_a(pol.get(i),txt[i][1]) for i in txt}
ids=sorted(you); n=len(ids)
print('your labels:',dict(collections.Counter(you.values())), '| blanks:',sum(not you[i] for i in ids))
def kap(a,b):
    po=sum(a[i]==b[i] for i in ids)/n;ca=collections.Counter(a[i] for i in ids);cb=collections.Counter(b[i] for i in ids)
    pe=sum(ca[k]*cb[k] for k in ca)/n/n;return po,(po-pe)/(1-pe)
def bin_(a,b,V):
    x=[a[i] in V for i in ids];y=[b[i] in V for i in ids];po=sum(p==q for p,q in zip(x,y))/n
    pa,pb=sum(x)/n,sum(y)/n;pe=pa*pb+(1-pa)*(1-pb);tp=sum(p and q for p,q in zip(x,y))
    return f'agree {po:.2f} kappa {(po-pe)/(1-pe):.2f} (both {tp}, only-first {sum(x)-tp}, only-second {sum(y)-tp})'
for name,o in (('SCORER v2 (TEST)',v2),('Claude labels',cl)):
    po,k=kap(you,o);print(f'== you vs {name}: 5-label agree {po:.2f} kappa {k:.2f}')
    print('   strict (human):',bin_(you,o,{'human'}));print('   broad (human+mixed):',bin_(you,o,{'human','mixed'}))
    for i in ids:
        if you[i]!=o[i]: print(f'     {i} you={you[i]} {name.split()[0]}={o[i]} | {txt[i][0][:32]} | {txt[i][1][:90]}')
