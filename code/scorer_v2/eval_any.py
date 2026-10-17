import csv,sys,json,collections,importlib
sys.path.insert(0,'scorer_v2'); classify_a=importlib.import_module(sys.argv[1]).classify_a
d,blind,keyf=sys.argv[2],sys.argv[3],sys.argv[4]
rows=list(csv.reader(open(f'{d}/{blind}',encoding='utf-8-sig').read().splitlines()))[1:]
lab={x[0]:x[3].strip().lower() for x in rows}; txt={x[0]:(x[1],x[2]) for x in rows}
key=json.load(open(f'{d}/{keyf}')); pol={}
for i,(q,rep) in txt.items():
    for r in map(json.loads,open(f'{d}/replies/{key[i]}.jsonl')):
        if r['reply']==rep and r['question']==q: pol[i]=r.get('polarity');break
v={i:classify_a(pol.get(i),txt[i][1]) for i in txt}; ids=sorted(lab); n=len(ids)
po=sum(v[i]==lab[i] for i in ids)/n;ca=collections.Counter(v[i] for i in ids);cb=collections.Counter(lab[i] for i in ids);pe=sum(ca[k]*cb[k] for k in ca)/n/n
out=f'5-label kappa {(po-pe)/(1-pe):.2f}'
for nm,V in (('strict',{'human'}),('broad',{'human','mixed'})):
    a=[v[i] in V for i in ids];b=[lab[i] in V for i in ids];p2=sum(x==y for x,y in zip(a,b))/n;pa,pb=sum(a)/n,sum(b)/n;pe2=pa*pb+(1-pa)*(1-pb)
    out+=f' | {nm} kappa {(p2-pe2)/(1-pe2):.2f}'
print(out);[print('   ',i,'hand=',lab[i],'scorer=',v[i],'|',txt[i][1][:90]) for i in ids if v[i]!=lab[i]]
