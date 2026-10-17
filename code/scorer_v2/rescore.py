import json,sys,itertools,os
sys.path.insert(0,'scorer_v2'); from score_v21 import classify_a
d=sys.argv[1]; out={}
def U(x,y): return sum((a>b)+0.5*(a==b) for a in x for b in y)
def mw(x,y):
    allv=x+y;n=len(x);u0=U(x,y);c=t=0
    for idx in itertools.combinations(range(len(allv)),n):
        s=set(idx);t+=1;c+=U([allv[i] for i in idx],[allv[i] for i in range(len(allv)) if i not in s])>=u0-1e-9
    return c/t
def rate(m,reading,qset):
    R=[r for r in map(json.loads,open(f'{d}/replies/{m}.jsonl')) if r.get('category')=='trigger' and (qset=='all' or r.get('polarity')=='either_or')]
    L=[classify_a(r.get('polarity'),r['reply']) for r in R]
    V={'strict':{'human'},'broad':{'human','mixed'}}[reading]
    return sum(l in V for l in L)/len(L), len(L)
G=lambda p,n:[f'{p}{i}' for i in range(1,n+1)]
for qset in ('either_or','all'):
  for reading in ('broad','strict'):
    r=lambda m:rate(m,reading,qset)[0]
    par={m:round(100*r(m)) for m in ('base','O1','C1','BEN','O1_lora','C1_lora','O1_merge','C1_merge','O1_q4','C1_q4')}
    K,P,D,RK,RP,GK,GP=[[r(m) for m in G(p,n)] for p,n in (('K',8),('P',8),('D',4),('RK',4),('RP',4),('GK',4),('GP',4))]
    mean=lambda x:sum(x)/len(x)
    beta=(mean(K)-mean(P))/((r('O1')-r('C1')) or 1)
    print(f'## {qset} | {reading}  (n per model {rate("K1",reading,qset)[1]})')
    print('  parents/routes:',par)
    for nm,a,b in (('K>P',K,P),('K>D',K,D),('RK>RP',RK,RP),('GK>GP',GK,GP),('RK>K (style carried part?)',RK,K)):
        print(f'  {nm}: {[round(100*v) for v in a]} vs {[round(100*v) for v in b]}  p={mw(a,b):.4g}')
    print(f'  beta(distill)={beta:.2f}  K mean {100*mean(K):.0f}%  P mean {100*mean(P):.0f}%  D mean {100*mean(D):.0f}%')
    gk_gap=mean(GK)-mean(GP); k_gap=mean(K[:4])-mean(P[:4])
    print(f'  grandchildren gap {100*gk_gap:.0f} pts vs their parents gap {100*k_gap:.0f} pts -> retained {gk_gap/k_gap if k_gap else float("nan"):.2f}')
