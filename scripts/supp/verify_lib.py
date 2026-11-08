from supp_lib import *
seg=segments();tot={k:np.zeros(3,int) for k in ('A','F5P','repair','F5R','raw','rawn')};bad=0
for s in seg['full']:
    F=frame(s);C=configs(s);g=F['gt']
    cp=evaluate(as_objects(C['F5P'],F),g);cr=evaluate(as_objects(C['F5R'],F),g)
    if cp.tolist()!=F['f5p_counts'] or cr.tolist()!=F['f5r_counts'] or evaluate(as_objects(C['repair'],F),g).tolist()!=F['counts']['repair_only']: bad+=1
    tot['F5P']+=cp;tot['F5R']+=cr;tot['A']+=evaluate(as_objects(C['A'],F),g);tot['repair']+=evaluate(as_objects(C['repair'],F),g)
    tot['raw']+=evaluate(as_objects(C['raw'],F),g);tot['rawn']+=evaluate(as_objects(C['raw'],F,corrected=False),g)
print('mismatch frames',bad)
for k,v in tot.items():print(k,v.tolist(),{a:round(b,4) for a,b in prf(v).items() if a in('precision','recall','f1')})
