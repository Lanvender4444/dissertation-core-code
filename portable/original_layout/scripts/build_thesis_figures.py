"""Generate editable SVG and vector PDF figures from frozen thesis evidence.

Diagrams are explicitly schematic. Charts read audited records, not invented data.
Run in an isolated uv environment with svglib==1.6.0 and reportlab==4.4.4.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT=Path(__file__).resolve().parents[1]
NAVY='#16324f'; TEAL='#007f82'; BLUE='#3576b8'; ORANGE='#cc762c'; RED='#b74b48'
PALE='#eef3f7'; GRAY='#5e6b78'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class Figure:
    def __init__(self,title,subtitle,h=590):
        self.w=1100;self.h=h;self.parts=[];self.values={}
        self.rect(0,0,self.w,h,'#ffffff',stroke='none')
        self.text(36,43,title,30,NAVY,bold=True)
        self.text(36,76,subtitle,20,GRAY)
    def rect(self,x,y,w,h,fill=PALE,stroke='#b3c2cf',r=10):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def text(self,x,y,value,size=23,color=NAVY,anchor='start',bold=False):
        value=str(value)
        width=stringWidth(value,'Helvetica-Bold' if bold else 'Helvetica',size)
        lo=x-width/2 if anchor=='middle' else x-width if anchor=='end' else x
        if lo<-1 or lo+width>self.w+1 or y-size<-1 or y>self.h+1:
            raise ValueError(f'Label outside figure: {value}')
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Helvetica" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')
    def line(self,x1,y1,x2,y2,color=GRAY,dash=False,width=3):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
    def polygon(self,pts,fill=PALE,stroke=GRAY):
        self.parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def arrow(self,x1,y1,x2,y2,color=TEAL,dash=False):
        self.line(x1,y1,x2,y2,color,dash)
        a=math.atan2(y2-y1,x2-x1)
        self.polygon([(x2,y2),(x2-13*math.cos(a-.45),y2-13*math.sin(a-.45)),
                      (x2-13*math.cos(a+.45),y2-13*math.sin(a+.45))],color,color)
    def circle(self,x,y,r,fill=TEAL,stroke='none'):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    def box(self,x,y,w,h,lines,fill=PALE,color=NAVY):
        self.rect(x,y,w,h,fill)
        for i,value in enumerate(lines):self.text(x+w/2,y+30+i*29,value,22,color,'middle',i==0)
    def footer(self,value):self.text(36,self.h-22,value,19,GRAY)
    def save(self,out,name,kind,sources):
        svg=out/f'{name}.svg';pdf=out/f'{name}.pdf'
        svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'+''.join(self.parts)+'</svg>',encoding='utf-8')
        drawing=svg2rlg(str(svg))
        if drawing is None:raise ValueError('SVG did not render')
        renderPDF.drawToFile(drawing,str(pdf))
        return dict(name=name,kind=kind,svg=str(svg.relative_to(ROOT)),pdf=str(pdf.relative_to(ROOT)),
                    svg_sha256=sha(svg),pdf_sha256=sha(pdf),values=self.values,sources=sources,
                    native_size=[self.w,self.h])


def run(args):
    args.out=args.out.resolve()
    if args.out.exists():raise FileExistsError('Append-only figure set')
    args.out.mkdir(parents=True)
    evidence_path=ROOT/'thesis/evidence_snapshot_v6.json'
    ev=json.loads(evidence_path.read_text());rows=[]
    key=ev['current_key'];full=ev['five']['all'];three=ev['three']
    def save(f,name,kind='schematic',extra=()):
        paths=[evidence_path,*extra] if kind=='data_chart' else [ROOT/'thesis/archive/review_v4/dissertation_en.md',*extra]
        rows.append(f.save(args.out,name,kind,{str(p.relative_to(ROOT)):sha(p) for p in paths}))

    f=Figure('Camera configuration and evidence roles','Conceptual top view; positions and fields of view are not drawn to scale')
    f.rect(452,208,196,290,'#dfe7ee',NAVY,30)
    f.arrow(550,203,550,110,NAVY);f.text(570,130,'Forward',22)
    for c,x in ((2,485),(3,550),(4,615)):
        f.circle(x,221,12,TEAL);f.arrow(x,213,x,153,TEAL);f.text(x,263,f'C{c}',22,TEAL,'middle',True)
    for c,x,end in ((1,451,305),(5,649,795)):
        f.circle(x,290,13,BLUE);f.arrow(x,290,end,290,BLUE)
        f.text(370 if c==1 else 733,270,f'Cam{c}',24,BLUE,'middle',True)
    f.box(36,348,340,110,['Cam1 / Cam5','Mei fisheye observations','Lateral coverage'],fill='#edf5fb')
    f.box(724,348,340,110,['Cam2 / Cam3 / Cam4','Pinhole observations','Forward association + stereo'])
    f.text(550,403,'IMU reference',22,NAVY,'middle');f.text(550,437,'Vehicle',25,NAVY,'middle',True)
    f.footer('Lateral fisheye views are not overhead cameras; calibration is supplied, not newly estimated.')
    save(f,'camera_roles')

    f=Figure('Observation rays and uncertain depth','A native pixel constrains a direction; a detector-box centre is not a fixed surface point')
    for x,label,color in ((170,'Pinhole view',TEAL),(900,'Mei fisheye view',BLUE)):
        f.circle(x,415,18,color);f.text(x,470,label,24,color,'middle',True)
    f.polygon([(494,217),(578,181),(653,217),(615,278),(530,300)],'#e6edf5','#94a7b8')
    f.arrow(170,415,684,180,TEAL);f.arrow(900,415,455,161,BLUE)
    f.circle(575,230,12,ORANGE)
    f.text(580,151,'Feasible region (schematic)',23,NAVY,'middle')
    f.line(265,390,332,320,TEAL);f.text(66,310,'Ray from native pixel',22,TEAL)
    f.line(816,383,735,313,BLUE);f.text(762,303,'Different visible face',22,BLUE)
    f.box(368,350,355,117,['Rectified pinhole stereo','Z ~ f B / d','Depth sensitivity grows as 1 / d^2'])
    f.footer('Identity evidence and location availability are separate; unreliable location remains unknown.')
    save(f,'rays_and_uncertainty')

    f=Figure('Three detector-input policies','Historical full image, historical BAND, and current five-view crop mosaic are different inputs',670)
    for x,title in ((35,'Full-image 960'),(395,'Full image + BAND'),(755,'Five-view mosaic')):
        f.text(x+155,129,title,24,NAVY,'middle',True);f.rect(x,156,310,260,'#f3f5f7')
        if x<700:
            f.polygon([(x+60,407),(x+145,219),(x+260,407)],'#cdd7df','#b3c2cf')
            f.circle(x+120,260,9,ORANGE)
        else:
            f.rect(x+20,180,270,97,'#d7edeb',TEAL)
            f.rect(x+20,296,270,97,'#e1eaf7',BLUE)
            f.line(x+20,286,x+290,286,RED,True)
            f.text(x+155,241,'960 x 416',23,TEAL,'middle')
            f.text(x+155,357,'960 x 416',23,BLUE,'middle')
    f.rect(419,290,263,63,'none',ORANGE,0)
    f.text(190,461,'One whole-image pass',22,NAVY,'middle')
    f.text(550,461,'Extra road-band detections',22,NAVY,'middle')
    f.text(910,461,'One 960 x 832 pass',22,NAVY,'middle')
    f.text(910,501,'Native crops: 1312 x 568',20,NAVY,'middle')
    f.text(910,535,'Artificial-seam boxes discarded',19,RED,'middle')
    f.text(550,585,'Current mosaic omits pixels outside selected strips; it is not unrestricted full-image detection.',22,NAVY,'middle')
    f.footer('Fixed YOLO11n weights do not make different detector inputs a camera-count-only ablation.')
    save(f,'detector_input_policies')

    f=Figure('Historical E1: object-level duplicate handling','Decision schematic for a lower-ranked group; every shared camera must pass the overlap condition',760)
    f.box(284,113,535,70,['Rank groups by view count, then score'])
    f.arrow(550,183,550,212)
    f.box(284,213,535,75,['Same observation UID in a shared view?'])
    f.arrow(819,250,985,250,RED);f.text(902,233,'Yes',20,RED,'middle')
    f.arrow(985,250,985,278,RED);f.box(903,279,164,62,['Suppress'],fill='#faeeed',color=RED)
    f.arrow(550,288,550,330);f.text(565,319,'No',20)
    f.box(206,331,685,105,['Compatible classes and at least one shared camera?','All shared-view IoS values pass containment?'])
    f.arrow(550,436,550,482);f.text(565,466,'Yes',20)
    f.box(206,483,685,103,['Engineering depth gate also passes?','|Zg - Zh| <= max(2.5 sigma_combined, 1.5 m, 0.12 Zmin)'])
    f.arrow(550,586,550,632,RED);f.text(565,615,'Yes',20,RED)
    f.box(384,633,330,65,['Suppress lower-ranked group'],fill='#faeeed',color=RED)
    f.text(38,470,'No / failed gate:',20,TEAL);f.text(38,498,'retain candidate',20,TEAL)
    f.footer('E1 can suppress with only one shared camera. The later two-shared-view veto rule is a different branch.')
    save(f,'e1_decision')

    f=Figure('Protected recovery: preserve A, append B','Illustrative matching graph, not measured identities; individual positive labels need not be additive',670)
    f.rect(36,126,450,397,'#edf5f4',TEAL);f.text(60,164,'Protected objects A (unchanged)',25,TEAL,bold=True)
    f.rect(520,126,265,397,'#fdf4e9',ORANGE);f.text(542,164,'Residual B',25,ORANGE,bold=True)
    for x,y,label,col in ((145,250,'a1',TEAL),(145,430,'a2',TEAL),(652,285,'b1',ORANGE),(652,465,'b2',ORANGE)):
        f.circle(x,y,31,col);f.text(x,y+8,label,24,'#ffffff','middle',True)
    for y,label in ((250,'GT1'),(430,'GT2'),(575,'GT3')):
        f.circle(976,y,33,NAVY);f.text(976,y+8,label,22,'#ffffff','middle',True)
    f.arrow(179,250,938,250,TEAL);f.arrow(179,430,938,430,TEAL)
    f.arrow(683,294,945,566,ORANGE);f.arrow(683,473,945,566,ORANGE)
    f.text(280,323,'Existing matches remain feasible',20,TEAL,'middle')
    f.text(480,577,'b1 alone: +1; b2 alone: +1; together: only +1',22,ORANGE,'middle')
    f.footer('nu(A union B) >= nu(A) requires unchanged matching eligibility; it does not guarantee precision or identity.')
    save(f,'protected_recovery_graph')

    f=Figure('Chronological fitting and evaluation protocol','Original filename indices; schematic widths. All segments are from one repeatedly examined sequence',540)
    widths=[285,235,220,285];x=36
    segments=[('train','Early','0-600','Fit scores',TEAL),('development','Development','601-800','Select threshold',ORANGE),
              ('guard','Guard','801-1099','Later check',BLUE),('late','Late','1100-1700','Later check',NAVY)]
    for w,(keyseg,title,indices,role,col) in zip(widths,segments):
        f.rect(x,164,w,105,col,col,0);f.text(x+w/2,204,title,22,'#ffffff','middle',True)
        f.text(x+w/2,238,indices,20,'#ffffff','middle')
        meta=ev['segment_meta'][keyseg];metric=ev['five'][keyseg][key]
        f.text(x+w/2,316,f"{meta['frames']} frames",21,NAVY,'middle')
        f.text(x+w/2,350,f"{metric['gt']:,} GT",21,NAVY,'middle')
        f.text(x+w/2,386,role,18,GRAY,'middle');x+=w
    f.text(550,450,'Late segment retains 555 frames; 46 filename indices in the full range are absent.',22,NAVY,'middle')
    f.footer('Chronological separation does not establish new-scene independence; full-sequence results include fitting data.')
    f.values={s:ev['segment_meta'][s] for s in ('train','development','guard','late')};save(f,'evaluation_protocol','data_chart')

    f=Figure('Precision and recall: keep evaluation domains separate','Full-sequence aggregates; bars have a zero baseline and the same 0-100% scale',740)
    f.text(291,129,'Cam234 GT: 13,473',25,NAVY,'middle',True)
    f.text(804,129,'Cam12345 GT: 24,845',25,NAVY,'middle',True)
    panels=[(47,[(k,three[k]) for k in ('960/cam2','960/cam3','960/cam4','strict_no_band')],['Cam2','Cam3','Cam4','Fusion']),
            (566,[('old',ev['original_three_on_five']),('A',full['protected']),('final',full[key]),('raw',full['raw'])],['Old 3-view','A','Final 5-view','Raw pool'])]
    vals={}
    for start,items,names in panels:
        f.line(start,496,start+468,496);f.line(start,178,start,496)
        for tick in (0,25,50,75,100):
            y=496-tick*3.0;f.line(start,y,start+468,y,'#d4dde5',width=1)
            f.text(start-8,y+6,str(tick),18,GRAY,'end')
        for j,((tag,m),label) in enumerate(zip(items,names)):
            x=start+33+j*113
            for dx,field,col in ((0,'precision',NAVY),(38,'recall',TEAL)):
                val=m[field]*100;f.rect(x+dx,496-val*3,28,val*3,col,col,0)
                f.text(x+dx+14,487-val*3,f'{val:.2f}',17,col,'middle')
            f.text(x+33,529,label,19,NAVY,'middle');vals[tag]={a:m[a] for a in ('precision','recall')}
    f.rect(326,578,24,24,NAVY,NAVY,0);f.text(364,599,'Precision',22)
    f.rect(572,578,24,24,TEAL,TEAL,0);f.text(610,599,'Recall',22)
    f.text(550,652,'The 3-view to 5-view difference also changes detector input and fitted post-processing.',22,NAVY,'middle')
    f.footer('The raw-pool row is five-camera concatenation, not a single-camera baseline. Only A-to-final controls recovery.')
    f.values=vals;save(f,'precision_recall_domains','data_chart')

    f=Figure('Where same-input matching capacity is lost','Eligible TP capacity at saved stages; connecting segments do not represent tracked physical objects',610)
    vals=[full['raw']['tp'],full['raw_corrected']['tp'],14428,full[key]['tp']]
    labels=['Raw candidates','Box-corrected','A + all residuals','Selected final']
    for j,(value,label) in enumerate(zip(vals,labels)):
        x=102+j*263;y=458-value/15000*272
        f.rect(x,y,160,458-y,TEAL if j<3 else NAVY,'none',0)
        f.text(x+80,y-16,f'{value:,}',29,NAVY,'middle',True);f.text(x+80,501,label,21,NAVY,'middle')
        if j:
            diff=value-vals[j-1];f.text(x-40,154,f'{diff:+,}',23,RED if diff<0 else TEAL,'middle',True)
    f.line(80,458,1060,458)
    f.text(550,549,'Net change: +204 - 36 - 3,822 = -3,654 hits relative to raw candidates',23,NAVY,'middle')
    f.footer('High-precision admission accounts for most lost capacity; the all-residual pool is not a deployable result.')
    f.values={'stage_tp':vals,'net':vals[-1]-vals[0]};save(f,'matching_capacity','data_chart')

    f=Figure('Controlled recovery at the retained threshold 0.55','Same detector pool, boxes, protected A and five-view GT; full sequence includes fitting frames',590)
    baseline=full['protected'];vals={}
    for j,(tag,label) in enumerate(((key,'Singleton'),('pair/independent/tau0.55','Pair'),('shared/independent/tau0.55','Shared-state'))):
        x=153+j*342;m=full[tag];tp=m['tp']-baseline['tp'];fp=m['fp']-baseline['fp'];vals[tag]={'added_tp':tp,'added_fp':fp}
        for dx,v,c in ((0,tp,TEAL),(91,fp,ORANGE)):
            f.rect(x+dx,411-v*.85,66,v*.85,c,c,0);f.text(x+dx+33,396-v*.85,str(v),30,c,'middle',True)
        f.text(x+77,453,label,25,NAVY,'middle',True)
    f.line(90,411,1040,411)
    f.rect(286,484,24,24,TEAL,TEAL,0);f.text(324,505,'Added TP',22)
    f.rect(578,484,24,24,ORANGE,ORANGE,0);f.text(616,505,'Added FP',22)
    f.footer('Singleton: +0.97 recall points, -0.40 precision points. Pair/shared branches did not improve the locked point.')
    f.values=vals;save(f,'controlled_recovery','data_chart')

    f=Figure('Approximate position availability is not BEV accuracy','Audited full viewer replay; denominator is displayed object-frame outputs, not eligible GT')
    av=ev['position_availability'];n=av['displayed'];located=av['located'];unknown=n-located
    # Polygon sectors avoid SVG path conversion differences and remain editable vector geometry.
    for start,end,color in ((0,located/n*360,TEAL),(located/n*360,360,'#d9e2ea')):
        pts=[(280,320)]+[(280+170*math.cos(math.radians(a-90)),320+170*math.sin(math.radians(a-90))) for a in [start+i*(end-start)/160 for i in range(161)]]
        f.polygon(pts,color,color)
    f.circle(280,320,112,'#ffffff');f.text(280,315,f'{located/n*100:.2f}%',40,TEAL,'middle',True)
    f.text(280,351,'located',24,NAVY,'middle')
    for y,value,desc,color in ((228,located,'Approximate position available',TEAL),(332,unknown,'No accepted metric position',GRAY),(438,n,'Total displayed outputs',NAVY)):
        f.text(557,y,f'{value:,}',36,color,bold=True);f.text(557,y+37,desc,23,color)
    f.footer('101 small-GT-only outputs are displayed but ignored in TP/FP; availability is not localisation accuracy.')
    f.values={'displayed':n,'located':located,'not_located':unknown,'availability':located/n};save(f,'bev_availability','data_chart')

    f=Figure('Why a metric overlay can drift on a bowl texture','Display-coordinate schematic: assumed surface and physical object need not coincide',650)
    f.polygon([(80,418),(215,354),(346,311),(501,287),(663,300),(819,346),(985,419)],'#e7eef4','#90a4b6')
    f.circle(184,194,18,NAVY);f.text(110,145,'Virtual camera',24,NAVY)
    f.circle(629,283,12,TEAL);f.text(650,272,'Q: display surface',22,TEAL)
    f.circle(752,187,12,ORANGE);f.text(770,168,'P: metric estimate',22,ORANGE)
    f.arrow(202,195,626,283,TEAL);f.arrow(202,192,748,187,ORANGE)
    f.box(65,461,455,100,['Image overlay','Use source-camera owner + pixel LUT','Anchor to the same texture mapping'],fill='#edf5f4',color=TEAL)
    f.box(568,461,455,100,['Metric BEV layer','Keep original P and uncertainty','Do not snap P onto bowl surface'],fill='#fdf4e9',color=ORANGE)
    f.footer('Source-map alignment fixes rendering consistency, not detector accuracy or physical 3D location.')
    save(f,'bowl_overlay_coordinates')

    record=dict(complete=True,figures=rows,svg_count=len(rows),vector_pdf_count=len(rows),
                generator_sha256=sha(Path(__file__)),data_source_sha256=sha(evidence_path),
                scope='Original code-native illustrations and plots of frozen audited results; no new algorithm experiment.')
    with (args.out/'manifest.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    print(json.dumps(dict(complete=True,svg_count=len(rows),output=str(args.out))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args())
