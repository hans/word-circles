"""Digitize the gold-syllable (gray) curve of Fig 6 in docs/ref.pdf -> docs/research/fig6-gold-targets.csv values.

Reads the figure's embedded raster (page 6, xref 393, 1144x617 px: its native resolution, so a
300-dpi page render adds nothing). Axes are calibrated from tick pixels; each point is the
mean linear extrapolation of the gray segments on either side of the marker column, which also
recovers markers occluded by other curves.

  uv run --with pymupdf --with pillow python -I docs/research/fig6_digitize.py docs/ref.pdf
"""
import io, sys
import numpy as np
import pymupdf
from PIL import Image
doc = pymupdf.open(sys.argv[1])
a = np.array(Image.open(io.BytesIO(pymupdf.Pixmap(doc, 393).tobytes("png"))).convert("RGB")).astype(int)
g=a.max(2); neutral=(a.max(2)-a.min(2))<=6
gray=neutral&(g>=110)&(g<=185)
panels={'F1':(101,488,15,246,(246,0),(19,100)),'TokenF1':(101,488,309,540,(540,0),(313,100)),
 'OS':(656,1031,19,250,(220,0),(19,300)),'R':(644,1031,307,538,(414,0),(314,100))}
cats={'F1':[135,188,241,295,348,401,455],'TokenF1':[135,188,241,295,348,401,455],
 'OS':[690,743,796,850,903,957,1010],'R':[677,731,784,837,891,944,997]}
def col_y(x,yt,yb,prev=None):
    ys=np.where(gray[yt+1:yb-1,x])[0]+yt+1
    if len(ys)==0: return None
    grps=np.split(ys,np.where(np.diff(ys)>2)[0]+1)
    grps=[gg for gg in grps if len(gg)<=12]  # line thickness, not marker/bars
    if not grps: return None
    if prev is not None: grp=min(grps,key=lambda gg:abs(gg.mean()-prev))
    else: grp=max(grps,key=len)
    return grp.mean()
for name,(xl,xr,yt,yb,(p0,v0),(p1,v1)) in panels.items():
    cs=cats[name]; conv=lambda y: v0+(y-p0)*(v1-v0)/(p1-p0)
    row=[]
    for i,cx in enumerate(cs):
        ests=[]
        for side in (-1,1):
            j=i+side
            if j<0 or j>=len(cs): continue
            nx=cs[j]; lo,hi=sorted((cx+side*9, nx-side*9))
            pts=[(x,col_y(x,yt,yb)) for x in range(lo,hi+1)]
            pts=[(x,y) for x,y in pts if y is not None]
            if len(pts)<8: continue
            X,Y=np.array(pts).T
            # robust fit: iterate dropping outliers
            for _ in range(3):
                k,b=np.polyfit(X,Y,1); r=np.abs(Y-(k*X+b)); m=r<max(2,2*np.median(r)+1)
                X,Y=X[m],Y[m]
            ests.append(conv(k*cx+b))
        row.append([round(e,1) for e in ests])
    print(name); [print('  ',r) for r in row]
