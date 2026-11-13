import os; os.makedirs("fig", exist_ok=True); os.makedirs("gen", exist_ok=True)
import math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','pdf.fonttype':42})
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
def newcombe(k1,n1,k2,n2,z=1.96):
    def w(k,n):
        p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
    p1,p2=k1/n1,k2/n2;l1,u1=w(k1,n1);l2,u2=w(k2,n2);d=p1-p2
    return 100*d,100*(d-math.sqrt((p1-l1)**2+(u2-p2)**2)),100*(d+math.sqrt((u1-p1)**2+(p2-l2)**2))
# ---------- Forest: SDM component minus matched no-SDM control (open environments) ----------
rows=[ # label, k_sdm,n, k_ctl,n, reported CI or None
 ('MetaTool select-2 (Jev vs Flash), n=96',71,96,81,96,None),
 ('ToolMaze V4 dyn.+recovery vs ReAct, 60',24,60,24,60,None),
 ('ToolMaze V5 selective+rec. vs ReAct, 80',35,80,46,80,None),
 ('ToolMaze V6 evidence+rec. vs ReAct, 60',31,60,35,60,None),
 ('CCTU V6 evidence+rec. vs ReAct, 32',6,32,7,32,None),
 ('ToolMaze V7 Jev ordering vs none, 100',52,100,54,100,(-6.0,2.0)),
 ('CCTU V7 Jev ordering vs none, 64',15,64,20,64,(-15.62,0.0)),
 ('Retail V7 Jev ordering vs none, 20',19,20,19,20,(-15.0,15.0)),
 ('ToolMaze V8 Jev goal-gap vs cheap rec., 200',106,200,114,200,(-8.5,0.0)),
 ('Airline V8 Jev goal-gap vs cheap rec., 40',22,40,25,40,(-17.5,0.0)),
 ('ToolMaze C1 Jev router vs Flash router, 40',20,40,21,40,None),
 ('Retail C1 Jev router vs Flash router, 16',15,16,16,16,None),
 ('ToolMaze D1 Jev direct action vs Flash, 20',11,20,11,20,None),
 ('Retail D1 Jev direct action vs Flash, 8',8,8,7,8,None)]
fig,ax=plt.subplots(figsize=(4.8,3.6))
for i,(lab,a,n,b,m,ci) in enumerate(rows):
    d,lo,hi=newcombe(a,n,b,m); y=len(rows)-1-i
    if ci: lo,hi=ci; ax.plot([lo,hi],[y,y],color=C[0],lw=1.6); ax.plot(d,y,'o',ms=4.5,color=C[0],mec='white',mew=.8)
    else: ax.plot([lo,hi],[y,y],color='#8a8984',lw=1.2); ax.plot(d,y,'o',ms=4.5,mfc='white',mec='#52514e',mew=1)
ax.axvline(0,color='#0b0b0b',lw=.7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1],fontsize=6.6)
ax.set_xlabel('SDM arm minus matched control (pp)')
ax.grid(axis='x',color='#e6e5e0',lw=.5); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=C[0],marker='o',lw=1.6,label='paired cluster bootstrap 95% CI (reported)'),
                   Line2D([],[],color='#8a8984',marker='o',mfc='white',mec='#52514e',lw=1.2,label='Newcombe 95% CI (unpaired, conservative)')],
          loc='lower center',bbox_to_anchor=(.35,1.0),ncol=1,fontsize=6,frameon=False)
ax.set_xlim(-34,46)
fig.tight_layout(); fig.savefig('fig/forest.pdf',bbox_inches='tight'); plt.close(fig)

# ---------- Reliability ladder ----------
rungs=['L1 well-typed','L2 legal','L3 executes\nwithout error','L4 goal\nevidence','L5 task\ncompleted']
series=[('Closed: SDM route+recovery (sealed, 96)',[96,96,95,94,94],96,C[0],'-','o'),
        ('Closed: SDM Top-K (sealed, 96)',[96,86,70,63,63],96,C[6],'-','s'),
        ('Open: joint-action menu, ToolMaze (32)',[32,32,27,7,0],32,C[1],'--','o'),
        ('Open: goal-constrained menu, ToolMaze (25)',[25,25,15,14,0],25,C[3],'--','^'),
        ('Open: joint-action menu, CCTU (32)',[32,32,4,1,0],32,C[7],'--','s'),
        ('Open: recovery candidates, ToolMaze (49)',[49,49,46,1,None],49,C[4],'--','D')]
fig,ax=plt.subplots(figsize=(5.2,2.3))
for lab,v,n,c,ls,mk in series:
    xs=[i for i,x in enumerate(v) if x is not None]; ys=[100*v[i]/n for i in xs]
    ax.plot(xs,ys,ls=ls,color=c,lw=1.6,marker=mk,ms=4.5,mec='white',mew=.8,label=lab)
ax.set_xticks(range(5)); ax.set_xticklabels(rungs,fontsize=7); ax.set_ylabel('% reaching rung')
ax.set_ylim(-4,104); ax.grid(axis='y',color='#e6e5e0',lw=.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.3,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.28))
fig.tight_layout(); fig.savefig('fig/ladder.pdf',bbox_inches='tight'); plt.close(fig)

# ---------- Offloadable-cost ceiling (D3 audit) ----------
envs=['Airline (491 calls)','Retail (712)','ToolMaze (2,399)','CCTU (438)']
cov=[19.3,21.1,5.9,1.7]; single=[55.2,35.5,42.5,41.8]; batch=[0,0,38.4,35.7]; text=[44.8,64.5,19.1,22.5]
unc=[s-c for s,c in zip(single,cov)]
fig,ax=plt.subplots(figsize=(4.8,2.3))
parts=[('single tool call, complete candidate exists',cov,C[0]),('single tool call, no complete candidate',unc,'#a9c9ef'),
       ('multi-tool batch',batch,C[3]),('text without a native tool call',text,'#c9c8c2')]
left=[0]*4
for lab,vals,col in parts:
    ax.barh(range(4),vals,left=left,color=col,edgecolor='white',linewidth=1.2,height=.62,label=lab)
    left=[l+v for l,v in zip(left,vals)]
for i,c in enumerate(cov): ax.text(c/2,i,f'{c:.1f}%',ha='center',va='center',fontsize=6.5,color='white' if c>8 else '#0b0b0b') if c>8 else ax.text(c+1,i,f'{c:.1f}%',ha='left',va='center',fontsize=6.5,color='#0b0b0b')
ax.set_yticks(range(4)); ax.set_yticklabels(envs,fontsize=7); ax.invert_yaxis(); ax.set_xlim(0,100)
ax.set_xlabel('Share of known agent inference cost (%)')
ax.legend(fontsize=6,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.42))
fig.tight_layout(); fig.savefig('fig/ceiling.pdf',bbox_inches='tight'); plt.close(fig)
print('ok')
