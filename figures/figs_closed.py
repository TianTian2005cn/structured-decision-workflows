import os; os.makedirs("fig", exist_ok=True); os.makedirs("gen", exist_ok=True)
import math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','pdf.fonttype':42})
C=['#2a78d6','#eb6834','#1baf7a','#eda100']
groups=['Basic\n(dev, 100)','Medium\n(dev, 100)','Hard\n(dev, 100)','Transfer\n(dev, 100)','Ledger held-out\n(test, 96)','Ledger held-out,\nNL request (96)','Orders held-out\n(test, 96)']
n=[100,100,100,100,96,96,96]
arms={'Top-K':[50,44,43,21,63,54,37],'Top-K + recovery':[52,44,45,24,78,63,41],
      'Constraint routing':[86,80,61,63,76,77,55],'Routing + recovery':[96,88,69,83,94,96,68]}
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
fig,ax=plt.subplots(figsize=(6.6,2.3));w=.19
for j,(a,ks) in enumerate(arms.items()):
    xs=[i+(j-1.5)*w for i in range(len(groups))];ys=[100*k/m for k,m in zip(ks,n)]
    lo=[y-100*wil(k,m)[0] for y,k,m in zip(ys,ks,n)];hi=[100*wil(k,m)[1]-y for y,k,m in zip(ys,ks,n)]
    ax.bar(xs,ys,width=w*.9,color=C[j],label=a,edgecolor='white',linewidth=.6)
    ax.errorbar(xs,ys,yerr=[lo,hi],fmt='none',ecolor='#52514e',elinewidth=.6,capsize=1.2)
ax.axvline(3.5,color='#c9c8c2',lw=.8,ls=':')
ax.set_xticks(range(len(groups)));ax.set_xticklabels(groups,fontsize=6.5);ax.set_ylim(0,105);ax.set_ylabel('Workflow success (%)')
ax.grid(axis='y',color='#e6e5e0',lw=.5);ax.set_axisbelow(True)
ax.legend(ncol=4,fontsize=6.5,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.0))
fig.tight_layout();fig.savefig('fig/closed.pdf',bbox_inches='tight')
