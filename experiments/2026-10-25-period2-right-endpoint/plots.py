"""Publication-scale raster figures; x/y and numerical roots remain distinct."""
from research import *
from algebra_check import p2cycles

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.16,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#146a9a';RED='#b84a30';PURPLE='#8252a1';GRAY='#9da6ad'

def csvrows(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def main():
    q=np.load(OUT/'branch.npy');infoq=[info(x) for x in q]
    stable=np.array([r['rho']<1 for r in infoq]);fold=json.loads((OUT/'summary.json').read_text())['fold']
    kf=fold['k'];qf=np.array([fold['ax'],fold['ay'],kf])
    scan=np.load(OUT/'scan.npz');ks=scan['ks'];tail=scan['tail'];period=scan['period'];ly=scan['lyapunov']
    other=csvrows('all_period2.csv');fixed=csvrows('all_fixedpoints.csv')
    fig,axs=plt.subplots(2,1,figsize=(12,9),sharex=True,constrained_layout=True)
    for coord,ax in enumerate(axs):
        # Original-seed scatter and a thinner sample of the other fifteen seeds.
        ax.plot(np.repeat(ks,128),tail[:,:,0,coord].T.ravel(),',',c='#343c44',alpha=.25,rasterized=True,label='Original seed: last 128 states')
        ax.plot(np.repeat(ks,16*15),tail[-16:,:,1:,coord].transpose(1,0,2).ravel(),',',c='#a4a9af',alpha=.11,rasterized=True,label='Other 15 seeds: last 16 states')
        for group in [0,1]:
            rows=[r for r in other if r['physical']=='True' and ((float(r['ay'])>.7) if group else (float(r['ax'])<.86))]
            for field in [('ax','bx'),('ay','by')][coord]:
                ax.plot([float(r['k']) for r in rows],[float(r[field]) for r in rows],color=GRAY,ls=(0,(3,3)),lw=1.1,label='Other unstable period-2' if group==0 and field in ['ax','ay'] else None)
        for j,field in enumerate([('ax','bx'),('ay','by')][coord]):
            vals=np.array([r[field] for r in infoq])
            ax.plot(q[:,2],np.where(stable,vals,np.nan),c=BLUE,lw=2.1,label='Tracked period-2: attracting' if j==0 else None)
            ax.plot(q[:,2],np.where(~stable,vals,np.nan),c=RED,lw=1.8,ls='--',label='Tracked period-2: unstable' if j==0 else None)
            ax.plot(kf,fold[field],'o',ms=4,c='black')
        rows=[r for r in fixed if r['physical']=='True' and float(r['x'])>.5]
        ax.plot([float(r['k']) for r in rows],[float(r[['x','y'][coord]]) for r in rows],c=PURPLE,ls='-.',lw=1.8,label='Period-1 fixed point: repelling')
        ax.axvline(kf,c='black',ls=':',lw=1)
        ax.set(ylabel=['x coordinate','y coordinate'][coord],ylim=(.28,.98),xlim=(.143,.147))
    axs[0].set_title('Right endpoint: a finite-amplitude period-2 fold, not a period-1 merger',loc='left',fontsize=14,pad=12)
    axs[0].legend(ncol=2,loc='lower left',fontsize=8,framealpha=.95)
    axs[1].set_xlabel('Coupling k');axs[1].text(.14538,.31,f'Fold k = {kf:.12f}',fontsize=9)
    fig.savefig(OUT/'bifurcation_right_end.png',dpi=260);plt.close(fig)
    # Dense arclength sampling through the fold: no parameter sorting.
    q0=np.array([.8863394107528403,.5609843322345676,.1452]);dense,_=trace(q0,1,ds=.00001,n=5000)
    np.save(OUT/'dense_fold.npy',dense)
    rr=[info(x) for x in dense];st=np.array([r['rho']<1 for r in rr]);x=1e6*(dense[:,2]-kf)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    series=[('ay','Phase A: y coordinate'),('gap','Two-phase distance ||A - B||'),('mu2_real','Critical Floquet multiplier')]
    for ax,(field,label) in zip(axs.flat,series):
        vals=np.array([r[field] for r in rr])
        ax.plot(x,np.where(st,vals,np.nan),c=BLUE,lw=2,label='Attracting period-2')
        ax.plot(x,np.where(~st,vals,np.nan),c=RED,lw=1.8,ls='--',label='Saddle period-2')
        ax.plot(0,fold[field],'ko',ms=5);ax.axvline(0,c='black',ls=':',lw=.8)
        ax.set(xlim=(-35,3),xlabel=r'$10^6 (k-k_{SN})$',ylabel=label)
    axs[0,0].set_ylim(.563,.570);axs[0,0].legend(fontsize=8)
    axs[0,1].set_ylim(.5035,.508)
    axs[1,0].set_ylim(.955,1.045);axs[1,0].axhline(1,c='black',ls=':',lw=.8)
    ax=axs[1,1];use=(x>-35)&(x<3)
    for mask,color,ls in [(use&st,BLUE,'-'),(use&~st,RED,'--')]:
        ax.plot(np.where(mask,dense[:,0],np.nan),np.where(mask,dense[:,1],np.nan),c=color,ls=ls,lw=2)
    ax.plot(fold['ax'],fold['ay'],'ko',ms=5)
    ax.set(xlabel='Phase A: x coordinate',ylabel='Phase A: y coordinate',title='The two branches join smoothly in state-parameter space')
    fig.suptitle('Only the two period-2 branches meet; the two temporal phases stay apart',fontsize=14)
    fig.savefig(OUT/'fold_detail.png',dpi=260);plt.close(fig)
    # Show the entire tracked component down to zero coupling.
    fig,axs=plt.subplots(2,1,figsize=(11,8),sharex=True,constrained_layout=True)
    for ax,fields in zip(axs,[('ax','bx'),('ay','by')]):
        for j,f in enumerate(fields):
            vals=np.array([r[f] for r in infoq])
            ax.plot(q[:,2],np.where(stable,vals,np.nan),c=BLUE,lw=2,label='Attracting period-2' if j==0 else None)
            ax.plot(q[:,2],np.where(~stable,vals,np.nan),c=RED,ls='--',lw=1.4,label='Unstable period-2' if j==0 else None)
        ax.set(ylabel=fields[0][1]+' coordinate',xlim=(0,.15));ax.axvline(kf,c='k',ls=':',lw=.8)
    pd=json.loads((OUT/'validation.json').read_text())['left_flip']['k']
    for ax in axs:ax.axvline(pd,c=GRAY,ls=':',lw=1)
    axs[0].legend(loc='lower right');axs[0].set_title('The tracked period-2 component: both arms return to k = 0',loc='left',fontsize=13)
    axs[1].set_xlabel('Coupling k');fig.savefig(OUT/'full_component.png',dpi=240);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axs[0,0];ax.fill_between(ks,ly.min(1),ly.max(1),color=BLUE,alpha=.2,label='Range over 16 seeds')
    ax.plot(ks,ly[:,0],c=BLUE,lw=.9,label='Original seed');ax.axhline(0,c='k',lw=.7)
    ax.axvline(kf,c='k',ls=':',lw=1);ax.set(xlabel='k',ylabel='Largest Lyapunov estimate / iteration',title='Finite-time estimates near the fold');ax.legend(fontsize=8)
    ax=axs[1,0];ax.plot(ks,(period==2).mean(1),c=BLUE,lw=2);ax.axvline(kf,c='k',ls=':',lw=1)
    ax.set(xlabel='k',ylabel='Fraction of 16 seeds classified as period 2',ylim=(-.05,1.05),title='Classification uses joint 2D recurrence')
    basin=np.load(OUT/'basin_checks.npz')
    for ax,target in [(axs[0,1],.1453),(axs[1,1],.145324)]:
        i=np.argmin(abs(basin['ks']-target));vals=(basin['period'][i]==2).reshape(51,51)
        ax.imshow(vals,origin='lower',extent=(.005,.995,.005,.995),vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
        ax.set(xlabel='Coupling-onset x',ylabel='Coupling-onset y',title=f'k = {target}: {int(vals.sum())}/2601 period-2 seeds')
    fig.savefig(OUT/'initial_condition_checks.png',dpi=240);plt.close(fig)
    # Sweeps and saddle-node ghost: different sampled points are never connected.
    sw=np.load(OUT/'sweeps.npz');fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key in zip(axs[0],['up','down']):
        ax.plot(np.repeat(sw['ks'],128),sw[key][:,:,1].ravel(),',',c=BLUE,alpha=.5)
        ax.axvline(kf,c='k',ls=':',lw=1);ax.set(xlabel='k',ylabel='Late y',title=f'{key.capitalize()} sweep: 6000 transient steps / k')
    rows=csvrows('ghost_exit_times.csv');delta=np.array([float(r['delta_k']) for r in rows]);steps=np.array([int(r['first_exit_steps']) for r in rows])
    ax=axs[1,0];ax.loglog(delta,steps,'o-',c=BLUE,label='Measured first exit')
    ax.loglog(delta,1.034/np.sqrt(delta),'--',c=RED,label=r'$1.034 / \sqrt{k-k_{SN}}$')
    ax.set(xlabel=r'$k-k_{SN}$',ylabel='Map iterations',title='Saddle-node ghost: a long near-periodic transient');ax.legend(fontsize=8)
    ax=axs[1,1];rows=csvrows('long_summary.csv');xx=np.array([float(r['k']) for r in rows]);lo=np.array([float(r['lyapunov_min']) for r in rows]);hi=np.array([float(r['lyapunov_max']) for r in rows]);ax.errorbar(1e6*(xx-kf),(hi+lo)/2,yerr=(hi-lo)/2,fmt='o',c=BLUE,capsize=3)
    ax.axhline(0,c='k',lw=.7);ax.set_xscale('symlog',linthresh=.1)
    ax.set(xlabel=r'$10^6(k-k_{SN})$; symmetric-log scale',ylabel='Largest Lyapunov estimate',title='200k transient + 131072 measured steps; 16 seeds')
    fig.savefig(OUT/'sweeps_and_transients.png',dpi=240);plt.close(fig)
    # Coexisting attractors found inside a narrow window to the left of the fold.
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cb=np.load(OUT/'coexistence_basin.npz');cl=np.load(OUT/'coexistence_long.npz')
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    colors=[BLUE,'#d99b28','#a4488e'];labels=[2,6,18]
    p=cb['period'][0];categories=np.zeros_like(p)
    for i,periodval in enumerate(labels):categories[p==periodval]=i
    cmap=ListedColormap(colors)
    im=axs[0].imshow(categories.reshape(51,51),origin='lower',extent=(.005,.995,.005,.995),cmap=cmap,vmin=-.5,vmax=2.5,interpolation='nearest')
    cbar=fig.colorbar(im,ax=axs[0],ticks=[0,1,2],shrink=.8);cbar.ax.set_yticklabels(['Period 2','Period 6','Period 18'])
    axs[0].set(xlabel='Coupling-onset x',ylabel='Coupling-onset y',title='Three sampled basins at k = 0.143305')
    ki=np.argmin(abs(cl['ks']-.143305))
    for i,periodval in enumerate(labels):
        si=np.where(cl['period'][ki]==periodval)[0][0];orb=cl['tail'][-periodval:,ki,si,:]
        axs[1].scatter(orb[:,0],orb[:,1],s=[65,40,22][i],c=colors[i],marker=['o','s','^'][i],label=f'Period {periodval}',zorder=3)
    inset=axs[1].inset_axes([.22,.42,.40,.30])
    for i,periodval in [(1,6),(2,18)]:
        si=np.where(cl['period'][ki]==periodval)[0][0];orb=cl['tail'][-periodval:,ki,si,:]
        inset.scatter(orb[:,0],orb[:,1],s=40,c=colors[i],marker=['o','s','^'][i])
    inset.set(xlim=(.497,.509),ylim=(.3431,.3445),title='Zoom: 6 and 18 differ')
    inset.tick_params(labelsize=7);inset.title.set_fontsize(8)
    axs[1].set(xlabel='x',ylabel='y',title='Distinct attracting cycles; points show one full period')
    axs[1].legend();fig.savefig(OUT/'coexisting_attractors.png',dpi=260);plt.close(fig)
    print('Created six figure PNGs')

if __name__=='__main__':main()
