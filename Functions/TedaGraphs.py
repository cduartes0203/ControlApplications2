import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.gridspec as gridspec
import matplotlib as mpl

def cm(x):
    return x/2.54

def plot_RUL_CI(teda, startX=None, startY=None, endX=None, endY=None,
                anchor=None, w=5.5, h=2, out=None, name='Name', lw1=1.5,
                lw2=1, mk1=3, mk2=6, ftcks=7, flbl=8.5, fttl=8, flgnd=7,
                loc=None, rect=[0, 0, 1, 1], ncol=1, tol=25):

    rulR = teda.rulR
    rulP = teda.rulP
    rulL = teda.rulL
    rulU = teda.rulU
    x = teda.cycleP

    eolX = teda.eolX
    eolY = rulP[np.where(x==eolX)[0][0]]
    end = np.where(x==eolX)[0][0] +1

    cmap = mpl.colormaps.get_cmap('tab20').resampled(len(teda.c))
    colors = [cmap(i) for i in range(len(teda.c))]

    fig, ax = plt.subplots(figsize=(w, h))

    ax.fill_between(x[:end], (1 - tol / 100) * rulR[:end], (1 + tol / 100) * rulR[:end],
                    color='gray', alpha=0.25, label=f"T-{tol}%", linewidth=lw1)
    ax.fill_between(x[:end], rulU[:end], rulL[:end],
                    color='skyblue', alpha=0.25, label=f"Uncertainty", linewidth=lw1)
    ax.plot([eolX, eolX], [-10, np.max(teda.rulU) * 1.2], color='black', linestyle='--', linewidth=lw2)
    ax.plot([x[0], x[-1]], [eolY,eolY], color='black', linestyle='--', linewidth=lw2)
    
    ax.plot(x[:end], rulR[:end], linestyle='-', linewidth=lw1, color='black', label="R-RUL")
    ax.plot(x[:end], rulP[:end], linestyle='-', linewidth=lw1, color='blue', label="P-RUL")
    ax.plot(x[:end], rulU[:end], linestyle='-', linewidth=lw2, color='blue')
    ax.plot(x[:end], rulL[:end], linestyle='-', linewidth=lw2, color='blue')


    for i,cloud in enumerate(teda.c):
        gx, gy = np.array(cloud.t)+teda.st-1, cloud.rul
        ax.plot(gx, gy, marker='o', color=colors[i], markersize=mk1,
            linestyle='', label=f'G{cloud.ID}')
    ax.plot(eolX, eolY, marker='x', color='black', markersize=mk2, linestyle='', markeredgewidth=mk2/3, label='EoL')
    ax.set_xlim(startX, endX)
    ax.set_ylim(startY, endY)
    ax.set_xlabel("Cycle", fontsize=flbl)
    ax.set_ylabel("RUL/Cycle", fontsize=flbl)
    ax.set_title(f'{name} - Granular RUL Prediction', fontsize=fttl)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.grid(True, linewidth=0.5)

    ax.legend(fontsize=flgnd, framealpha=0.85, loc=loc, bbox_to_anchor=anchor, ncol=ncol, columnspacing=0.7)

    fig.tight_layout(rect=rect)

    # Salvar
    if out is not None:
        fig.savefig(out, dpi=500, transparent=False)

    plt.show()

def plot_HI(teda,w=6,h=4,rect =[0,0,1,1],out=None,name=None,lnwdth=0.75,ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, anchor=None,png=True,  m1=1,m2=4,m3=1,ncol=None):
    if ncol==None:
        if teda.gCreated<5: ncol=1
        else: ncol=2
    if png:
        ext = '.png'
    else:
        ext = '.eps'
    x = teda.cycleP
    HI = teda.HI
    eolHI = teda.eol
    startX=teda.nI-2
    startY=0.15
    endX=teda.cycleP[-1]+2
    endY=np.max(teda.HI)*1.025
    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for j in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation)-1
    names = [f'G{i+1}' for i in range(qtd)]
    xr,yr = [[] for i in range(qtd)],[[] for i in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] ==i+1:
                yr[i].append(HI[l])
                xr[i].append(x[l])
 
    plt.figure(figsize=(w, h))
    for i in range(len(xr)):
        plt.plot(xr[i],yr[i],linestyle=' ',linewidth=lnwdth, marker='o', markersize = m1,label = names[i])
    plt.plot([0,endX],[eolHI,eolHI], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot([teda.eolX,teda.eolX],[-1,1], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot(teda.eolX, eolHI, marker='x', color='black', markersize=m2, linestyle='',markeredgewidth=m3, label='EOL')
    plt.xlim(startX, endX)
    plt.ylim(startY, endY)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("HI/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} HI',fontsize=fttl)  # Define o título do gráfico
    plt.grid(False)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = 'center left',bbox_to_anchor=anchor, ncol=ncol)
    plt.tight_layout(rect=rect) 
    if out != None and name != None:
        plt.savefig(out+'HI_'+name+ext, dpi=500,transparent=False)
    plt.show()  

'''def plot_DSI(teda,w=6,h=4,rect =[0,0,1,1],out=None,name=None,lnwdth=0.75,ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, anchor=None,png=True, m1=1,m2=4,m3=1,ncol=None):
    if ncol==None:
        if teda.gCreated<5: ncol=1
        else: ncol=2
    if png:
        ext = '.png'
    else:
        ext = '.eps'
    x = teda.cycleP
    DSI = teda.DSI
    eolDSI = teda.eolDSI
    startX=teda.nI-2
    startY=-np.min(teda.DSI)*5
    endX=teda.cycleP[-1]+2
    endY=np.max(teda.DSI)*1.025

    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for j in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation)-1
    names = [f'G{i+1}' for i in range(qtd)]
    xr,yr = [[] for i in range(qtd)],[[] for i in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] ==i+1:
                yr[i].append(DSI[l])
                xr[i].append(x[l])
 
    plt.figure(figsize=(w, h))
    for i in range(len(xr)):
        plt.plot(xr[i],yr[i],linestyle=' ',linewidth=lnwdth, marker='o', markersize = m1,label = names[i])
    plt.plot([0,endX],[eolDSI,eolDSI], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot([teda.eolX,teda.eolX],[-0.1,1], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',markeredgewidth=m3, label='EOL')
    #plt.xlim(startX, endX)
    #plt.ylim(startY, endY)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("DSI/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} DSI',fontsize=fttl)  # Define o título do gráfico
    plt.grid(False)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = 'upper right',bbox_to_anchor=anchor, ncol=ncol)
    plt.tight_layout(rect=rect) 
    if out != None and name != None:
        plt.savefig(out+'DSI_'+name+ext, dpi=500,transparent=False)
    plt.show()  

def plot_multiple_HI(teda_list, w=6, h=4,title_check=True, lnwdth=0.75, ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, m1=1, m2=4, m3=1, bearings_id=None, png=True, out=None, rect=[-0.025, 0.06, 1.02, 1.1]):
    n = len(teda_list)

    if png: ext = '.png'
    else: ext = '.eps'

    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)

    # Identifica o teda com maior número de gCreated e seu índice
    max_idx = max(range(n), key=lambda i: teda_list[i].gCreated)
    teda_max = teda_list[max_idx]

    for idx, teda in enumerate(teda_list):
        ax = axes[0, idx]
        x = teda.cycleP
        DSI = teda.HI
        eolDSI = teda.eol
        startX = teda.nI-4
        startY = -0.05
        endX = teda.cycleP[-1] + 4
        endY = 1.03+1

        # Processa ativações
        activation = []
        for arr in teda.cloud_activation2:
            aux = np.array([None for _ in range(teda.gCreated + 1)])
            for k in range(len(arr)):
                aux[int(arr[k])] = int(arr[k])
            activation.append(aux)
        activation = np.array(activation).T
        qtd = len(activation) - 1
        names = [f'G{i + 1}' for i in range(qtd)]
        xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

        for i in range(qtd):
            gran = activation[i + 1]
            for l in range(len(gran)):
                if gran[l] == i + 1:
                    yr[i].append(DSI[l])
                    xr[i].append(x[l])

        # Plot individual
        for i in range(len(xr)):
            ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth,
                    marker='o', markersize=m1, label=names[i])
        ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',
                markeredgewidth=m3, label='EOL')

        ax.set_xlim(startX, endX)
        ax.set_ylim(startY, endY)
        ax.tick_params(axis='both', labelsize=ftcks)
        ax.set_xlabel("Ciclo", fontsize=flbl)
        ax.set_ylabel("IS/Ciclo", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)

        if title_check: ax.set_title(f'{bearings_id[idx]}', fontsize=fttl)
        else:
            ax.annotate(bearings_id[idx],
                xy=(0, 0),                # canto superior direito
                xycoords='axes fraction',  # usa coordenadas normalizadas
                xytext=(5, 5),         # desloca um pouco para dentro
                textcoords='offset points',
                fontsize=flgnd,
                color='black',
                ha='left', va='bottom',      # alinha o texto à direita e ao topo
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
        ax.grid(False)

    # Usar os handles reais da instância com maior gCreated
    ax_max = axes[0, max_idx]
    handles, labels = ax_max.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    plt.suptitle('IS Granular', fontsize=10)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.05)

    if out is not None:
        plt.savefig(out + 'Bearings_HI' + ext, dpi=500, transparent=False)

    plt.show()'''

def plot_multiple_DSI(teda_list, w=6, h=4, lnwdth=0.75, ftcks=7, flbl=8.5, fttl=8.0, 
            flgnd=7, m1=1, m2=4, m3=1, bearings=None, png=True, out=None, rect=[-0.015, 0.06, 1.02, 1.1]):
    n = len(teda_list)

    if png: ext = '.png'
    else: ext = '.eps'

    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)

    # Identifica o teda com maior número de gCreated e seu índice
    max_idx = max(range(n), key=lambda i: teda_list[i].gCreated)
    teda_max = teda_list[max_idx]

    for idx, teda in enumerate(teda_list):
        ax = axes[0, idx]
        x = teda.cycleP
        DSI = teda.DSI
        eolDSI = teda.eolDSI
        startX = teda.nI-3
        startY = -0.05
        endX = teda.cycleP[-1] + 4
        endY = 0.95

        # Processa ativações
        activation = []
        for arr in teda.cloud_activation2:
            aux = np.array([None for _ in range(teda.gCreated + 1)])
            for k in range(len(arr)):
                aux[int(arr[k])] = int(arr[k])
            activation.append(aux)
        activation = np.array(activation).T
        qtd = len(activation) - 1
        names = [f'G{i + 1}' for i in range(qtd)]
        xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

        for i in range(qtd):
            gran = activation[i + 1]
            for l in range(len(gran)):
                if gran[l] == i + 1:
                    yr[i].append(DSI[l])
                    xr[i].append(x[l])

        # Plot individual
        for i in range(len(xr)):
            ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth,
                    marker='o', markersize=m1, label=names[i])
        ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',
                markeredgewidth=m3, label='EOL')

        ax.set_xlim(startX, endX)
        ax.set_ylim(startY, endY)
        ax.tick_params(axis='both', labelsize=ftcks)
        ax.set_xlabel("Ciclo", fontsize=flbl)
        ax.set_ylabel("IED/Ciclo", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)
        ax.set_title(f'{bearings[idx]}', fontsize=fttl - 1)
        ax.grid(False)

    # Usar os handles reais da instância com maior gCreated
    ax_max = axes[0, max_idx]
    handles, labels = ax_max.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    plt.suptitle('IED Granular', fontsize=10)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.05)

    if out is not None:
        plt.savefig(out + 'Bearings_DSI' + ext, dpi=500, transparent=False)

    plt.show()


def plot_multiple_HIandDSI(teda_list, w=cm(14), h=cm(10),title_check=True, lnwdth=0.75, ftcks=7,
                            flbl=8, fttl=8, flgnd=7, m1=2, m2=5, m3=1.2,
                              bearings_id=None, png=True, out=None, 
                              rect=[-0.025, 0.06, 1.02, 1.1], title='IED e IS Granulares'):
    n = len(teda_list)

    if png: ext = '.png'
    else: ext = '.eps'

    fig, axes = plt.subplots(2, n, figsize=(w, h), squeeze=False)

    # Identifica o teda com maior número de gCreated e seu índice
    max_idx = max(range(n), key=lambda i: teda_list[i].gCreated)
    teda_max = teda_list[max_idx]

    for idx, teda in enumerate(teda_list):
        ax = axes[1, idx]
        x = teda.cycleP
        DSI = teda.HI
        eolDSI = teda.eol
        startX = teda.nI-4
        startY = -0.05
        endX = teda.cycleP[-1] + 4
        endY = 0.99

        # Processa ativações
        activation = []
        for arr in teda.cloud_activation2:
            aux = np.array([None for _ in range(teda.gCreated + 1)])
            for k in range(len(arr)):
                aux[int(arr[k])] = int(arr[k])
            activation.append(aux)
        activation = np.array(activation).T
        qtd = len(activation) - 1
        names = [f'G{i + 1}' for i in range(qtd)]
        xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

        for i in range(qtd):
            gran = activation[i + 1]
            for l in range(len(gran)):
                if gran[l] == i + 1:
                    yr[i].append(DSI[l])
                    xr[i].append(x[l])

        # Plot individual
        for i in range(len(xr)):
            ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth,
                    marker='o', markersize=m1, label=names[i])
        ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',
                markeredgewidth=m3, label='FV')

        yticks = [i * 0.2 for i in range(12)] + [0.3]

        if len(x)>=80: xticks = [i * 25 for i in range(12)]
        if len(x)<80: xticks = [i * 10 for i in range(12)]

        ax.set_yticks(sorted(yticks))
        ax.set_xticks(sorted(xticks))
        ax.set_xlim(startX, endX)
        ax.set_ylim(startY, endY)
        ax.tick_params(axis='both', labelsize=ftcks)
        ax.set_xlabel("Ciclo", fontsize=flbl)
        ax.set_ylabel("IS/Ciclo", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)

        if title_check: ax.set_title(f'{bearings_id[idx]}', fontsize=fttl)
        else:
            ax.annotate(bearings_id[idx],
                xy=(0, 0),                # canto superior direito
                xycoords='axes fraction',  # usa coordenadas normalizadas
                xytext=(5, 5),         # desloca um pouco para dentro
                textcoords='offset points',
                fontsize=flgnd,
                color='black',
                ha='left', va='bottom',      # alinha o texto à direita e ao topo
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
        ax.grid(False)

    for idx, teda in enumerate(teda_list):
        ax = axes[0, idx]
        x = teda.cycleP
        DSI = teda.DSI
        eolDSI = teda.eolDSI
        startX = teda.nI-3
        startY = -0.05
        endX = teda.cycleP[-1] + 4
        endY = 0.95

        # Processa ativações
        activation = []
        for arr in teda.cloud_activation2:
            aux = np.array([None for _ in range(teda.gCreated + 1)])
            for k in range(len(arr)):
                aux[int(arr[k])] = int(arr[k])
            activation.append(aux)
        activation = np.array(activation).T
        qtd = len(activation) - 1
        names = [f'G{i + 1}' for i in range(qtd)]
        xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

        for i in range(qtd):
            gran = activation[i + 1]
            for l in range(len(gran)):
                if gran[l] == i + 1:
                    yr[i].append(DSI[l])
                    xr[i].append(x[l])

        # Plot individual
        for i in range(len(xr)):
            ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth,
                    marker='o', markersize=m1, label=names[i])
        ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',
                markeredgewidth=m3, label='FV')

        if len(x)>=80: xticks = [i * 25 for i in range(12)]
        if len(x)<80: xticks = [i * 10 for i in range(12)]

        ax.set_yticks(sorted(yticks))
        ax.set_xticks(sorted(xticks))
        ax.set_xlim(startX, endX)
        ax.set_ylim(startY, endY)
        ax.tick_params(axis='both', labelsize=ftcks)
        #ax.set_xlabel("Cycle", fontsize=flbl)
        ax.set_ylabel("IED/Ciclo", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)
        if title_check: ax.set_title(f'{bearings_id[idx]}', fontsize=fttl)
        else:
            ax.annotate(bearings_id[idx],
                xy=(0, 0),                # canto superior direito
                xycoords='axes fraction',  # usa coordenadas normalizadas
                xytext=(5, 5),         # desloca um pouco para dentro
                textcoords='offset points',
                fontsize=flgnd,
                color='black',
                ha='left', va='bottom',      # alinha o texto à direita e ao topo
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
        ax.grid(False)

    # Usar os handles reais da instância com maior gCreated
    ax_max = axes[0, max_idx]
    handles, labels = ax_max.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.05,hspace=0.3)

    if out is not None:
        plt.savefig(out + 'RLRNN_HiDsi' + '.pdf', dpi=500, transparent=False)

    plt.show()

def metrics(teda, w=6, h=4, out=None, name=None, lnwdth=2, ftcks=14, flbl=16, fttl=18, flgnd=14):
    rulR = np.array(teda.rulR)
    rulP = np.array(teda.rulP)

    rmse = []
    mae = []
    for i in range(1, len(rulR)):  # Começa de 1 para evitar vetores vazios
        rmse.append(np.sqrt(mean_squared_error(rulR[:i], rulP[:i])))
    for i in range(1,len(rulR)+1):
      mae.append((mean_absolute_error(rulR[:i], rulP[:i])))
    
    # Cria uma figura com 2 gráficos lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(w, h))

    # Primeiro gráfico: RMSE
    axes[0].plot(rmse, linewidth=lnwdth)
    axes[0].set_title(f'{name[:-4]}: \n RUL Prediction RMSE',fontsize=fttl)
    axes[0].set_xlabel('Cycle', fontsize=flbl)
    axes[0].set_ylabel('RMSE', fontsize=flbl)
    axes[0].grid(True, linestyle='--', linewidth=0.5)
    axes[0].tick_params(labelsize=ftcks)

    # Segundo gráfico: rulR vs rulP
    axes[1].plot(mae, linewidth=lnwdth)
    axes[1].set_title(f'{name[:-4]}: \n RUL Prediction MAE',fontsize=fttl)
    axes[1].set_xlabel('Cycle', fontsize=flbl)
    axes[1].set_ylabel('MAE', fontsize=flbl)
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].tick_params(labelsize=ftcks)

    plt.tight_layout()

    # Salva se desejar
    if out is not None and name is not None:
        plt.savefig(out + 'Performance_'+name, dpi=500)
    
    plt.show()

def metric2(teda, w=6, h=4, out=None, name=None, lnwdth=2, ftcks=14, flbl=16, fttl=18, flgnd=14):
    rulR = np.array(teda.rulR)
    rulP = np.array(teda.rulP)

    rmse = []
    mae = []
    for i in range(1, len(rulR)):  # Começa de 1 para evitar vetores vazios
        rmse.append(np.sqrt(mean_squared_error(rulR[:i], rulP[:i])))
    for i in range(1,len(rulR)+1):
      mae.append((mean_absolute_error(rulR[:i], rulP[:i])))
    
    # Cria uma figura com 2 gráficos lado a lado
    plt.figure(figsize=(w, h))

    # Primeiro gráfico: RMSE
    plt.plot(rmse,color='blue',linewidth=lnwdth,label='RMSE')
    plt.plot(mae,color='red',linewidth=lnwdth,label='MAE')

    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("Metric Value",fontsize=flbl)  # Nome do eixo Y

    plt.title(f'{name[:-4]}:\nPerformance',fontsize=fttl)  # Define o título do gráfico
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=1,loc='upper right')
    plt.tight_layout()

    # Salva se desejar
    if out is not None and name is not None:
        plt.savefig(out + 'Performance_'+name, dpi=500)
    
    plt.show()

def plot_multiple_RMSE(teda_list, w=6, h=2, lnwdth=0.75,
                        ftcks=7, flbl=8, fttl=8.0, flgnd=7,
                        m1=1, m2=4, m3=1,
                        bearings_id=None, png=True,title_check=True,
                        out=None, rect=[-0.015, 0.06, 1.02, 1.1]):
    n = len(teda_list)

    if png: ext = '.png'
    else: ext = '.eps'

    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)

    # Identifica o teda com maior número de gCreated e seu índice
    max_idx = max(range(n), key=lambda i: teda_list[i].gCreated)
    teda_max = teda_list[max_idx]
    p1,p2,p3,p4 = 0.25,0.5,0.75,1
    for idx, teda in enumerate(teda_list):
        ax = axes[0, idx]
        
        x,y = teda.RMSE()
        ax.plot(x,y, color='blue', linewidth=lnwdth,marker='o',label='RMSE',markersize=m1)
        ax.tick_params(axis='both', labelsize=ftcks)
        if len(x) >=80: xticks = [i * 35 for i in range(10)]
        if len(x) <80: xticks = [i * 10 for i in range(10)]
        yticks = [i * 5 for i in range(10)]
        ax.set_xticks(sorted(xticks))
        ax.set_yticks(sorted(yticks))
        ax.set_xlim(x[0]-5, x[-1]+5)
        if len(x) <80: xticks = ax.set_xlim(x[0]-2, x[-1]+2)
        ax.set_ylim(0, 15)
        ax.set_xlabel("Ciclo", fontsize=flbl)
        ax.set_ylabel("RMSE/Ciclo", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)
        if title_check: ax.set_title(f'{bearings_id[idx]}', fontsize=fttl)
        else:
            ax.annotate(bearings_id[idx],
                xy=(1, 1),                # canto superior direito
                xycoords='axes fraction',  # usa coordenadas normalizadas
                xytext=(-5, -5),         # desloca um pouco para dentro
                textcoords='offset points',
                fontsize=flgnd,
                color='black',
                ha='right', va='top',      # alinha o texto à direita e ao topo
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
        ax.grid(True)

    # Usar os handles reais da instância com maior gCreated
    ax_max = axes[0, max_idx]
    handles, labels = ax_max.get_legend_handles_labels()

    #fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    plt.suptitle('Predição de RUL: RMSE', fontsize=10)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.05)

    if out is not None:
        plt.savefig(out + 'RLRNN_RMSE' + '.pdf', dpi=500, transparent=False)

    plt.show()

def _plot_RUL_CI_on_ax(ax, teda,startX=None, name='Name', lw1=1, lw2=0.75, ftcks=7, flbl=8, fttl=8, flgnd=7, m1=1, m2=5, m3=1.5):
    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for _ in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation) - 1
    names = [f'G{i+1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

    eolRUL = teda.rulP[int(teda.eolX - teda.nI)]
    rulR, rulP, rulL, rulU = teda.rulR, teda.rulP, teda.rulL, teda.rulU
    x = teda.cycleP

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] == i+1:
                yr[i].append(rulP[l])
                xr[i].append(x[l])

    if startX==None:
        startX = teda.cycleP[int(np.where(teda.rulP == np.max(teda.rulP) - 1)[0][0])]+1
    endX = int(teda.eolX)
    startY = 0
    endY = np.max(teda.rulP) - 2.5

    ax.plot(x, rulR, '-', linewidth=lw1, color='black', label="Real RUL")
    ax.plot(x, rulP, '-', linewidth=lw2, color='blue', label="Predicted RUL")
    ax.plot(x, rulU, '--', linewidth=lw1, color='blue')
    ax.plot(x, rulL, '--', linewidth=lw1, color='blue')
    ax.fill_between(x, 0.8 * rulR, 1.2 * rulR, color='skyblue', alpha=0.15, label="Tolerance 20%")
    ax.fill_between(x, rulL, rulU, color='lightgray', alpha=0.25, label="Uncertainty")

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', marker='o', markersize=m1*2, label=names[i])

    ax.plot(teda.eolX, eolRUL, marker='x', color='black', markersize=m2, linestyle='', markeredgewidth=m3, label='EOL')
    ax.plot([teda.eolX, teda.eolX], [-200, +200], color='black', linewidth=lw2, linestyle=':')
    ax.plot([-200, +200], [eolRUL, eolRUL], color='black', linewidth=lw2, linestyle=':')
    ax.set_xlim(startX, endX + 1)
    ax.set_ylim(startY, endY + 1)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.set_xlabel("Ciclo", fontsize=flbl)
    ax.set_ylabel("RUL/Ciclo", fontsize=flbl)
    ax.set_title(f'{name}', fontsize=fttl)
    ax.grid(True, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _plot_HI_on_ax(ax, teda, name=None, lnwdth=0.75, ftcks=7, flbl=8, fttl=8, flgnd=7, m1=1, m2=5, m3=1.5):
    x = teda.cycleP
    HI = teda.HI
    eolHI = teda.eol
    startX = teda.nI - 2
    startY = 0.15
    endX = x[-1] + 2
    endY = np.max(HI) * 1.025

    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for _ in range(teda.gCreated + 1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T

    qtd = len(activation) - 1
    names = [f'G{i+1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] == i+1:
                yr[i].append(HI[l])
                xr[i].append(x[l])

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth, marker='o', markersize=m1, label=names[i])

    ax.plot([0, endX], [eolHI, eolHI], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot(teda.eolX, eolHI, marker='x', color='black', markersize=m2, linestyle='', markeredgewidth=m3, label='EOL')

    ax.set_xlim(startX, endX)
    ax.set_ylim(startY, endY)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.set_xlabel("Ciclo", fontsize=flbl)
    ax.set_ylabel("IS/Ciclo", fontsize=flbl)
    ax.set_title(f'{name}', fontsize=fttl)
    ax.grid(False)


def _plot_DSI_on_ax(ax, teda, name=None, lnwdth=0.75, ftcks=7, flbl=8, fttl=8, flgnd=7, m1=1, m2=5, m3=1.5):
    x = teda.cycleP
    DSI = teda.DSI
    eolDSI = teda.eolDSI
    startX = teda.nI - 2
    startY = np.min(DSI) -0.025
    endX = x[-1] + 2
    endY = np.max(DSI) * 1.025

    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for _ in range(teda.gCreated + 1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T

    qtd = len(activation) - 1
    names = [f'G{i+1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] == i+1:
                yr[i].append(DSI[l])
                xr[i].append(x[l])

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth, marker='o', markersize=m1, label=names[i])

    ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='', markeredgewidth=m3, label='EOL')

    ax.set_xlim(startX, endX)
    ax.set_ylim(startY, endY)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.set_xlabel("Cycle", fontsize=flbl)
    ax.set_ylabel("DSI/Cycle", fontsize=flbl)
    ax.set_title(f'{name}', fontsize=fttl)
    ax.grid(False)


def plot_RUL_HI_DSI_side_by_side(teda,startX=None, name="Name", w=cm(14), h=cm(6), out=None, png=True, rect=[-0.025, 0.09, 1.025, 1.08]):
    fig, axes = plt.subplots(1, 3, figsize=(w, h), gridspec_kw={'width_ratios': [1.5, 1.5, 3]})

    # Plot subplots e captura da legenda apenas do primeiro
    _plot_DSI_on_ax(axes[0], teda, name='a) DSI Granular')
    _plot_HI_on_ax(axes[1], teda, name='b) HI Granular HI')
    handles, labels = _plot_RUL_CI_on_ax(axes[2], teda, startX=startX,name='c) Granular RUL prediction')

    plt.suptitle(name, fontsize=10)

    # Legenda horizontal centralizada abaixo da figura
    if len(labels)< 8: ncol=len(labels)
    else: ncol=7
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=ncol, fontsize=7, framealpha=0.85,
               columnspacing=0.5, handletextpad=0.5)

    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.35)
    if out is not None:
        ext = '.png' if png else '.eps'
        plt.savefig(out + name + ext, dpi=500, transparent=True)

    plt.show()

def plot_RUL_ax(ax,teda, startX=None, startY=None, endX=None, endY=None,
                anchor=None, out=None, name='Name',
                lw1=1.25, lw2=1, mk1=5, mk2=1.5, ftcks=8, flbl=10, fttl=8, flgnd=8,
                dotsize=2, loc=None, ncol=None, tol=30,
                title=True, xlabel=True, ylabel=True):

    if ncol is None:
        ncol = 2 if teda.gCreated > 4 else 1

    rulR = teda.rulR
    rulP = teda.rulP
    rulL = teda.rulL
    rulU = teda.rulU
    eolRUL = teda.rulP[int(teda.eolX - teda.nI)]
    x = teda.cycleP
    end = np.where(teda.cycleP == teda.eolX)[0][0] + 1

    size = int(teda.eolX/6)
    while size%5!=0:
        size=size+1

    # Organizando os dados por granulação
    activation = []
    for arr in teda.cloud_activation2[:end]:
        aux = np.array([None for _ in range(teda.gCreated + 1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation) - 1

    names = [f'G{i + 1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]
    for i in range(qtd):
        gran = activation[i + 1]
        for l in range(len(gran)):
            if gran[l] == i + 1:
                yr[i].append(rulP[l])
                xr[i].append(x[l])

    # Limites dos eixos
    if endX is None:
        if len(x) >= 60: endX=x[-1]+0.5
        elif len(x) < 60: endX=x[-1]+0.5
    if startX is None:
        if len(x) >= 60: startX=x[-1]*0.15
        elif len(x) < 60: startX=x[0]
    if startY is None:
        if rulR[0] >= 90: startY=eolRUL - 10
        elif rulR[0] < 90: startY = eolRUL - 5
    if endY is None:
        endY = np.max(rulR)*1
    
    # Linhas principais
    ax.fill_between(x[:], (1 - tol / 100) * rulR[:], (1 + tol / 100) * rulR[:],
                    color='gray', alpha=0.6, label=f"Tolerância \u00B1{tol}%", linewidth=lw1)
    ax.fill_between(x[:end], rulL[:end], rulU[:end], color='skyblue', alpha=0.5, label="Incerteza",)
    ax.plot(x[:], rulR[:], linestyle='-', linewidth=lw1*0.8, color='black', label="RUL Real")
    ax.plot(x[:end], rulP[:end], linestyle='-', marker='o', markersize=dotsize, linewidth=lw1, color='blue', label="RUL Predita")
    ax.plot(x[:end], rulU[:end], linestyle='-', linewidth=lw1, color='blue')
    ax.plot(x[:end], rulL[:end], linestyle='-', linewidth=lw1, color='blue')
    ax.plot([teda.eolX, teda.eolX], [-1, np.max(teda.rulU*20) * 1.2], color='black', linestyle=':', linewidth=lw2)
    ax.plot([-200, 200], [eolRUL, eolRUL], color='black', linestyle=':', linewidth=lw2)

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', marker='o', markersize=dotsize, label=names[i])

    ax.plot(teda.eolX, eolRUL, marker='x', color='black', markersize=mk1,
            linestyle='', markeredgewidth=mk2, label='FV')

    # Ajustes visuais
    if len(teda.cycleP) >=80: xticks = [i * 15 for i in range(10)]
    if len(teda.cycleP) <80: xticks = [i * 5 for i in range(60)] 

    if teda.eolX not in xticks:
        xticks=xticks+[teda.eolX]


    if (teda.rulR[0]) >=80: 
        yticks = [i * 30 for i in range(1,20)]+[eolRUL]
        #endY=np.max(teda.rulU[0:])*1.3
    if (teda.rulR[0]) <80: 
        yticks = [i * 10 for i in range(1,20)]+[eolRUL]
        #endY=np.max(teda.rulU[0:])*1.3
    
        
    ax.set_xticks(sorted(xticks))
    ax.set_yticks(sorted(yticks))
    ax.set_xlim(startX, endX)
    ax.set_ylim(startY, endY)
    if xlabel: ax.set_xlabel("Ciclo", fontsize=flbl)
    if ylabel: ax.set_ylabel("RUL/Ciclo", fontsize=flbl)
    
    if title: ax.set_title(f'{name}', fontsize=fttl)
    else:
        ax.annotate(name,
            xy=(1, 1),                # canto superior direito
            xycoords='axes fraction',  # usa coordenadas normalizadas
            xytext=(-5, -5),         # desloca um pouco para dentro
            textcoords='offset points',
            fontsize=flgnd,
            color='black',
            ha='right', va='top',      # alinha o texto à direita e ao topo
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
    
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.grid(True, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels

def plot_RUL_GRID(tedas,startX=None,name='Bearings',title='Predição Granular de RUL' ,bearings_id=None,
                                  w=cm(14), h=cm(12), out=None, png=True,
                                    rect=[-0.025,0.06,1.025,1.02]):
    
    ncols = 3
    nrows = int(len(tedas)/ncols)
    handles, labels=[],[]
    ylabel= True
    if bearings_id==None: bearings_id=bearings_id=[f'Bearing {i+1}'for i in range(ncols)]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), gridspec_kw={'width_ratios': [ncols for i in range(ncols)]})

    if nrows==1:
        for j in range(ncols):
            if j>0: ylabel=False
            handles1, labels1 = plot_RUL_ax(axes[j], tedas[j], 
                                    name=bearings_id[j],ylabel=ylabel, title=False)
            handles.append(handles1)
            labels.append(labels1)
    
    if nrows>1:
        xlabel=False
        for i in range(nrows):
            ylabel=True
            for j in range(ncols):
                if j>0: ylabel=False
                if i==nrows-1: xlabel=True
                handles1, labels1 = plot_RUL_ax(axes[i,j], tedas[j+(ncols*i)], 
                                        name=bearings_id[j+(ncols*i)],xlabel=xlabel,ylabel=ylabel, title=False)
                handles.append(handles1)
                labels.append(labels1)

    id = max(enumerate(labels), key=lambda x: len(x[1]))[0]

    plt.suptitle(title, fontsize=10)
    if len(labels[id])< 8: ncol=len(labels[id])
    else: ncol=7

    fig.legend(handles[id], labels[id], loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, fontsize=7, framealpha=0.85,
               columnspacing=0.5, handletextpad=0.5)
    
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.17)

    if out is not None:
        ext = '.png' if png else '.eps'
        plt.savefig(out + name + '.pdf', dpi=500, transparent=False)

    plt.show()

def plot_RUL_LIST(tedas,name='Bearings2',title='Predição Granular de RUL',
                   bearings_id=None,w=cm(14), h=cm(12),
                   out=None,rect=[-0.15,0.035,1.02,1.02],png=True):

    nrow = len(tedas)
    # Se houver apenas um subplot, gridspec_kw pode causar um erro.
    # Removendo temporariamente o argumento para simplificar.
    # plt.subplots retorna um único objeto Axes se nrow=1.
    fig, axes = plt.subplots(nrow, 1, figsize=(w, h))

    # Converte 'axes' em um array 1D, se já não for.
    axes = np.ravel(axes)

    handles, labels=[],[]
    if bearings_id==None: bearings_id=[f'Bearing {i+1}' for i in range(nrow)]

    for i,teda in enumerate(tedas):
        if i==len(tedas)-1: xlabel=True
        else: xlabel=False
        handle, label = plot_RUL_ax(axes[i], tedas[i],name=bearings_id[i],
                                    title=False,xlabel=xlabel)
        handles.append(handle)
        labels.append(label)

    id = max(enumerate(labels), key=lambda x: len(x[1]))[0]

    plt.suptitle(title, fontsize=12)
    if len(labels[id])< 8: ncol=len(labels[id])
    else: ncol=7

    fig.legend(handles[id], labels[id], loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=6, fontsize=9, framealpha=0.85,
               columnspacing=0.5, handletextpad=0.5)
    
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(hspace=0.35)

    if out is not None:
        ext = '.pdf' if png else '.eps'
        plt.savefig(out + name + '.pdf', dpi=500, transparent=True)

    plt.show()

def plot_TS(teda,startX=None,startY=None,endX=None,endY=None,
                anchor=None,w=cm(10),h=cm(5),out=None,name='Name',
                lw1=1.25,lw2=0.5,lw3=0.75,ftcks=7,flbl=8,fttl=8,flgnd=7,dotsize=2,
                loc='lower center',rect=[-0.03,-0.06,1.02,1.06],
                png=False,ncol=None,plt_L=True,plt_U=True,plt_P=True,plt_R=True):
    
    if startX==None:  startX=0
    if endX==None:  endX=len(teda.HIp)
    if startY==None:  startY=np.min(teda.HIpL)*0.95
    if endY==None:  endY=np.max(teda.HIpU)*1.05

    plt.figure(figsize=(w, h)) 
    if plt_R: plt.plot(teda.cycleP, teda.HI, linestyle='-',linewidth=lw1, color='Black', label="Real")
    if plt_P: plt.plot(teda.cycleP, teda.HIp, linestyle='--',linewidth=lw2, color='red', label="Predicted")
    if plt_L: plt.plot(teda.cycleP, teda.HIpL, linestyle='-',linewidth=lw3, color='green', label="Lower bound")
    if plt_U: plt.plot(teda.cycleP, teda.HIpU, linestyle='-',linewidth=lw3, color='blue', label="Upper bound")
    plt.xlim(startX, endX)
    plt.ylim(startY, endY)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("RUL/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} - Time series prediction',fontsize=fttl)  # Define o título do gráfico
    plt.grid(True,linewidth=0.5)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = loc,bbox_to_anchor=anchor, ncol=2,columnspacing=0.5)
    plt.tight_layout(rect=rect) 
    if out is not None:
        ext = '.png' if png else '.eps'
        plt.savefig(out + name + ext, dpi=500, transparent=True)
    plt.show()

def plot_TS_Metrics(teda,name='predictions', w=cm(14), h=cm(9),s=952,e=1020,
             lw1=1,lw2=1.25, ftcks=7,
            flbl=8, fttl=8, flgnd=7, m1=2, m2=5, m3=1.2,
                              png=True, out=None, 
                              rect=[-0.02, 0.05, 1, 1.03], title='Granular DSI and HI'):

    if png: ext = '.png'
    else: ext = '.eps'
    x=teda.cycleP
    HI=(teda.HI/10).copy()
    HIp=(teda.HIp/10).copy()
    ndei=teda.NDEI()
    yL=int(np.min(HI[s:e]))-5
    yU=int(np.max(HI[s:e]))+5
    xL= np.min(x[s:e])
    xU= np.max(x[s:e])

    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1])  # mesma altura nas duas linhas

    # Gráfico 1: ocupa toda a primeira linha (todas as 4 colunas)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.tick_params(labelsize=ftcks)
    ax1.plot(x, HI,color='blue', label="Real",linestyle='-',linewidth=lw1)
    ax1.plot(x, HIp,color='red', label="Predicted",linestyle=':',linewidth=lw2)
    ax1.plot([s,s],[yL,yU],color='black',linestyle='-',linewidth=lw1*1.5)
    ax1.plot([e,e],[yL,yU],color='black',linestyle='-',linewidth=lw1*1.5)
    ax1.plot([s,e],[yL,yL],color='black',linestyle='-',linewidth=lw1*1.5)
    ax1.plot([s,e],[yU,yU],color='black',linestyle='-',linewidth=lw1*1.5)
    ax1.set_xlim(0, x[-1])
    #ax1.set_ylim(15, 110)
    #ax1.set_xlabel('Time', fontsize=flbl)
    ax1.set_ylabel('Temperature (°C)', fontsize=flbl)
    ax1.set_title("Death Valley Temperature Time Series Prediction",fontsize=fttl)
    #ax1.legend(fontsize=flgnd, ncol=3)
    ax1.annotate('Zoom in',
            xy=(0.825, 1.15),                # canto superior direito
            xycoords='axes fraction',  # usa coordenadas normalizadas
            xytext=(-10, -10),         # desloca um pouco para dentro
            textcoords='offset points',
            fontsize=flgnd,
            color='black',
            ha='right', va='top',      # alinha o texto à direita e ao topo
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0))
    ax1.grid(False)

    # Gráfico 2: ocupa 3 colunas da segunda linha (75%)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.tick_params(labelsize=ftcks)
    ax2.plot(x[s:e], HI[s:e],color='blue', label="Real",linestyle='-',linewidth=lw1*1.35)
    ax2.plot(x[s:e], HIp[s:e],color='red', label="Predicted",linestyle='--',linewidth=lw1)
    ax2.set_xlim(x[s], x[e-1])
    ax2.set_ylim(yL, yU)
    #ax2.set_xlabel('Time', fontsize=flbl)
    ax2.set_ylabel('Temperature (°C)', fontsize=flbl)
    #ax2.set_title("Zoom in",fontsize=fttl)
    #ax2.legend(fontsize=flgnd, ncol=2)
    ax2.annotate('Zoom in',
            xy=(0.125, 1.1),                # canto superior direito
            xycoords='axes fraction',  # usa coordenadas normalizadas
            xytext=(-10, -10),         # desloca um pouco para dentro
            textcoords='offset points',
            fontsize=flgnd,
            color='black',
            ha='right', va='top',      # alinha o texto à direita e ao topo
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
    ax2.grid(False)

    # Gráfico 3: ocupa 1 coluna restante da segunda linha (25%)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.tick_params(labelsize=ftcks)
    ax3.plot(x, ndei, 'g',linewidth=lw1, label="NDEI")
    ax3.set_xlabel('Time', fontsize=flbl)
    ax3.set_ylabel('NDEI', fontsize=flbl)
    ax3.set_xlim(0, x[-1])
    #ax3.set_title("NDEI",fontsize=fttl)
    ax3.annotate(f'{round(ndei[-1],3)}',
            xy=(1.0125, 0.55),                # canto superior direito
            xycoords='axes fraction',  # usa coordenadas normalizadas
            xytext=(-10, -10),         # desloca um pouco para dentro
            textcoords='offset points',
            fontsize=flgnd,
            color='black',
            ha='right', va='top',      # alinha o texto à direita e ao topo
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0))
    ax3.grid(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=7, framealpha=0.85,
               columnspacing=0.5, handletextpad=0.5)
    #fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    #plt.suptitle(title, fontsize=10)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.7,hspace=0.5)

    if out is not None:
        plt.savefig(out + name + ext, dpi=500, transparent=False)

    plt.show()

def plot_DSI(teda, w=15, h=4,start=None,ftrs=None, out=None, title=None,
             mergeID=True,lnwdth=0.75, ftcks=7, flbl=9, fttl=8.5,
             dsix=0,dsiy=1,
             flgnd=7, anchor=None, m1=3, m2=4, m3=1, ncol=1):
    
    L,U = np.inf,-np.inf
    qtd = len(teda.c)
    DSIs = teda.DSIs[start:ftrs]
    series = [[[] for _ in range(qtd)] for i in range(len(DSIs)+1)]
    cmap = mpl.colormaps.get_cmap('tab20').resampled(qtd)
    colors = [cmap(i) for i in range(qtd)]
    #ChangePoints = teda.ChangePoint
    
    yticks = [i*0.25 for i in range(1,6)]+[0]
    merged_track = [(sorted(c.track, reverse=True)) for c in teda.c if len(c.track)>1]
    merged_track = [f'G{cloud[0]} = {cloud[1:]}' for cloud in merged_track]
    merged_track  = "; ".join(merged_track)
    if len(merged_track) == 0: merged_track = 'Merged Granules: None'
    else: merged_track = 'Merged Granules: ' + merged_track

    merged_clouds = []
    for cloud in teda.c:
        if len(cloud.track) > 1:
            merged_clouds.append(cloud.track)
    for i, array in enumerate(teda.cloud_activation):
        for ref in merged_clouds:
            if np.intersect1d(ref, array).size > 0:
                u = np.intersect1d(ref, array)
                u[:] = ref[0]
                teda.cloud_activation[i] = u

    clouds_ID = [cloud.ID for cloud in teda.c]
    names = [f'G{cloud_ID}' for cloud_ID in clouds_ID]

    for i, cloud in enumerate(teda.cloud_activation):
        for j, cloud_ID in enumerate(clouds_ID):
            if np.flip(cloud)[-1] == cloud_ID:
            #if (cloud)[-1] == cloud_ID:
                for k in range(len(series)-1):
                    series[k][j].append(DSIs[k][i])
                series[-1][j].append(i)
    
    fig = plt.figure(figsize=(w, h))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12,
                    wspace=0.12, hspace=0.15)
    
    gs = fig.add_gridspec(nrows=len(DSIs), ncols=2, width_ratios=[1, 2.5],
    wspace=0.125, hspace=0.15)
    
    ax1 = fig.add_subplot(gs[:, 0]) 
    axes = [fig.add_subplot(gs[i, 1]) for i in range(len(DSIs))]

    for i in range(len(series[0])):
        ax1.plot(series[dsix][i], series[dsiy][i], linestyle=' ', linewidth=lnwdth, marker='o'
                          , markersize=m1, label=names[i], color=colors[i])
        ax1.plot(teda.c[i].mean[0], teda.c[i].mean[1], linestyle=' '
                          , color='black', marker='x', markersize=m2)
    
    ax1.plot(teda.c[i].mean[0], teda.c[i].mean[1], linestyle=' '
                          , color='black', marker='x', markersize=m2, label = 'G centroid')

    ax1.set_xlabel(f"DSI-{dsix+1}", fontsize=flbl)
    ax1.set_ylabel(f"DSI-{dsiy+1}", fontsize=flbl)
    ax1.legend(fontsize=flgnd, framealpha=0.85, bbox_to_anchor=anchor, ncol=ncol)
    ax1.grid()
    for j,ax in enumerate(axes):
        #L,U = np.inf,-np.inf
        for i in range(len(series[0])):
            ax.plot(series[-1][i], series[j][i], linestyle=' ', marker='o',
                    markersize=m1, label=None, color=colors[i])
            
            if L>np.min(series[j][i]):
                L = np.min(series[j][i]) -0.15*np.min(series[j][i])
            else: L = L
            if U<np.max(series[j][i]):
                U = np.max(series[j][i])+ 0.05*np.max(series[j][i])
            else: U = U

            #yticks = list(np.linspace(L, U, 5))
            #ax.set_ylim(L, U)
            #ax.set_yticks(sorted(yticks))

            ax.set_ylabel(f"DSI-{j+1} Value", fontsize=flbl)
            ax.set_xlabel(f"Cycle", fontsize=flbl)
            #ax.set_yticks(sorted(yticks))
            
            if j< len(axes)-1: 
                ax.set_xlabel('')
                ax.tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)
            ax.grid()
        #for cp in ChangePoints:
        #    ax.plot([cp[0],cp[0]],[L,U], linestyle='--')
        #    ax.plot([cp[1],cp[1]],[L,U], linestyle='--')

    fig.suptitle(title,y=0.95)
    if mergeID: fig.text(0.5, 0.1, merged_track, fontsize=10, ha='center', va='bottom', transform=fig.transFigure)
    if out is not None and title is not None:
        plt.savefig(out, dpi=500, transparent=False)
    
    plt.show()

def plot_DSI_3D(teda, w=15, h=4,start=None,ftrs=None, out=None, title=None,
             mergeID=True,lnwdth=0.75, ftcks=7, flbl=9, fttl=8.5,
             dsix=0,dsiy=1,dsiz=2,
             flgnd=7, anchor=None, m1=3, m2=4, m3=1, ncol=1):
    
    L,U = np.inf,-np.inf
    qtd = len(teda.c)
    DSIs = teda.DSIs[start:ftrs]
    series = [[[] for _ in range(qtd)] for i in range(len(DSIs)+1)]
    #cmap = mpl.colormaps.get_cmap('tab20').resampled(qtd)
    #colors = [cmap(i) for i in range(qtd)]

    cmap = mpl.colormaps.get_cmap('tab20').resampled(10)
    colors = [cmap(i) for i in range(10)]
    radii = [cloud.R for cloud in teda.c]
    #ChangePoints = teda.ChangePoint
    
    yticks = [i*0.25 for i in range(1,6)]+[0]
    merged_track = [(sorted(c.track, reverse=True)) for c in teda.c if len(c.track)>1]
    merged_track = [f'G{cloud[0]} = {cloud[1:]}' for cloud in merged_track]
    merged_track  = "; ".join(merged_track)
    if len(merged_track) == 0: merged_track = 'Merged Granules: None'
    else: merged_track = 'Merged Granules: ' + merged_track

    merged_clouds = []
    for cloud in teda.c:
        if len(cloud.track) > 1:
            merged_clouds.append(cloud.track)
    for i, array in enumerate(teda.cloud_activation):
        for ref in merged_clouds:
            if np.intersect1d(ref, array).size > 0:
                u = np.intersect1d(ref, array)
                u[:] = ref[0]
                teda.cloud_activation[i] = u

    clouds_ID = [cloud.ID for cloud in teda.c]
    names = [f'G{cloud_ID}' for cloud_ID in clouds_ID]

    for i, cloud in enumerate(teda.cloud_activation):
        for j, cloud_ID in enumerate(clouds_ID):
            if np.flip(cloud)[-1] == cloud_ID:
            #if (cloud)[-1] == cloud_ID:
                for k in range(len(series)-1):
                    series[k][j].append(DSIs[k][i])
                series[-1][j].append(i)
    
    fig = plt.figure(figsize=(w, h))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12,
                    wspace=0.12, hspace=0.15)
    
    gs = fig.add_gridspec(nrows=len(DSIs), ncols=2, width_ratios=[1, 2.5],
    wspace=0.125, hspace=0.15)
    
    ax1 = fig.add_subplot(gs[:, 0], projection='3d') 
    axes = [fig.add_subplot(gs[i, 1]) for i in range(len(DSIs))]

    for i in range(len(series[0])):
        ax1.scatter(series[dsix][i], series[dsiy][i], series[dsiz][i], 
                marker='o', s=m1*5, label=names[i], color=colors[i], alpha=0.6)
        ax1.scatter(teda.c[i].mean[0], teda.c[i].mean[1], teda.c[i].mean[2], 
                color='black', marker='x', s=m2*10)
        
        center = teda.c[i].mean
        ax1.scatter(center[0], center[1], center[2], 
                    color='black', marker='x', s=m2*10)

        # 3. Cálculo e Plot da Esfera (Volume ocupado)
        if radii is not None :
        #if radii is not None and i !=0:
            r = radii[i]
            
            # Criar a malha esférica (u: longitude, v: latitude)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            
            # Coordenadas paramétricas da esfera
            x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

            # Plotar a superfície transparente
            ax1.plot_surface(x, y, z, color=colors[i], alpha=0.1, 
                            linewidth=0, antialiased=True)
            
            # Opcional: Adicionar um aramado (wireframe) sutil para dar volume
            ax1.plot_wireframe(x, y, z, color=colors[i], alpha=0.1, linewidth=0.5)
    
    ax1.scatter([], [], [], color='black', marker='x', s=m2*10, label='G centroid')

    ax1.set_xlabel(f"DSI-{dsix+1}", fontsize=flbl)
    ax1.set_ylabel(f"DSI-{dsiy+1}", fontsize=flbl)
    ax1.set_ylabel(f"DSI-{dsiz+1}", fontsize=flbl)
    ax1.legend(fontsize=flgnd, framealpha=0.85, bbox_to_anchor=anchor, ncol=ncol)
    ax1.grid()
    for j,ax in enumerate(axes):
        #L,U = np.inf,-np.inf
        for i in range(len(series[0])):
            ax.plot(series[-1][i], series[j][i], linestyle=' ', marker='o',
                    markersize=m1, label=None, color=colors[i])
            
            if L>np.min(series[j][i]):
                L = np.min(series[j][i]) -0.15*np.min(series[j][i])
            else: L = L
            if U<np.max(series[j][i]):
                U = np.max(series[j][i])+ 0.05*np.max(series[j][i])
            else: U = U

            #yticks = list(np.linspace(L, U, 5))
            #ax.set_ylim(L, U)
            #ax.set_yticks(sorted(yticks))

            ax.set_ylabel(f"DSI-{j+1} Value", fontsize=flbl)
            ax.set_xlabel(f"Cycle", fontsize=flbl)
            #ax.set_yticks(sorted(yticks))
            
            if j< len(axes)-1: 
                ax.set_xlabel('')
                ax.tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)
            ax.grid()
        #for cp in ChangePoints:
        #    ax.plot([cp[0],cp[0]],[L,U], linestyle='--')
        #    ax.plot([cp[1],cp[1]],[L,U], linestyle='--')

    fig.suptitle(title,y=0.95)
    if mergeID: fig.text(0.5, 0.1, merged_track, fontsize=10, ha='center', va='bottom', transform=fig.transFigure)
    if out is not None and title is not None:
        plt.savefig(out, dpi=500, transparent=False)
    
    plt.show()



def plot_DSI_plotly(teda, start=None, ftrs=None, title=None, 
                    dsix=0, dsiy=1, dsiz=2, opacity_sphere=0.15):
    
    # --- 1. Processamento de Dados (Lógica Original) ---
    qtd = len(teda.c)
    DSIs = teda.DSIs[start:ftrs]
    n_dsi = len(DSIs)
    
    # Reconstruindo a estrutura de séries (conforme seu código original)
    series = [[[] for _ in range(qtd)] for i in range(n_dsi + 1)]
    clouds_ID = [cloud.ID for cloud in teda.c]
    
    # Lógica de ativação e merge simplificada para o Plotly
    for i, cloud_act in enumerate(teda.cloud_activation):
        for j, cloud_ID in enumerate(clouds_ID):
            if np.flip(cloud_act)[-1] == cloud_ID:
                for k in range(n_dsi):
                    series[k][j].append(DSIs[k][i])
                series[-1][j].append(i) # Ciclos/Tempo

    # --- 2. Configuração do Layout de Subplots ---
    # Para muitos DSIs, o espaçamento vertical deve ser pequeno
    v_spacing = min(0.05, 1.25 / n_dsi)
    
    fig = make_subplots(
        rows=n_dsi, cols=2,
        column_widths=[0.8, 0.2],
        specs=[[{'type': 'scene', 'rowspan': n_dsi}, {'type': 'xy'}] if i == 0 else [None, {'type': 'xy'}] 
               for i in range(n_dsi)],
        subplot_titles=[None] + [f"DSI-{i+1} Evolution" for i in range(n_dsi)],
        vertical_spacing=v_spacing,
        horizontal_spacing=0.01
    )

    # Cores
    import plotly.express as px
    colors = px.colors.qualitative.T10

    # --- 3. Iteração por Nuvem (Granule) ---
    for i in range(qtd):
        name = f"G{clouds_ID[i]}"
        color = colors[i % len(colors)]
        center = teda.c[i].mean
        r = (teda.c[i].R) if hasattr(teda.c[i], 'R') else 0.1
        r2 = (teda.c[i].Dmax) if hasattr(teda.c[i], 'R') else 0.1
        #print(r,r2)
        # A. Gráfico 3D (Esquerda) - Pontos
        fig.add_trace(go.Scatter3d(
            x=series[dsix][i], y=series[dsiy][i], z=series[dsiz][i],
            mode='markers', name=name,
            marker=dict(size=3, color=color, opacity=0.7),
            legendgroup=name
        ), row=1, col=1)

        # B. Gráfico 3D (Esquerda) - Esfera de Volume
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = r * np.cos(u) * np.sin(v) + center[0]
        ys = r * np.sin(u) * np.sin(v) + center[1]
        zs = r * np.cos(v) + center[2]

        xs2 = r2 * np.cos(u) * np.sin(v) + center[0]
        ys2 = r2 * np.sin(u) * np.sin(v) + center[1]
        zs2 = r2 * np.cos(v) + center[2]

        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=opacity_sphere,
            colorscale=[[0, color], [1, color]],
            showscale=False, name=f"{name} Volume",
            legendgroup=name, showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

        '''fig.add_trace(go.Surface(
            x=xs2, y=ys2, z=zs2,
            opacity=opacity_sphere/2,
            colorscale=[[0, color], [1, color]],
            showscale=False, name=f"{name} Volume",
            legendgroup=name, showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)'''

        # C. Gráfico 3D (Esquerda) - Centroide
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers', marker=dict(size=5, symbol='x', color='black'),
            showlegend=False, legendgroup=name, name=f"Centroid {name}"
        ), row=1, col=1)

        # D. Gráficos 2D (Direita) - Evolução Temporal
        for j in range(n_dsi):
            fig.add_trace(go.Scatter(
                x=series[-1][i], y=series[j][i],
                mode='markers', name=name,
                marker=dict(size=4, color=color),
                showlegend=False, legendgroup=name
            ), row=j+1, col=2)

    # --- 4. Ajustes de Layout e Eixos ---
    fig.update_layout(
        title_text=title,
        height=600,
        width=1500,
        showlegend=True,
        template="plotly_white",
        scene=dict(
            xaxis_title=f"DSI-{dsix+1}",
            yaxis_title=f"DSI-{dsiy+1}",
            zaxis_title=f"DSI-{dsiz+1}",
            aspectmode='cube'
        ),
        margin=dict(l=20, r=20, t=80, b=100)
    )

    # Adicionando anotação de Merged Granules (equivalente ao fig.text do Matplotlib)
    merged_text = "Nenhum Merge Detectado" # Exemplo, pode usar sua lógica original
    fig.add_annotation(
        text=merged_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.05, showarrow=False,
        font=dict(size=10, color="gray")
    )

    fig.show()


def plot_DSI2(teda, w=15, h=4,ftrs=None, out=None, title=None,
             mergeID=True,lnwdth=0.75, ftcks=7, flbl=9, fttl=8.5,
             dsix=0,dsiy=1,
             flgnd=7, anchor=None, m1=3, m2=4, m3=1, ncol=1):
    if ftrs is None: ftrs = 2
    cmap = mpl.colormaps.get_cmap('tab20').resampled(len(teda.c))
    colors = [cmap(i) for i in range(len(teda.c))]

    #cmap = mpl.colormaps.get_cmap('tab20').resampled(10)
    #colors = [cmap(i) for i in range(10)]

    fig = plt.figure(figsize=(w, h))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12,
                    wspace=0.12, hspace=0.15)
    gs = fig.add_gridspec(nrows=ftrs, ncols=2, width_ratios=[1, 2.5],
    wspace=0.125, hspace=0.15)
    ax1 = fig.add_subplot(gs[:, 0]) 
    axes = [fig.add_subplot(gs[i, 1]) for i in range((ftrs))]

    for i,cloud in enumerate(teda.c):
        x = np.array(cloud.x).T[-2]
        y = np.array(cloud.x).T[-1]
        if len(cloud.track) == 1: lbl = f'G{cloud.ID}'
        else: lbl = f'G{cloud.ID}: {cloud.track[1:]}'
            
        ax1.plot(x,y, linestyle=' ', linewidth=lnwdth, marker='o'
                          , markersize=m1, label=lbl, color=colors[i])
        ax1.plot(cloud.mean[-2], cloud.mean[-1], linestyle=' '
                          , color='black', marker='x', markersize=m2)
    
    ax1.plot(cloud.mean[-2], cloud.mean[-1], linestyle=' '
                          , color='black', marker='x', markersize=m2, label = 'G centroid')

    ax1.set_xlabel(f"DSI-{dsix+1}", fontsize=flbl)
    ax1.set_ylabel(f"DSI-{dsiy+1}", fontsize=flbl)
    ax1.legend(fontsize=flgnd, framealpha=0.85, bbox_to_anchor=anchor, ncol=ncol)
    ax1.grid()
    for j,ax in enumerate(axes):
        L,U = np.inf,-np.inf
        for i, cloud in enumerate(teda.c):
            y = np.array(cloud.x).T[-ftrs+j]
            ax.plot(cloud.t, y, linestyle=' ', marker='o',
                    markersize=m1, label=None, color=colors[i])
            
            if L>np.min(y): L = np.min(y) - 1.15*np.min(y)
            if U<np.max(y): U = np.max(y) + 0.15*np.max(y)

        yticks = np.round(np.linspace(0, 1, 6), 2).tolist()[1:] + [0]
        ax.set_yticks(sorted(yticks))
        #ax.set_ylim(L-0.05*U, U+0.05*U)
        ax.set_ylabel(f"DSI-{j+1} Value", fontsize=flbl)
        ax.set_xlabel(f"Cycle", fontsize=flbl)
        
        if j< len(axes)-1: 
            ax.set_xlabel('')
            ax.tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)
        ax.grid()


    fig.suptitle(title,y=0.95)
    if out is not None and title is not None:
        plt.savefig(out, dpi=500, transparent=False)
    
    plt.show()
    
def plot_DSI_3D2(teda, w=15, h=4,ftrs=None, out=None, title=None,
             mergeID=True,lnwdth=0.75, ftcks=7, flbl=9, fttl=8.5,
             dsix=0,dsiy=1,dsiz=2,
             flgnd=7, anchor=None, m1=3, m2=4, m3=1, ncol=1):
    if ftrs is None: ftrs = 2
    #cmap = mpl.colormaps.get_cmap('tab20').resampled(len(teda.c))
    #colors = [cmap(i) for i in range(len(teda.c))]

    cmap = mpl.colormaps.get_cmap('tab20').resampled(10)
    colors = [cmap(i) for i in range(10)]
    radii = [cloud.R for cloud in teda.c]

    fig = plt.figure(figsize=(w, h))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12,
                    wspace=0.12, hspace=0.15)
    gs = fig.add_gridspec(nrows=ftrs, ncols=2, width_ratios=[1, 2.5],
    wspace=0.125, hspace=0.15)
    ax1 = fig.add_subplot(gs[:, 0], projection='3d') 
    axes = [fig.add_subplot(gs[i, 1]) for i in range((ftrs))]

    for i,cloud in enumerate(teda.c):
        x = np.array(cloud.x).T[-3][5:]
        y = np.array(cloud.x).T[-2][5:]
        z = np.array(cloud.x).T[-1][5:]
        
        if len(cloud.track) == 1: lbl = f'G{cloud.ID}'
        else: lbl = f'G{cloud.ID}: {cloud.track[1:]}'
        
        ax1.scatter(x, y, z, 
                marker='o', s=m1*5, label=lbl, color=colors[i], alpha=0.4)
        ax1.plot(cloud.mean[-3], cloud.mean[-2], cloud.mean[-1], linestyle=' '
                          , color='black', marker='x', markersize=m2)
        
        center = cloud.mean
        ax1.scatter(center[-3], center[-2], center[-1], 
                    color='black', marker='x', s=m2*10)
        
        if radii is not None :
            r = radii[i]
            
            # Criar a malha esférica (u: longitude, v: latitude)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            
            # Coordenadas paramétricas da esfera
            x = r * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = r * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

            # Plotar a superfície transparente
            ax1.plot_surface(x, y, z, color=colors[i], alpha=0.1, 
                            linewidth=0, antialiased=True)
            
            ax1.plot_wireframe(x, y, z, color=colors[i], alpha=0.5, linewidth=0.5)

    
    ax1.scatter([], [], [], color='black', marker='x', s=m2*10, label='G centroid')
    
    ax1.set_xlabel(f"DSI-{dsix+1}", fontsize=flbl)
    ax1.set_ylabel(f"DSI-{dsiy+1}", fontsize=flbl)
    ax1.set_zlabel(f"DSI-{dsiz+1}", fontsize=flbl)
    ax1.legend(fontsize=flgnd, framealpha=0.85, bbox_to_anchor=anchor, ncol=ncol)
    ax1.grid()
    for j,ax in enumerate(axes):
        L,U = np.inf,-np.inf
        for i, cloud in enumerate(teda.c):
            y = np.array(cloud.x).T[-ftrs+j]
            ax.plot(cloud.t, y, linestyle=' ', marker='o',
                    markersize=m1, label=None, color=colors[i])
            
            if L>np.min(y): L = np.min(y) - 1.15*np.min(y)
            if U<np.max(y): U = np.max(y) + 0.15*np.max(y)

        yticks = np.round(np.linspace(0, 1, 6), 2).tolist()[1:] + [0]
        ax.set_yticks(sorted(yticks))
        ax.set_ylim(L-0.05*U, U+0.05*U)
        ax.set_ylabel(f"DSI-{j+1} Value", fontsize=flbl)
        ax.set_xlabel(f"Cycle", fontsize=flbl)
        
        if j< len(axes)-1: 
            ax.set_xlabel('')
            ax.tick_params(labelleft=False,labelsize=0.1,axis='x',colors='1',)
        ax.grid()


    fig.suptitle(title,y=0.95)
    if out is not None and title is not None:
        plt.savefig(out, dpi=500, transparent=False)
    
    plt.show()
