# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:33:11 2018

@author: ysye
"""
#run CIRCLET for RNA-seq dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import minmax_scale
from math import log
from sklearn.metrics import roc_curve, auc
from scipy import stats
from matplotlib.colors import ListedColormap


#os.chdir('E:/Users/yusen/Project/Project3/Python code/CICRLET_package/src/CIRCLET')
from . import CIRCLET_DEFINE
from . import CIRCLET_CORE

bcolors_3=['#EF4F50','#587FBF','#CCCCCC']        
bcolors_6=['#587FBF','#3FA667','#EF4F50','#FFAAA3','#414C50','#D3D3D3']                   
bcolors_12=['#CC1B62','#FBBC00','#0E8934','#AC1120','#EA7B00','#007AB7',
            '#9A35B4','#804E1F' ,'#BEAB81','#D32414','#75AB09','#004084']



def Rnaseq_feature_selection(Rnaseq_data,Rnaseq_dir):
    """
    focus on cell cycle annoated genes
    """
    #gene_files_add='./result_files/CellCycleGenes'
    Type='CellCycle'
    Marker_genes_add=Rnaseq_dir+'/GO_term_summary_'+Type+'.xlsx'
    MCellCyclegenes=pd.read_excel(Marker_genes_add,"Annotation",header=0,index_col=None)
    CellCyclegenes=MCellCyclegenes.loc[MCellCyclegenes['Mouse Gene Symbol']!='None',['Mouse Gene Symbol']]
    annotate_genes=set(Rnaseq_data.columns).intersection(set(CellCyclegenes['Mouse Gene Symbol']))
    Rnaseq_data_fs=Rnaseq_data[list(annotate_genes)]
    return Rnaseq_data_fs
    
  
def change_index_Rnaseq(passed_qc_sc_DF_cond,soft_add,software,UBIs):
    """
    which measures how frequent an
    experimentally determined single cell labels changes along the time-series.
    """
    #read order of single cell
    if (software=='wishbone') | (software=='CIRCLET'):
        phenotime=pd.read_table(soft_add,header=None,index_col=0)
        phenotime.columns=['Pseudotime']
        phenotime['cond']=passed_qc_sc_DF_cond
        ordIndex=phenotime.sort_values(by='Pseudotime')
        #cond_order=[cond for cond in ordIndex['cond'] if cond in UBI]
    cond_order=[cond for cond in ordIndex['cond'] if cond in UBIs]
    #generate penalty table
    penal_table=np.ones((len(UBIs),len(UBIs)))
    for loc in range(len(UBIs)):
        penal_table[loc,loc]=0
        #if loc==0:
        #    penal_table[loc,1]=0
        #    penal_table[loc,2]=0
    penal_table=(np.triu(penal_table)+np.triu(penal_table).T)
    
    penalty_sum=0
    sc_number=len(cond_order)
    for k,cond in enumerate(cond_order):
        phase1=UBIs.index(cond)
        phase2=UBIs.index(cond_order[(k+1)%sc_number])
        penalty_sum+=penal_table[phase1,phase2]
    change_score=1-(penalty_sum-4)/(sc_number-4)
    return change_score     
    

def computing_AUC(Rank_list):
    """
    Compulating AUC
    """
    y_true=Rank_list['bench']
    y_score=np.max(Rank_list['Pseudotime'])-Rank_list['Pseudotime']
    fpr,tpr,threshold = roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)
    if roc_auc<0.5:
        roc_auc=1-roc_auc
    #plt.plot(fpr,tpr)
    return roc_auc

    


def evaluation_ranks_Rnaseq(passed_qc_sc_DF_cond,soft_con_distri_Res_add,software,UBIs,key='not'):
    """
    Calculate the AUC curve values according to the order of the rankings 
    between the two UBI pairs to obtain the distribution of AUC values.
    """
    if (software=='wishbone') | (software=='CIRCLET'):
        wishbone_phenotime=pd.read_table(soft_con_distri_Res_add,header=None,index_col=0)
        wishbone_phenotime.columns=['Pseudotime']
        wishbone_phenotime['cond']=passed_qc_sc_DF_cond
        ordIndex_soft=wishbone_phenotime.sort_values(by='Pseudotime')
    
    #fig,ax=plt.subplots()
    roc_auc_DF=list()
    #k=0
    for k, UB1 in enumerate(UBIs):
        UB1=UBIs[k]
        UB2=UBIs[(k+1)%len(UBIs)]
        Rank_list=ordIndex_soft.loc[(ordIndex_soft['cond']==UB1) | (ordIndex_soft['cond']==UB2)]
        #X=Rank_list.ix[Rank_list['cond']==UB1,'Pseudotime']
        #Y=Rank_list.ix[Rank_list['cond']==UB2,'Pseudotime']
        #if k==3:
        #    Rank_list['bench']=0
        #    Rank_list.loc[Rank_list['cond']==UB2,'bench']=1
        #else:
        Rank_list['bench']=0
        Rank_list.loc[Rank_list['cond']==UB1,'bench']=1
        cell1=Rank_list.index[0:10]
        cell2=ordIndex_soft.index[0:10]
        cell3=Rank_list.index[-10:]
        cell4=ordIndex_soft.index[-10:]
        if ((len(cell1.intersection(cell2))>3) & (len(cell3.intersection(cell4))>3) & (key=='acc')):
            roc_auc=0
            cell_UB1=(Rank_list['cond']==UB1).index
            for cell in cell_UB1:
                #cell=Rank_list.index[k]
                X=Rank_list.loc[ cell,'Pseudotime']
                Rank_list['Pseudotime']=(Rank_list['Pseudotime']-X+1.0)%1
                new_roc_auc=computing_AUC(Rank_list)
                roc_auc=np.max([roc_auc,new_roc_auc])
                print(roc_auc)
        else:
            roc_auc=computing_AUC(Rank_list)
        roc_auc_DF.append(roc_auc)
    
    roc_auc_DF=pd.DataFrame(roc_auc_DF,index=UBIs)
    return roc_auc_DF
    
#soft_con_distri_Res_add=soft_add
def plot_evaluate_heat_Rnaseq(data_types,soft_con_distri_Res_add,software,UBIs):
    """
    plot evaluate heatmap
    """
    if (software=='wishbone') | (software=='CIRCLET'):
        wishbone_phenotime=pd.read_table(soft_con_distri_Res_add,header=None)
        wishbone_phenotime.columns=['cellnames','Pseudotime']
        ordIndex_soft=wishbone_phenotime.sort_values(by='Pseudotime').index    
    
    #compare with Fluorescence-activated cell sorting
    Fluo_comp=np.zeros((len(UBIs),len(ordIndex_soft)))
    #j=0; rank=ordIndex_soft[j]
    for j,rank in enumerate(ordIndex_soft):
        i=UBIs.index(data_types[rank])
        #i=UsingBatchIDs.index(sc_cell.split("_")[0])
        Fluo_comp[i,j]=i+1
    fig,ax=plt.subplots(figsize=(4,2.5))
    #cmap=["#B4F8FF","#2FD4E6","#992F71","#E61898","#99862F","#E6C018","#0FFF01"]
    cmap=["#EFEFEF","#EF4F50","#3FA667","#587FBF"]
    #my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
    #my_cmap = ListedColormap(cmap.as_hex())
    my_cmap = ListedColormap(cmap)
    sns.heatmap(Fluo_comp,xticklabels=False, yticklabels=False,cmap=my_cmap,ax=ax)
    plt.title(soft_con_distri_Res_add.split('/')[-1].rstrip('.txt')) 
    



      
#software='wishbone'   
def evaluate_software_Rnaseq(data_types,soft_add,software,type_names,key='not'):
    UBIs=type_names
    
    roc_auc_DF=evaluation_ranks_Rnaseq(data_types,soft_add,software,UBIs,key='not')
    
    change_score=change_index_Rnaseq(data_types,soft_add,software,UBIs)

    #con_distri_features=(con_distri_features.T/np.sum(con_distri_features,axis=1)).T
    plot_evaluate_heat_Rnaseq(data_types,soft_add,software,UBIs)
    #for value in Entropys:
    #    print("%f" % value)
    CS=pd.Series(change_score,index=['LCS'])
    #CC=evaluate_continue_change(con_distri_features,soft_add,software)
    #CC=pd.Series( [CC], index=['continue_change'])
    #Nature_evaluation=pd.concat([Results,roc_auc_DF,CS])
    #evaluation=pd.concat([roc_auc_DF,CS,CC])
    roc_auc_DF.index=[ 'AUC:'+phase+'-'+type_names[(m+1)%len(type_names)] for m,phase in enumerate(type_names)]
    
    evaluation=pd.concat([roc_auc_DF,CS])
    return evaluation    

def plot_tsne(scdata,data_type):
    values=scdata.tsne
    bcolors_3=["#EF4F50","#3FA667","#587FBF"]           
    with plt.style.context(('seaborn-ticks')):
        colors=[bcolors_3[cond-1] for cond in data_type]
        fig,ax=plt.subplots(figsize=(4,2.5))
        ax.scatter(values['x'],values['y'],c=colors,s=10)
        ax.spines['right'].set_color('none')#右脊柱设为无色
        ax.spines['top'].set_color('none')#上脊柱设为无色
        #ax.plot(values.loc[start_cell,'x'],values.loc[start_cell,'y'],'kD')
        plt.show()

#Rnaseq_dir='E:/Users/yusen/Project/Project3/Python code/CICRLET_package/src/CIRCLET/DATA/MESC_RNA-seq'
def Get_SC_RNAseq_Features(Rnaseq_dir,filename='EMSC_RNA-seq',feature_selection=1):
    """
    1)	Extracting feature: This process only uses the part of CIRCLET, not including the step of Extracting feature. 
    Thus, we collected a single-cell RNA-seq dataset consisting of 182 cells for G1, S and G2/M phases and use a set of 
    959 annotated genes of cell cycle for analysis with variation above the background level in [2].
    
    param: Rnaseq_dir: inputing address of RNA-seq dataset
    param: filename
    param： feature_selection: if 1, only focusing on cell cycle annotated genes, else focusing on all valid genes.
    
    return: scdata
    return: data_tyeps: cells' class used
    """
    
    Rnaseq_add=Rnaseq_dir+'/'+filename+'.xlsx'
    Rnaseq_data=pd.read_excel(Rnaseq_add,"expression",header=0,index_col=0)
    Rnaseq_data.columns=[gene.lstrip("'").rstrip("'") for gene in Rnaseq_data.columns]
    data_type=Rnaseq_data.index
    Rnaseq_data.index=np.arange(1,len(data_type)+1)
    data_types_num=[59,58,65]
    type_names=['G1','S','G2/M']
    data_types=[type_names[ind-1] for ind in data_type]
    if not os.path.exists('./temp_data'):
        os.makedirs('./temp_data')
    #data=temp_data.loc[(temp_data>0).apply(np.sum,axis=1)>=temp_data.shape[1]/2,:]
    #temp_data=temp_data+0.0001
    Rnaseq_data_value= minmax_scale(Rnaseq_data,feature_range=(0.01,1),axis=0,copy=True)
    Rnaseq_data=pd.DataFrame(Rnaseq_data_value,columns=Rnaseq_data.columns,index=Rnaseq_data.index)
    #feature_selection=1
    if feature_selection==1:
        Rnaseq_data_fs=Rnaseq_feature_selection(Rnaseq_data,Rnaseq_dir)
        Rnaseq_data_fs.to_csv('./temp_data/temp.txt',sep='\t')
    else:
        Rnaseq_data.to_csv('./temp_data/temp.txt',sep='\t')
    fileadd='./temp_data/temp.txt'
    scdata=CIRCLET.SCData.from_csv(os.path.expanduser(fileadd),
                        data_type='sc-seq',normalize=True) 
    return scdata,data_types



def Reduce_dimension_RNAseq(scdata,data_type):
    
    """
    2)	Reducing feature dimensions:
    param: scdata
    param: data_type: cells' class used

    return data: low dimension's data    
    """
    scdata.run_pca()
    #fig, ax = scdata.plot_pca_variance_explained(ylim=(0, 5), n_components=10)
    NO_CMPNTS=5
    scdata.run_tsne(n_components=NO_CMPNTS, perplexity=30)
    #fig, ax = scdata.plot_tsne()
    plot_tsne(scdata,data_type)
    # Run diffusion maps
    scdata.run_diffusion_map()
    #plot diffusion maps based on tsne
    fig, ax = scdata.plot_diffusion_components()
    scdata.plot_diffusion_eigen_vectors()
    components_list=[0,1,2,3,4,5,6,7,8]
    data=scdata.diffusion_eigenvectors.loc[:, components_list].values
    return data

def Getting_trajectory_RNAseq(Rnaseq_dir,scdata,data,filename='MESC_RNA-seq'):
    """
    3)	Getting the circular trajectory from Hi-C maps:
    param: Rnaseq_dir: inputing address of RNA-seq dataset
    param: scdata
    param: data: low dimension's data  
    param: filename
    
    return Result_file: trajectory of RNAseq inferred by CIRCLET
    """
    
    import scipy.sparse.csgraph
    circ = CIRCLET.Wishbone(scdata)
    #random choose starting cell
    start_cell=121
    s = np.where(scdata.diffusion_eigenvectors.index == start_cell)[0][0]
    #branch=True
    res = CIRCLET_CORE.CIRCLET(
            data,
            s=s, k=3,l=3, num_waypoints=100, branch=True)

    trajectory=res['Trajectory']
    branches=res['Branches']
    trajectory = -(trajectory - np.min(trajectory)) / (np.max(trajectory) - np.min(trajectory))
    circ.trajectory = pd.Series(trajectory, index=scdata.data.index) 
    #branch = None
    if np.all(branches!=None):
        circ.branch = pd.Series([np.int(i) for i in branches], index=scdata.data.index)
    
    #wb.waypoints = list(scdata.data.index[res['Waypoints']])
    #Set branch colors
    branch_cs=['#EF4F50','#587FBF']
    if np.all(branches!=None):
        #wb.branch_colors = dict( zip([2, 3], qualitative_colors(2)))
        circ.branch_colors = dict( zip([2, 3], branch_cs))
    #print(circ.trajectory)
    #circ.plot_wishbone_on_tsne()
    
    
    Result_add=Rnaseq_dir+'/result_files'
    if not os.path.exists(Result_add):
        os.makedirs(Result_add)
    Result_file=Result_add+"/CIRC_"+ filename +".txt"
    circ.trajectory.to_csv(Result_file, sep = "\t")
    return Result_file

    
def evaluate_result_RNAseq(Result_file,data_types,type_names,software='CIRCLET'):
    """
    4)	Evaluating the results:
    param: Result_file: trajectory of Rnaseq inferred by CIRCLET
    param: data_type: cells' class used
    param: type_names: cell cycle of cells's class
    param: software
    
    return: four evaluation indexes' scores inferred by CIRCLET
    """
    #evaluating the results
    evaluation=evaluate_software_Rnaseq(data_types,Result_file,software,type_names,key='not')
    print(evaluation)
    #print(np.sum(evaluation))
    return evaluation






