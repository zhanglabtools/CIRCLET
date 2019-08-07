# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:33:07 2018

@author: ysye
"""


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
from scipy.spatial import distance

#os.chdir('E:/Users/yusen/Project/Project3/Python code/CICRLET_package/src/CIRCLET')
from . import CIRCLET_DEFINE
from . import CIRCLET_CORE

bcolors_3=['#EF4F50','#587FBF','#CCCCCC']       
bcolors_6=['#587FBF','#3FA667','#EF4F50','#FFAAA3','#414C50','#D3D3D3']                    
bcolors_12=['#CC1B62','#FBBC00','#0E8934','#AC1120','#EA7B00','#007AB7',
            '#9A35B4','#804E1F' ,'#BEAB81','#D32414','#75AB09','#004084']


def change_index(passed_qc_sc_DF_cond,soft_add,software,UBI=['1CDU', '1CD_G1', '1CD_eS', '1CD_mS', '1CD_lS_G2']):
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
    elif software=='multi-metric':
        passed_qc_sc_DF=pd.read_table(soft_add,header=0,index_col=0)
        phenotime=passed_qc_sc_DF[['ord','cond']]
        ordIndex=phenotime.sort_values(by='ord')
    cond_order=[cond for cond in ordIndex['cond'] if cond in UBI]
    
    #generate penalty table
    penal_table=np.ones((len(UBI),len(UBI)))
    
    for loc in range(len(UBI)):
        penal_table[loc,loc]=0
        #if loc==0:
        #    penal_table[loc,1]=0
        #    penal_table[loc,2]=0
    penal_table=(np.triu(penal_table)+np.triu(penal_table).T)
    
    penalty_sum=0
    sc_number=len(cond_order)
    for k,cond in enumerate(cond_order):
        phase1=UBI.index(cond)
        phase2=UBI.index(cond_order[(k+1)%sc_number])
        penalty_sum+=penal_table[phase1,phase2]
    change_score=1-(penalty_sum-4)/(sc_number-4)
    return change_score 



def evaluate_continue_change(con_distri_features,soft_add,software):
    #roc_auc_DF=evaluation_ranks(passed_qc_sc_DF_cond,soft_add,software,UBI=UBIs[1:5])
    #plot_evaluate_heat(passed_qc_sc_DF_RO,soft_add,con_distri_features,software,UBIs)
    if software=='multi-metric':
        passed_qc_sc_DF=pd.read_table(soft_add,header=0,index_col=0)
        phenotime=passed_qc_sc_DF[['ord']]
    elif (software=='wishbone') | (software=='CIRCLET'):
        phenotime=pd.read_table(soft_add,header=None,index_col=0)
    phenotime.columns=['Pseudotime']
    ordIndex=phenotime.sort_values(by='Pseudotime')
    old_sc_name=ordIndex.index[-1]
    sc_name=ordIndex.index[0]
    corr_list=list()
    for sc_name in ordIndex.index:
        x=con_distri_features.loc[old_sc_name]
        y=con_distri_features.loc[sc_name]
        old_sc_name=sc_name
        #temp=stats.pearsonr(x,y)[0]
        #temp=distance.cosine(x,y)
        #temp=np.abs(distance.cosine(x,y)-1)
        temp=np.abs(distance.correlation(x,y)-1)
        corr_list.append(temp)
    evaluation_value=np.mean(corr_list)
    #print(evaluation_value)
    return evaluation_value


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


#soft_con_distri_Res_add=soft_add
def evaluation_ranks(passed_qc_sc_DF_cond,soft_con_distri_Res_add,software,UBI,key='not'):
    """
    Calculate the AUC curve values according to the order of the rankings 
    between the two UBI pairs to obtain the distribution of AUC values.
    """
    #UsingBatchIDs=['1CDU', '1CDX1', '1CDX2', '1CDX3', '1CDX4', '1CDES']
    #UBIs=['1CDU', '1CD_G1', '1CD_eS', '1CD_mS', '1CD_lS_G2', 'NoSort']
 
    if software=='multi-metric':
        passed_qc_sc_DF=pd.read_table(soft_con_distri_Res_add,header=0,index_col=0)
        MM_phenotime=passed_qc_sc_DF[['ord','cond']]
        MM_phenotime.columns=['Pseudotime','cond']
        ordIndex_soft=MM_phenotime.sort_values(by='Pseudotime')
    elif (software=='wishbone') | (software=='CIRCLET'):
        wishbone_phenotime=pd.read_table(soft_con_distri_Res_add,header=None,index_col=0)
        wishbone_phenotime.columns=['Pseudotime']
        
        wishbone_phenotime['cond']=passed_qc_sc_DF_cond[wishbone_phenotime.index].values
        ordIndex_soft=wishbone_phenotime.sort_values(by='Pseudotime')
    
    #fig,ax=plt.subplots()
    roc_auc_DF=list()
    #k=1
    for k, UB1 in enumerate(UBI):
        UB1=UBI[k]
        UB2=UBI[(k+1)%len(UBI)]
        
        Rank_list=ordIndex_soft.loc[(ordIndex_soft['cond']==UB1) | (ordIndex_soft['cond']==UB2)]
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
    roc_auc_DF=pd.DataFrame(roc_auc_DF,index=UBI)
    return roc_auc_DF



def plot_evaluate_heat(passed_qc_sc_DF_RO,soft_con_distri_Res_add,software,UBI):
    """
    plot evaluate heatmap
    """
    ordIndex_Nature=passed_qc_sc_DF_RO['ord']
    if software=='multi-metric':
        #passed_qc_sc_DF=pd.read_table(soft_con_distri_Res_add,header=0,index_col=0)
        #ordIndex_soft=(passed_qc_sc_DF['ord'].T.values-1)
        cycle_cells_v2=np.zeros(len(ordIndex_Nature),dtype=int)
        for i,order in enumerate(ordIndex_Nature):
            cycle_cells_v2[int(order)-1]=i
        ordIndex_soft=cycle_cells_v2
    elif (software=='wishbone') | (software=='CIRCLET'):
        wishbone_phenotime=pd.read_table(soft_con_distri_Res_add,header=None)
        wishbone_phenotime.columns=['cellnames','Pseudotime']
        ordIndex_soft=wishbone_phenotime.sort_values(by='Pseudotime').index    
    
    Fluo_comp=np.zeros((len(UBI),len(ordIndex_soft)))
    
    for j,rank in enumerate(ordIndex_soft):
        sc_cell=passed_qc_sc_DF_RO.index[rank]
        i=UBI.index(passed_qc_sc_DF_RO.loc[sc_cell,'cond'])
        #i=UsingBatchIDs.index(sc_cell.split("_")[0])
        Fluo_comp[i,j]=i+1
    
    fig,ax=plt.subplots(figsize=(4,2.5))
    #cmap=["#B4F8FF","#2FD4E6","#992F71","#E61898","#99862F","#E6C018","#0FFF01"]
    cmap=["#EFEFEF","#EF4F50","#3FA667","#587FBF","#FFAAA3"]
    #my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
    #my_cmap = ListedColormap(cmap.as_hex())
    my_cmap = ListedColormap(cmap)
    sns.heatmap(Fluo_comp,xticklabels=False, yticklabels=False,cmap=my_cmap,ax=ax)
    #plt.title(soft_con_distri_Res_add.split('/')[-1].rstrip('.txt'))


#soft_add=Result_file
def evaluate_software(passed_qc_sc_DF_RO,data_type,soft_add,software,UBI,type_names,key='not'):
    #passed_qc_sc_DF_cond=passed_qc_sc_DF_RO['cond']
    roc_auc_DF=evaluation_ranks(data_type,soft_add,software,UBI,key)
    change_score=change_index(data_type,soft_add,software,UBI)
    
    plot_evaluate_heat(passed_qc_sc_DF_RO,soft_add,software,UBI)
    
    roc_auc_DF.index=[ 'AUC:'+phase+'-'+type_names[(m+1)%len(type_names)] for m,phase in enumerate(type_names)]
    
    CS=pd.Series(change_score,index=['LCS'])
    evaluation=pd.concat([roc_auc_DF,CS])
    return evaluation



def evaluate_natrue(passed_qc_sc_DF_RO,index,UBIs,UsingBatchIDs,key='not'):
    #passed_qc_sc_DF_cond=passed_qc_sc_DF_RO['cond']
    soft_add="./data_RO/passed_qc_sc_DF_RO.txt"
    software='multi-metric'
    Nature_evaluation=evaluate_software(passed_qc_sc_DF_RO,soft_add,software,UBIs,UsingBatchIDs,key)
    for value in Nature_evaluation.values:
        print("%.4f" % value)
    return Nature_evaluation


from sklearn.preprocessing import minmax_scale
#feature_files=sub_feature_files
def merge_features(feature_files,HiC_dir,index,Isminmax=True):
    #print(sub_feature_files,DS)
    temp_data=pd.DataFrame()
    #filename=sub_feature_files[0]
    for k,filename in enumerate(feature_files):
        feature_data=pd.read_table(HiC_dir+'/'+filename, sep='\t', header=0,index_col=0)
        if np.isnan(np.max(feature_data.values)):
            feature_data=feature_data.fillna(0)
        if Isminmax:
            fea_data= minmax_scale(feature_data,feature_range=(0.01,1),axis=0,copy=True)
            feature_data=pd.DataFrame(fea_data,columns=feature_data.columns,index=feature_data.index)
        else:
            feature_data=feature_data+0.01
        temp_data=pd.concat([temp_data,feature_data],axis=1)
    temp_data=temp_data.loc[index]
    if not os.path.exists('./temp_data'):
        os.makedirs('./temp_data')
    temp_data.to_csv('./temp_data/temp.txt',sep='\t')
    fileadd='./temp_data/temp.txt'
    return fileadd

def plot_tsne(scdata,data_type,index,UBI):
    values=scdata.tsne
    #values=pd.DataFrame(data[:,[1,2]])
    #values.index=scdata.tsne.index
    #values.columns=scdata.tsne.columns
    #read passed_qc_sc_DF
    #passed_qc_sc_DF=pd.read_table("./data/passed_qc_sc_DF.txt",header=0,index_col=0)
    #passed_qc_sc_DF_cond=passed_qc_sc_DF_RO['cond'][index]
    bcolors2=['#587FBF','#3FA667','#EF4F50','#FFAAA3','#414C50','#D3D3D3'] 
    #['#1F78B4', '#B2DF8A', '#33A02C', '#F89A99', '#E31A1C', '#FDBF6F']
    # color assigning
    #UBIs=['1CDU', '1CD_G1', '1CD_eS', '1CD_mS', '1CD_lS_G2', 'NoSort']
    #sns.palplot(sns.color_palette(bcolors))
    with plt.style.context(('seaborn-ticks')):
        colors=[bcolors2[UBI.index(cond)]for cond in passed_qc_sc_DF_cond]
        fig,ax=plt.subplots(figsize=(4,2.5))
        ax.scatter(values['x'],values['y'],c=colors,s=10)
        #plt.yticks([-30,0,30])
        ax.spines['right'].set_color('none')#右脊柱设为无色
        ax.spines['top'].set_color('none')#上脊柱设为无色
        plt.show()

    
#Nagano_dir='E:/Users/yusen/Project/Project3/Python code/CICRLET_package/src/CIRCLET/DATA/Nagano et al'
def evaluate_Nagano_study(Nagano_dir):
    """
    0)	Firstly, we evaluate the trajectory in Nagano et al. and get initial information of Hi-C dataset.
    
    param Nagano_dir: address about Nagano et al's trajectory
    
    return Nature_evaluation: four evaluation indexes' scores inferred by Nagano et al
    return index: cells’names with FACS lables in Nagano et al’s study
    return data_type: cells' class used
    return UBI: classes' names used
    return passed_qc_sc_DF_RO: Nagano et al's trajectory with FACS lables in Nagano et al’s study
    """
    All_UsingBatchIDs=['1CDU', '1CDX1', '1CDX2', '1CDX3', '1CDX4', '1CDES','1CDS1','1CDS2','2i','serum']
    UBIs=['1CDU', '1CD_G1', '1CD_eS', '1CD_mS', '1CD_lS_G2', 'NoSort']
    UsingBatchIDs=All_UsingBatchIDs[0:6]
    #bcolors=["#1F78B4","#B2DF8A","#33A02C","#F89A99","#E31A1C","#FDBF6F"]
    passed_qc_sc_DF=pd.read_table(Nagano_dir+"/passed_qc_sc_DF.txt",header=0,index_col=0)
    
    con_distri_features_add=Nagano_dir+'/CDD_100f_8.txt'
    con_distri_features=pd.read_table(con_distri_features_add,header=0,index_col=0)
    #remove noise cell
    X=con_distri_features
    #thr=np.percentile(X.values,5)
    data=X.iloc[401:1572,:]
    UBI=UBIs[1:5]
    #data=X
    #data=data.loc[(data>thr).apply(np.sum,axis=1)>=data.shape[1]/2,:]
    index=data.index
    
    passed_qc_sc_DF_RO=passed_qc_sc_DF.loc[index,:]
    order=passed_qc_sc_DF_RO['ord']
    passed_qc_sc_DF_RO['ord']=order.rank()
    passed_qc_sc_DF_RO.to_csv("./data_RO/passed_qc_sc_DF_RO.txt",sep='\t')
    
    data_type=passed_qc_sc_DF_RO['cond'][index]
    
    con_distri_features=con_distri_features.loc[index,:]
    #evaluate nature's paper
    Nature_evaluation=evaluate_natrue(passed_qc_sc_DF_RO,index,UBIs,UsingBatchIDs)
    return Nature_evaluation,index,data_type,UBI,passed_qc_sc_DF_RO



#HiC_dir='E:/Users/yusen/Project/Project3/Python code/CICRLET_package/src/CIRCLET/DATA/Hi-Cmaps'
def Get_SC_HiCmap_Features(HiC_dir,index):
    """
    1) merge features and get scdata for next step
    param: HiC_dir: address about features extracted by CICRLET
    param: index: cells’names with FACS lables in Nagano et al’s study
    return: scdata
    return: filename
    """
    #suggest use MCM、CDD、PCC
    feature_files=os.listdir(HiC_dir)
    Is=True
    fileadd=merge_features(feature_files,HiC_dir,index,Isminmax=Is)
    fflag_str=[file.rstrip('.txt') for file in feature_files]
    filename='+'.join(fflag_str)
    print(fileadd,filename)
    scdata=CIRCLET.SCData.from_csv(os.path.expanduser(fileadd),
                    data_type='sc-seq',normalize=True)
    
    return scdata,filename
    

def Reduce_dimension_HiCmap(scdata,data_type,passed_qc_sc_DF_RO,index,UBI):
    """
    2)	Reducing feature dimensions:
    param: scdata
    param: data_type: cells' class used
    param: passed_qc_sc_DF_RO: Nagano et al's trajectory with FACS lables in Nagano et al’s study
    param: index:cells’names with FACS lables in Nagano et al’s study
    param: UBI:classes' names used
    
    return data: low dimension's data     
    """
    scdata.run_pca()
    #fig, ax = scdata.plot_pca_variance_explained(ylim=(0, 5), n_components=10)
    NO_CMPNTS=5
    scdata.run_tsne(n_components=NO_CMPNTS, perplexity=30)
    #fig, ax = scdata.plot_tsne()
    plot_tsne(scdata,data_type,index,UBI)
    # Run diffusion maps
    scdata.run_diffusion_map()
    #plot diffusion maps based on tsne
    fig, ax = scdata.plot_diffusion_components()
    scdata.plot_diffusion_eigen_vectors()
    components_list=[0,1,2,3]
    data=scdata.diffusion_eigenvectors.loc[:, components_list].values
    return data

def Getting_trajectory_HiCmap(HiC_dir,scdata,data,filename):
    """
    3)	Getting the circular trajectory from Hi-C maps:
    param: HiC_dir: address about features extracted by CICRLET
    param: scdata
    param: data: low dimension's data  
    param: filename
    
    return Result_file: trajectory of Hi-C maps inferred by CIRCLET
    """
    import scipy.sparse.csgraph
    circ = CIRCLET.Wishbone(scdata)
    #random choose starting cell
    start_cell='1CDX2_356'
    s = np.where(scdata.diffusion_eigenvectors.index == start_cell)[0][0]
    #branch=True
    res = CIRCLET_CORE.CIRCLET(
            data,
            s=s, k=15,l=15, num_waypoints=150, branch=True)
    trajectory=res['Trajectory']
    branches=res['Branches']
    if branches is not None:
        trajectory=-trajectory*(branches*2-5)
    trajectory = (trajectory - np.min(trajectory)) / (np.max(trajectory) - np.min(trajectory))
    circ.trajectory = pd.Series(trajectory, index=scdata.data.index) 
    
    circ.waypoints = list(scdata.data.index[res['Waypoints']])
    
    #Set branch colors
    branch_cs=['#EF4F50','#587FBF']
    if np.all(branches!=None):
        #wb.branch_colors = dict( zip([2, 3], qualitative_colors(2)))
        circ.branch_colors = dict( zip([2, 3], branch_cs))
    #print(circ.trajectory)
    #circ.plot_wishbone_on_tsne()
    Result_add=HiC_dir+'/result_files'
    if not os.path.exists(Result_add):
        os.makedirs(Result_add)
    Result_file=Result_add+"/CIRC_"+ filename +".txt"
    circ.trajectory.to_csv(Result_file, sep = "\t")
    return Result_file

#type_names=['G1','ES','MS','G2']
def evaluate_result_RNAseq(passed_qc_sc_DF_RO,data_type,Result_file,UBI,type_names,software='CIRCLET'):
    """
    4)	Evaluating the results:
    param: passed_qc_sc_DF_RO: Nagano et al's trajectory with FACS lables in Nagano et al’s study
    param: data_type: cells' class used
    param: Result_file: trajectory of Hi-C maps inferred by CIRCLET
    param: UBI:classes' names used
    param: type_names: cell cycle of cells's class
    param: software
    
    return: five evaluation indexes' scores inferred by CIRCLET
    """
    #evaluating the results
    evaluation=evaluate_software(passed_qc_sc_DF_RO,data_type,Result_file,software,UBI,type_names)
    #print(np.sum(evaluation))
    return evaluation


