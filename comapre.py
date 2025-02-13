import requests
import pandas as pd 
import numpy as np 
import warnings
import pickle
warnings.filterwarnings("ignore")
np.random.seed(1)
from shape import Shape,finder
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from dtaidistance import ed
from scipy.stats import ttest_1samp
import matplotlib.colors as mcolors
from scipy.stats import linregress
import matplotlib.lines as mlines
from sklearn.tree import DecisionTreeRegressor,plot_tree
from scipy import stats
from cycler import cycler

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)

############
### Data ###
############

# Get regions
country_list = pd.read_csv('Datasets/country_list.csv',index_col=0)
df_conf=pd.read_csv('Datasets/reg_coun.csv',index_col=0)
df_conf=pd.Series(df_conf.region)
replace_c = {'Cambodia (Kampuchea)': 'Cambodia','DR Congo (Zaire)':'Congo, DRC',
             'Ivory Coast':'Cote d\'Ivoire', 'Kingdom of eSwatini (Swaziland)':'Swaziland',
             'Myanmar (Burma)':'Myanmar','Russia (Soviet Union)':'Russia',
             'Serbia (Yugoslavia)':'Serbia','Madagascar (Malagasy)':'Madagascar',
             'Macedonia, FYR':'Macedonia','Vietnam (North Vietnam)':'Vietnam',
             'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe',
             'United States of America':'United States','Solomon Islands':'Solomon Is.',
             'Bosnia-Herzegovina':'Bosnia and Herzegovina'}
df_conf.rename(index=replace_c, inplace=True)
df_conf['Sao Tome and Principe']='Africa'

# Load the data from Views API (Uncomment to run)

# # Test data I
# df_list_preds={f"fatalities001_2022_00_t01/cm?page={i}":i for i in range(1,8)}

# df_all=pd.DataFrame()
# for i in range(len(df_list_preds)):
#     response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_preds.keys())[i]}')
#     json_data = response.json()
#     df=pd.DataFrame(json_data["data"])
#     df=df[['country_id','month_id','month','sc_cm_sb_main']]
#     # Reverse log
#     df['sc_cm_sb_main']=np.exp(df['sc_cm_sb_main'])-1
#     df_all = pd.concat([df_all, df])
#     df_all=df_all.reset_index(drop=True)
# cc_sort=df_all.country_id.unique()
# cc_sort.sort()
# df_preds_test_1 = df_all.pivot(index="month_id",columns='country_id', values='sc_cm_sb_main')
# df_preds_test_1.to_csv('Datasets/df_preds_test_1.csv')

# # Test data II
# df_list_preds={f"fatalities001_2023_00_t01/cm?page={i}":i for i in range(1,8)}

# df_all=pd.DataFrame()
# for i in range(len(df_list_preds)):
#     response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_preds.keys())[i]}')
#     json_data = response.json()
#     df=pd.DataFrame(json_data["data"])
#     df=df[['country_id','month_id','month','sc_cm_sb_main']]
#     # Reverse log
#     df['sc_cm_sb_main']=np.exp(df['sc_cm_sb_main'])-1
#     df_all = pd.concat([df_all, df])
#     df_all=df_all.reset_index(drop=True)
# cc_sort=df_all.country_id.unique()
# cc_sort.sort()
# df_preds_test_2 = df_all.pivot(index="month_id",columns='country_id', values='sc_cm_sb_main')
# df_preds_test_2.to_csv('Datasets/df_preds_test_2.csv')

# # Input data
# df_list_input={f"predictors_fatalities002_0000_00/cm?page={i}":i for i in range(1,78)}

# df_input_t=pd.DataFrame()
# for i in range(len(df_list_input)):
#     response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
#     json_data = response.json()
#     df=pd.DataFrame(json_data["data"])
#     df=df[["country_id","month_id","ucdp_ged_sb_best_sum"]]
#     df_input_t = pd.concat([df_input_t, df])
#     df_input_t=df_input_t.reset_index(drop=True)

# # Add labels to input data
# df_input_s = df_input_t[df_input_t['country_id'].isin(cc_sort)]
# df_input = df_input_s.pivot(index="month_id",columns='country_id',values='ucdp_ged_sb_best_sum')
# df_input.index = pd.date_range('01/01/1990',periods=len(df_input),freq='M')
# df_input = df_input.iloc[:408,:]
# df_input.columns = country_list['name']

# # Save
# df_input.to_csv('Datasets/df_input.csv')

# # Fix missing values
# df_tot_m = df_input.copy()
# df_tot_m.replace(0, np.nan, inplace=True)
# df_tot_m = df_tot_m.dropna(axis=1, how='all')
# df_tot_m = df_tot_m.fillna(0)
# df_tot_m.to_csv('Datasets/df_tot_m.csv')

# # Save
# df_obs_1 = df_input.iloc[-24:-12,:]        
# df_obs_1.to_csv('obs1.csv')   
# df_obs_2 = df_input.iloc[-12:,:]        
# df_obs_2.to_csv('Datasets/obs2.csv')   

# df_v_1 = df_preds_test_1.iloc[:12,:]
# df_v_1.to_csv('Datasets/views1.csv')  
# df_v_2 = df_preds_test_2.iloc[:12,:]
# df_v_2.to_csv('Datasets/views2.csv')  

# Load
df_tot_m=pd.read_csv('Datasets/df_tot_m.csv',index_col=0,parse_dates=True)
df_input=pd.read_csv('Datasets/df_input.csv',index_col=0,parse_dates=True)
df_obs_1=pd.read_csv('Datasets/obs1.csv',index_col=0,parse_dates=True)
df_obs_2=pd.read_csv('Datasets/obs2.csv',index_col=0,parse_dates=True)
df_v_1=pd.read_csv('Datasets/views1.csv',index_col=0,parse_dates=True)  
df_v_2=pd.read_csv('Datasets/views2.csv',index_col=0,parse_dates=True) 
df_preds_test_1=pd.read_csv('Datasets/df_preds_test_1.csv',index_col=0,parse_dates=True)
df_preds_test_2=pd.read_csv('Datasets/df_preds_test_2.csv',index_col=0,parse_dates=True)

####################
### Shape finder ###
####################

# Run the ShapeFinder model (Uncomment to run)

# Step 1. Get reference repository 

# # Test I
# h_train=10
# dict_m={i :[] for i in df_input.columns} 
# 
# df_input_sub=df_input.iloc[:-24]
# for coun in range(len(df_input_sub.columns)):
#     if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#         shape = Shape()
#         shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
#         find = finder(df_tot_m.iloc[:-24],shape)
#         find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
#         min_d_d=0.1
#         while len(find.sequences)<5:
#             min_d_d += 0.05
#             find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
#         dict_m[df_input.columns[coun]]=find.sequences
#     else :
#         pass

# For saving        
# with open('Results/test1.pkl', 'wb') as f:
#     pickle.dump(dict_m, f) 

# # Test II
# h_train=10
# dict_m={i :[] for i in df_input.columns}
#
# df_input_sub=df_input.iloc[:-12]
# for coun in range(len(df_input_sub.columns)):
#     if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#         shape = Shape()
#         shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
#         find = finder(df_tot_m.iloc[:-12],shape)
#         find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
#         min_d_d=0.1
#         while len(find.sequences)<5:
#             min_d_d += 0.05
#             find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
#         dict_m[df_input.columns[coun]]=find.sequences
#     else :
#         pass
    
# For saving        
# with open('Results/test2.pkl', 'wb') as f:
#     pickle.dump(dict_m, f)
      
# Step 2. Make predictions

# Test I
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
len_mat=[]      
df_sure=[]
pr_list=[]   
pr_main=[]
pr_scale=[]  

df_input_sub=df_input.iloc[:-24]
horizon=12
h_train=10

for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq_c = tot_seq.copy()
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pr_main.append(pr.max())
        len_mat.append(len(tot_seq))
        pr_scale.append(df_input_sub.iloc[-h_train:,coun].sum())
        testu = (df_input.iloc[-24:-12,coun] - df_input_sub.iloc[-h_train:,coun].min()) / (df_input_sub.iloc[-h_train:,coun].max() - df_input_sub.iloc[-h_train:,coun].min())
        tot_seq_c = pd.concat([pd.DataFrame(tot_seq_c),pd.DataFrame(testu.reset_index(drop=True)).T],axis=0)
        linkage_matrix_2 = linkage(tot_seq_c, method='ward')
        clusters_2 = fcluster(linkage_matrix_2, horizon/3, criterion='distance')
        if len(pd.Series(clusters_2).value_counts())==1:
            df_sure.append(tot_seq_c)
        if len(pd.Series(clusters).value_counts())==len(pd.Series(clusters_2).value_counts()):
            pr_list.append(pr[clusters_2[-1]])
        else:
            pr_list.append(-1)       
    else:
        pr_list.append(None)
         
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]

for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        plt.figure(figsize=(10, 6))
        plt.plot(preds, label='Shape finder', linestyle="dashed",color='black',linewidth=2)
        plt.plot(df_preds_test_1.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_1.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        plt.xticks([0,2,4,6,8,10],["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],size=25)
        plt.yticks(size=25)
        if df_input_sub.columns[coun]=="Niger":
            plt.savefig(f"out/compare_preds_{df_input_sub.columns[coun]}_2022.jpeg",dpi=400,bbox_inches="tight")
        plt.show()    
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
# Save        
df_sf_1 = pd.concat(pred_tot_pr,axis=1)
df_sf_1.columns=country_list['name']
df_sf_1.to_csv('Datasets/sf1.csv')  

# Get MSE 
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
mse_list_raw = err_sf_pr.copy()
mse_list=np.log((err_views+1)/(err_sf_pr+1))    
 
# Test II
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq_c = tot_seq.copy()
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pr_main.append(pr.max())
        len_mat.append(len(tot_seq))
        pr_scale.append(df_input_sub.iloc[-h_train:,coun].sum())
        testu = (df_input.iloc[-12:,coun] - df_input_sub.iloc[-h_train:,coun].min()) / (df_input_sub.iloc[-h_train:,coun].max() - df_input_sub.iloc[-h_train:,coun].min())
        tot_seq_c = pd.concat([pd.DataFrame(tot_seq_c),pd.DataFrame(testu.reset_index(drop=True)).T],axis=0)
        linkage_matrix_2 = linkage(tot_seq_c, method='ward')
        clusters_2 = fcluster(linkage_matrix_2, horizon/3, criterion='distance')
        if len(pd.Series(clusters_2).value_counts())==1:
            df_sure.append(tot_seq_c)
        if len(pd.Series(clusters).value_counts())==len(pd.Series(clusters_2).value_counts()):
            pr_list.append(pr[clusters_2[-1]])
        else:
            pr_list.append(-1)
    else:
        pr_list.append(None)
        
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12

for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        plt.figure(figsize=(10, 6))
        plt.plot(preds, label='Shape finder', linestyle="dashed",color='black',linewidth=2)
        plt.plot(df_preds_test_2.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_2.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        plt.xticks([0,2,4,6,8,10],["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],size=25)
        plt.yticks(size=25)
        if df_input_sub.columns[coun]=="Colombia" or df_input_sub.columns[coun]=="Central African Republic" or df_input_sub.columns[coun]=="Cameroon":
            plt.savefig(f"out/compare_preds_{df_input_sub.columns[coun]}_2023.jpeg",dpi=400,bbox_inches="tight")
        plt.show()    
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  

# Average proportion of cases assigned to the majortiy cluster
print(np.mean(cluster_dist))
print(np.std(cluster_dist))

# Save
df_sf_2= pd.concat(pred_tot_pr,axis=1)
df_sf_2.columns=country_list['name']
df_sf_2.to_csv('Datasets/sf2.csv')  

# Get MSE 
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
mse_list2=np.log((err_views+1)/(err_sf_pr+1))  

###############################################    
### Compound between Shape Finder and ViEWS ###
############################################### 

mse_list_tot=np.concatenate([mse_list,mse_list2],axis=0)
mse_list_raw = np.concatenate([mse_list_raw,err_sf_pr],axis=0)
pr_list=pd.Series(pr_list)
pr_main=pd.Series(pr_main)
pr_scale=pd.Series(pr_scale)
len_mat=pd.Series(len_mat)
df_tot_res = pd.DataFrame([mse_list_tot,pr_list]).T
df_tot_res[2]=[np.nan]*len(df_tot_res)
df_tot_res[2][pr_list == (-1)]='New'
df_tot_res[2][(pr_list >= 0) & (pr_list < 0.5)]='Low'
df_tot_res[2][(pr_list >= 0.5) & (pr_list < 1)]='High'
df_tot_res[2][(pr_list == 1)]='Sure'
ind_keep = pr_list.dropna().index 
pr_list=pr_list.dropna()
nan_percentage = ((pr_list==-1).sum() / len(pr_list)) * 100
zero_to_half = ((pr_list >= 0) & (pr_list < 0.5)).sum() / len(pr_list) * 100
half_to_02 = ((pr_list >= 0.5) & (pr_list < 1)).sum() / len(pr_list) * 100
ones = (pr_list == 1).sum() / len(pr_list) * 100
df_tot_res = df_tot_res.dropna()
df_sel = pd.concat([df_tot_res.iloc[:,0].reset_index(drop=True),pr_scale,pr_main,len_mat],axis=1)
df_sel = df_sel.dropna()
df_sel.columns=['log MSE','Scale','Main_Pr','N_Matches']
df_sel['Confidence']=df_sel.iloc[:,2]*np.log10(df_sel.iloc[:,3])
n_df_sel= df_sel[df_sel['log MSE'] <= 0]
p_df_sel= df_sel[df_sel['log MSE'] > 0]

plt.figure(figsize=(14, 10))
plt.scatter(n_df_sel.iloc[:,1], n_df_sel.iloc[:,2]* np.log10(n_df_sel.iloc[:,3]), color='darkgrey', s=70)
plt.scatter(p_df_sel.iloc[:,1], p_df_sel.iloc[:,2]* np.log10(p_df_sel.iloc[:,3]), color='black', s=70)
plt.xscale('log')
plt.xlabel('Severity, Number of fatalities',size=20)
plt.ylabel('Confidence, p*log(N)',size=20)
plt.hlines(0.6,0.6,1000000, color='black', linestyle='--', linewidth=2)
plt.vlines(10000,0,2, color='black', linestyle='--', linewidth=2)
plt.fill_betweenx(y=[0, 0.6], x1=0, x2=10000, color='grey', alpha=0.1, hatch='/',label='Confidence too low')
plt.fill_betweenx(y=[0.6, 2], x1=10000, x2=100000000, color='grey', alpha=0.1, hatch='\\',label='Severity too high')
plt.fill_betweenx(y=[0, 0.6], x1=10000, x2=100000000, color='grey', alpha=0.1, hatch='x')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0.6,200000)
plt.ylim(0.1,1.8)
ax = plt.gca()
xticks = ax.get_xticks()
yticks = ax.get_yticks()
ax.set_xticklabels([f'{tick:.0f}' if tick != 10000 else f'$\mathbf{{{int(tick):,}}}$' for tick in xticks],fontsize=20)
ax.set_yticklabels([f'{tick:.1f}' if tick != 0.6000000000000001 else f'$\mathbf{{{tick:.1f}}}$' for tick in yticks],fontsize=20)
plt.savefig("out/compound_select.jpeg",dpi=400,bbox_inches="tight")
plt.show()

df_sel_s = df_sel.sort_values(['Scale'])
df_sel_s=df_sel_s[df_sel_s['Confidence']>0.6] 
df_sel_s=df_sel_s[df_sel_s['Scale']<10000]
df_keep_1 = df_sel_s.index
ind_keep_mse =  df_tot_res.index
ind_keep_mse=ind_keep_mse[df_keep_1]   
df_try=pd.concat([df_sel_s.iloc[:,0],pd.Series([0]*(111-len(df_sel_s)))])
ttest_1samp(df_try,0)
len(df_keep_1)/len(df_sel)
(len(df_sel)-len(df_keep_1))/len(df_sel)

##################
### Evaluation ###
##################

# Function to get difference explained
def diff_explained(df_input,pred,k=5,horizon=12):
    d_nn=[]
    for i in range(len(df_input.columns)):
        real = df_input.iloc[:,i]
        real=real.reset_index(drop=True)
        sf = pred.iloc[:,i]
        sf=sf.reset_index(drop=True)
        max_s=0
        if (real==0).all()==False:
            for value in real[1:].index:
                if (real[value]==real[value-1]):
                    1
                else:
                    max_exp=0
                    if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_s=max_s+max_exp 
            d_nn.append(max_s)
        else:
            d_nn.append(0) 
    return(np.array(d_nn))
    
# Calculate difference explained
d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1)
d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2)
d_nn = np.concatenate([d_nn,d_nn2])

d_b = diff_explained(df_input.iloc[-24:-24+horizon],df_preds_test_1.iloc[:12])
d_b2 = diff_explained(df_input.iloc[-12:],df_preds_test_2.iloc[:12])
d_b = np.concatenate([d_b,d_b2])

d_null = diff_explained(df_input.iloc[-24:-24+horizon],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))))
d_null2 = diff_explained(df_input.iloc[-12:],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))))
d_null = np.concatenate([d_null,d_null2])

d_t1 = diff_explained(df_input.iloc[-24:-24+horizon],df_input.iloc[-24-horizon:-24])
d_t12 = diff_explained(df_input.iloc[-12:],df_input.iloc[-24:-24+horizon])
d_t1= np.concatenate([d_t1,d_t12])

# Calculate MSE
err_sf_pr=[]
err_views=[]
err_zero=[]
err_t1=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
    err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
    err_zero.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pd.Series(np.zeros((horizon,)))))
    err_t1.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i],df_input.iloc[-24-horizon:-24,i]))
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
    err_views.append(mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
    err_zero.append(mean_squared_error(df_input.iloc[-12:,i], pd.Series(np.zeros((horizon,)))))
    err_t1.append(mean_squared_error(df_input.iloc[-12:,i],df_input.iloc[-24:-24+horizon,i]))
err_sf_pr = pd.Series(err_sf_pr)
err_views = pd.Series(err_views)
err_zero = pd.Series(err_zero)
err_t1 = pd.Series(err_t1)

# Calculate DTW distance
from tslearn.metrics import dtw
dtw_sf_pr=[]
dtw_views=[]
dtw_zero=[]
dtw_t1=[]
for i in range(len(df_input.columns)):   
    dtw_sf_pr.append(dtw(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
    dtw_views.append(dtw(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
    dtw_zero.append(dtw(df_input.iloc[-24:-24+horizon,i], pd.Series(np.zeros((horizon,)))))
    dtw_t1.append(dtw(df_input.iloc[-24:-24+horizon,i],df_input.iloc[-24-horizon:-24,i]))
for i in range(len(df_input.columns)):   
    dtw_sf_pr.append(dtw(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
    dtw_views.append(dtw(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
    dtw_zero.append(dtw(df_input.iloc[-12:,i], pd.Series(np.zeros((horizon,)))))
    dtw_t1.append(dtw(df_input.iloc[-12:,i],df_input.iloc[-24:-24+horizon,i]))
dtw_sf_pr = pd.Series(dtw_sf_pr)
dtw_views = pd.Series(dtw_views)
dtw_zero = pd.Series(dtw_zero)
dtw_t1 = pd.Series(dtw_t1)

err_mix = err_views.copy()
err_mix[ind_keep[df_keep_1]] = err_sf_pr.loc[ind_keep[df_keep_1]]
dtw_mix = dtw_views.copy()
dtw_mix[ind_keep[df_keep_1]] = dtw_sf_pr.loc[ind_keep[df_keep_1]]
d_mix = d_b.copy()
d_mix[ind_keep[df_keep_1]] = d_nn[ind_keep[df_keep_1]]
d_nn = d_nn[~np.isnan(d_nn)]
d_b = d_b[~np.isnan(d_b)]
d_null = d_null[~np.isnan(d_null)]
d_t1= d_t1[~np.isnan(d_t1)]
d_mix = d_mix[~np.isnan(d_mix)]

# Difference explained
means = [np.log((d_nn+1)/(d_mix+1)).mean(),np.log((d_b+1)/(d_mix+1)).mean(),np.log((d_null+1)/(d_mix+1)).mean(),np.log((d_t1+1)/(d_mix+1)).mean()]
std_error = [1.993*np.log((x+1)/(d_mix+1)).std()/np.sqrt(len((x-d_mix))) for x in [d_nn,d_b,d_null,d_t1]]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

# MSE
means = [np.log((err_sf_pr+1)/(err_mix+1)).mean(),np.log((err_views+1)/(err_mix+1)).mean(),np.log((err_zero+1)/(err_mix+1)).mean(),np.log((err_t1+1)/(err_mix+1)).mean()]
std_error = [1.993*np.log((x+1)/(err_mix+1)).std()/np.sqrt(len((x-err_mix))) for x in [err_sf_pr,err_views,err_zero,err_t1]]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

name=['SF','Views','Null','t-1']

fig,ax = plt.subplots(figsize=(12,8))
for i in range(4):
    plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],color="gray",s=150)
    plt.plot([mean_mse["mean"][i],mean_mse["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="gray")
    plt.plot([mean_mse["mean"][i]-mean_mse["std"][i],mean_mse["mean"][i]+mean_mse["std"][i]],[mean_de["mean"][i],mean_de["mean"][i]],linewidth=3,color="gray")
plt.scatter(0,0,color="black",s=150)
plt.xlabel("Accuracy ratio (MSE reversed)")
plt.ylabel("Difference explained ratio (DE)")

plt.xlim(0.5,-0.05)
plt.ylim(-0.2,0.06)
plt.text(0.285, 0.004, "t-1", size=20, color='dimgray')
plt.text(0.035,-0.158, "Null", size=20, color='dimgray')
plt.text(0.026, -0.03, "ViEWS", size=20, color='dimgray')
plt.text(0.138, -0.003, 'Shape finder', size=20,color="dimgray")
plt.text(0.033, 0.008, 'Compound', size=20,color="black")
ax.set_yticks([-0.2,-0.15,-0.1,-0.05,0,0.05])
ax.set_xticks([0, 0.1,0.2,0.3,0.4,0.5])
plt.savefig("out/scatter2.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# DTW 
means = [np.log((dtw_views+1)/(dtw_sf_pr+1)).mean(),np.log((dtw_zero+1)/(dtw_sf_pr+1)).mean(),np.log((dtw_t1+1)/(dtw_sf_pr+1)).mean()]
std_error = [1.993*np.log((x+1)/(dtw_sf_pr+1)).std()/np.sqrt(len((x-dtw_sf_pr))) for x in [dtw_views,dtw_zero,dtw_t1]]
mean_dtw = pd.DataFrame({
    'mean': means,
    'std': std_error
})
 
name=['ViEWS','Null','t-1']
fig,ax = plt.subplots(figsize=(12,8))
for i in range(3):
    plt.scatter(i,mean_dtw["mean"][i],color="gray",s=150)
    #plt.plot([mean_dtw["mean"][i],mean_dtw["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="gray")
    plt.plot([i,i],[mean_dtw["mean"][i]-mean_dtw["std"][i],mean_dtw["mean"][i]+mean_dtw["std"][i]],linewidth=3,color="gray")
plt.ylabel("Dynamic time warping ratio")
plt.xticks([0,1,2],name)
plt.axhline(0, linestyle='--',color='lightgrey')
plt.savefig("out/scatter_dtw.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Get DE for varying values of k
mean_mse_k=[]
mean_de_k=[]
for k in [1,5,10]:
    def diff_explained(df_input,pred,k=k,horizon=12):
        d_nn=[]
        for i in range(len(df_input.columns)):
            real = df_input.iloc[:,i]
            real=real.reset_index(drop=True)
            sf = pred.iloc[:,i]
            sf=sf.reset_index(drop=True)
            max_s=0
            if (real==0).all()==False:
                for value in real[1:].index:
                    if (real[value]==real[value-1]):
                        1
                    else:
                        max_exp=0
                        if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                            t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                            max_exp = np.exp(-(k*t))
                        else:
                            if value==horizon-1:
                                if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                    t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                    if max_exp<np.exp(-(k*t)):
                                        max_exp = np.exp(-(k*t))
                            elif value==1:
                                if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                    t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                    if max_exp<np.exp(-(k*t)):
                                        max_exp = np.exp(-(k*t))
                            else : 
                                if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                    t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                    if max_exp<np.exp(-(k*t)):
                                        max_exp = np.exp(-(k*t))
                                if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                    t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                    if max_exp<np.exp(-(k*t)):
                                        max_exp = np.exp(-(k*t))
                        max_s=max_s+max_exp 
                d_nn.append(max_s)
            else:
                d_nn.append(0) 
        return(np.array(d_nn))
    d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1)
    d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2)
    d_nn = np.concatenate([d_nn,d_nn2])
    d_b = diff_explained(df_input.iloc[-24:-24+horizon],df_preds_test_1.iloc[:12])
    d_b2 = diff_explained(df_input.iloc[-12:],df_preds_test_2.iloc[:12])
    d_b = np.concatenate([d_b,d_b2])
    d_null = diff_explained(df_input.iloc[-24:-24+horizon],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))))
    d_null2 = diff_explained(df_input.iloc[-12:],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))))
    d_null = np.concatenate([d_null,d_null2])
    d_t1 = diff_explained(df_input.iloc[-24:-24+horizon],df_input.iloc[-24-horizon:-24])
    d_t12 = diff_explained(df_input.iloc[-12:],df_input.iloc[-24:-24+horizon])
    d_t1= np.concatenate([d_t1,d_t12])
    err_mix = err_views.copy()
    err_mix[ind_keep[df_keep_1]] = err_sf_pr.loc[ind_keep[df_keep_1]]
    dtw_mix = dtw_views.copy()
    dtw_mix[ind_keep[df_keep_1]] = dtw_sf_pr.loc[ind_keep[df_keep_1]]
    d_mix = d_b.copy()
    d_mix[ind_keep[df_keep_1]] = d_nn[ind_keep[df_keep_1]]
    d_nn = d_nn[~np.isnan(d_nn)]
    d_b = d_b[~np.isnan(d_b)]
    d_null = d_null[~np.isnan(d_null)]
    d_t1= d_t1[~np.isnan(d_t1)]
    d_mix = d_mix[~np.isnan(d_mix)]
    
    # Difference explained
    means = [np.log((d_nn+1)/(d_mix+1)).mean(),np.log((d_b+1)/(d_mix+1)).mean(),np.log((d_null+1)/(d_mix+1)).mean(),np.log((d_t1+1)/(d_mix+1)).mean()]
    std_error = [1.993*np.log((x+1)/(d_mix+1)).std()/np.sqrt(len((x-d_mix))) for x in [d_nn,d_b,d_null,d_t1]]
    mean_de_k.append(means)
    
    # MSE
    means = [np.log((err_sf_pr+1)/(err_mix+1)).mean(),np.log((err_views+1)/(err_mix+1)).mean(),np.log((err_zero+1)/(err_mix+1)).mean(),np.log((err_t1+1)/(err_mix+1)).mean()]
    std_error = [1.993*np.log((x+1)/(err_mix+1)).std()/np.sqrt(len((x-err_mix))) for x in [err_sf_pr,err_views,err_zero,err_t1]]
    mean_mse_k.append(means) 
    
mean_mse_k=pd.DataFrame(mean_mse_k)
mean_de_k=pd.DataFrame(mean_de_k)
 
fig,ax = plt.subplots(figsize=(12,8))
plt.plot(mean_mse_k[0],mean_de_k[0],color="gray",markersize=10)
for i,m in zip([0,1,2],["o","x","v"]):
    plt.plot(mean_mse_k[0][i],mean_de_k[0][i],color="gray",marker=m,markersize=10)
plt.plot(mean_mse_k[1],mean_de_k[1],color="gray",markersize=10)
for i,m in zip([0,1,2],["o","x","v"]):
    plt.plot(mean_mse_k[1][i],mean_de_k[1][i],color="gray",marker=m,markersize=10)
plt.plot(mean_mse_k[2],mean_de_k[2],color="gray",markersize=10)
for i,m in zip([0,1,2],["o","x","v"]):
    plt.plot(mean_mse_k[2][i],mean_de_k[2][i],color="gray",marker=m,markersize=10)
plt.plot(mean_mse_k[3],mean_de_k[3],color="gray",markersize=10)
for i,m in zip([0,1,2],["o","x","v"]):
    plt.plot(mean_mse_k[3][i],mean_de_k[3][i],color="gray",marker=m,markersize=10)
plt.scatter(0,0,color="black",s=150)
plt.xlabel("Accuracy ratio (MSE reversed)")
plt.ylabel("Difference explained ratio (DE)")
ax.set_xticks([0, 0.1,0.2,0.3,0.4,0.5])   
plt.xlim(0.5,-0.05)            
plt.text(0.285, 0.004, "t-1", size=20, color='dimgray')
plt.text(0.035,-0.158, "Null", size=20, color='dimgray')
plt.text(0.026, -0.03, "ViEWS", size=20, color='dimgray')
plt.text(0.138, -0.003, 'Shape finder', size=20,color="dimgray")
plt.text(0.033, 0.008, 'Compound', size=20,color="black")
marker1 = plt.Line2D([0], [0], marker='o', color='gray', markersize=10,label="k=1")
marker2 = plt.Line2D([0], [0], marker='x', color='gray', markersize=10,label="k=5")
marker3 = plt.Line2D([0], [0], marker='v', color='gray', markersize=10,label="k=10")
plt.legend(handles=[marker1,marker2,marker3], loc='upper left')
plt.savefig("out/scatter_k_test.jpeg",dpi=400,bbox_inches="tight")

# Illustrate metrics        
fig, axs = plt.subplots(2, 2, figsize=(12,8))
axs[0, 0].plot(df_sf_2.iloc[:,47], label='Shape finder', linestyle="dashed",color='black',linewidth=2)
axs[0, 0].plot(df_preds_test_2.iloc[:12,47].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
axs[0, 0].plot(df_obs_2.iloc[:,47].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['left'].set_visible(False)
axs[0, 0].spines['bottom'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].set_title("Worse MSE",size=20)
axs[0, 0].set_ylabel("Better DE",size=20)
axs[0, 0].set_xlabel("Mali---2023",size=15)

axs[0, 1].plot(df_sf_1.iloc[:,117], linestyle="dashed",color='black',linewidth=2)
axs[0, 1].plot(df_preds_test_1.iloc[:12,117].reset_index(drop=True),linestyle="dotted",color='black',linewidth=2)
axs[0, 1].plot(df_obs_1.iloc[:,117].reset_index(drop=True),linestyle="solid",color="black",linewidth=2)
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 1].set_title('Better MSE',size=20)
axs[0, 1].set_xlabel("Saudi Arabia---2022",size=15)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['left'].set_visible(False)
axs[0, 1].spines['bottom'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)

axs[1, 0].plot(df_sf_1.iloc[:,88], linestyle="dashed",color='black',linewidth=2)
axs[1, 0].plot(df_preds_test_1.iloc[:12,88].reset_index(drop=True),linestyle="dotted",color='black',linewidth=2)
axs[1, 0].plot(df_obs_1.iloc[:,88].reset_index(drop=True),linestyle="solid",color="black",linewidth=2)
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['left'].set_visible(False)
axs[1, 0].spines['bottom'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].set_ylabel("Worse DE",size=20)
axs[1, 0].set_xlabel("Poland---2022",size=15)

axs[1, 1].plot(df_sf_1.iloc[:,152], linestyle="dashed",color='black',linewidth=2)
axs[1, 1].plot(df_preds_test_1.iloc[:12,152].reset_index(drop=True),linestyle="dotted",color='black',linewidth=2)
axs[1, 1].plot(df_obs_1.iloc[:,152].reset_index(drop=True),linestyle="solid",color="black",linewidth=2)
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['left'].set_visible(False)
axs[1, 1].spines['bottom'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].set_xlabel("Congo, DRC---2022",size=15)
plt.tight_layout()
plt.savefig("out/cross_tab.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# MSE for varying prediction horizon
def err(y_true,y_pred):
    return (np.log(y_true+1)-np.log(y_pred+1))**2

dict_hor = {'Shape finder':[],'ViEWS':[],'Zeros':[],'t-1':[],'Compound':[]}
dict_hor = {'Shape finder':[],'ViEWS':[],'Zeros':[],'t-1':[]}
for h in range(12):
    err_sf_pr=[]
    err_views=[]
    err_zero=[]
    err_t1=[]
    err_mix=[]
    for i in range(len(df_input.columns)):   
        if (df_input.iloc[-34:-24,i]==0).all() == True:
            err_mix.append(err(df_input.iloc[-24+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        elif i in ind_keep_mse:
            err_mix.append(err(df_input.iloc[-24+h,i], df_sf_1.iloc[h,i]))
        elif i not in ind_keep_mse.tolist():
            err_mix.append(err(df_input.iloc[-24+h,i], df_preds_test_1.iloc[h,i]))
        err_sf_pr.append(err(df_input.iloc[-24+h,i], df_sf_1.iloc[h,i]))
        err_views.append(err(df_input.iloc[-24+h,i], df_preds_test_1.iloc[h,i]))
        err_zero.append(err(df_input.iloc[-24+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        err_t1.append(err(df_input.iloc[-24+h,i],df_input.iloc[-36+h,i])) 
        if (df_input.iloc[-22:-12,i]==0).all() == True:
            err_mix.append(err(df_input.iloc[-12+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        elif i in ind_keep_mse-191:
            err_mix.append(err(df_input.iloc[-12+h,i], df_sf_2.iloc[h,i]))
        elif i not in ind_keep_mse-191:
            err_mix.append(err(df_input.iloc[-12+h,i], df_preds_test_2.iloc[h,i]))
        err_sf_pr.append(err(df_input.iloc[-12+h,i], df_sf_2.iloc[h,i]))
        err_views.append(err(df_input.iloc[-12+h,i], df_preds_test_2.iloc[h,i]))
        err_zero.append(err(df_input.iloc[-12+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        err_t1.append(err(df_input.iloc[-12+h,i],df_input.iloc[-24+h,i]))
    err_sf_pr = np.log((pd.Series(err_sf_pr)+1)/(pd.Series(err_mix)+1))
    err_views = np.log((pd.Series(err_views)+1)/(pd.Series(err_mix)+1))
    err_zero = np.log((pd.Series(err_zero)+1)/(pd.Series(err_mix)+1))
    err_t1 = np.log((pd.Series(err_t1)+1)/(pd.Series(err_mix)+1))
    dict_hor['Shape finder'].append([err_sf_pr.mean(),err_sf_pr.std()])
    dict_hor['ViEWS'].append([err_views.mean(),err_views.std()])
    dict_hor['Zeros'].append([err_zero.mean(),err_zero.std()])
    dict_hor['t-1'].append([err_t1.mean(),err_t1.std()])

test = pd.DataFrame(dict_hor)
horizons = np.arange(len(test))+1
mean_mse=[]
sd_mse=[]
for column in test.columns[[0,1,2,3]]:
    means = test[column].apply(lambda x: x[0])
    stds = test[column].apply(lambda x: x[1])
    mean_mse.append(means)
    sd_mse.append(stds)
mean_mse=pd.DataFrame(mean_mse).T
sd_mse=pd.DataFrame(sd_mse).T

# DE for varying prediction horizon
def diff_explained_h(df_input,pred,k=5,horizon=12):
    real = df_input.copy()
    real=real.reset_index(drop=True)
    sf = pred.copy()
    sf=sf.reset_index(drop=True)
    max_s=[]
    if (real==0).all()==False:
        for value in real[1:].index:
            if (real[value]==real[value-1]):
                max_exp=0
            else:
                max_exp=0
                if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                    t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                    max_exp = np.exp(-(k*t))
                else:
                    if value==horizon-1:
                        if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                            t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                            if max_exp<np.exp(-(k*t)):
                                max_exp = np.exp(-(k*t))
                    elif value==1:
                        if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                            t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                            if max_exp<np.exp(-(k*t)):
                                max_exp = np.exp(-(k*t))
                    else : 
                        if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                            t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                            if max_exp<np.exp(-(k*t)):
                                max_exp = np.exp(-(k*t))
                        if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                            t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                            if max_exp<np.exp(-(k*t)):
                                max_exp = np.exp(-(k*t))
            max_s.append(max_exp)
    else:
        max_s.append([0]*(horizon-1)) 
    return(max_s)

horizon=12
err_sf_pr=[]
err_views=[]
err_zero=[]
err_t1=[]
err_mix=[]
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        err_sf_pr.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        err_views.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
        err_zero.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], pd.Series(np.zeros((horizon,)))))
        err_t1.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i],df_input.iloc[-24-horizon:-24,i]))
        if i in ind_keep_mse:
            err_mix.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        else:
            err_mix.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-12:,i]==0).all()==False:
        err_sf_pr.append(diff_explained_h(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        err_views.append(diff_explained_h(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
        err_zero.append(diff_explained_h(df_input.iloc[-12:,i], pd.Series(np.zeros((horizon,)))))
        err_t1.append(diff_explained_h(df_input.iloc[-12:,i],df_input.iloc[-24:-24+horizon,i]))
        if i in ind_keep_mse-191:
            err_mix.append(diff_explained_h(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        else:
            err_mix.append(diff_explained_h(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))

mean_de=[]
std_de=[]
sum_tot_m=[]
sd_tot_m=[]
mod2 = pd.DataFrame(err_mix)
for mod in [err_sf_pr,err_views,err_zero,err_t1]:
    mod = pd.DataFrame(mod)
    mean_de.append(mod.mean())
    std_de.append(mod.std())
    sum_de=[]
    sd_m=[]
    for i in range(11):
        sum_de.append(np.log((mod.iloc[:,i]+1)/(mod2.iloc[:,i]+1)).mean())
        sd_m.append(np.log((mod.iloc[:,i]+1)/(mod2.iloc[:,i]+1)).std())
    sum_tot_m.append(sum_de)
    sd_tot_m.append(sd_m)
mean_de=pd.DataFrame(sum_tot_m).T
std_de =pd.DataFrame(sd_tot_m).T
mean_de.columns=['Shape finder','ViEWS','Zeros','t-1']
std_de.columns = ['Shape finder','ViEWS','Zeros','t-1']

colors = ['black', 'gray', 'royalblue',"lightskyblue"]
letters=['A','B','C','D']
mean_mse.columns=['Shape finder','ViEWS','Null','t-1']
mean_de.columns=['Shape finder','ViEWS','Null','t-1']
name_m = mean_mse.iloc[:, [0, 1,3]].columns.tolist()
fig,ax = plt.subplots(figsize=(12,8))
for i, col in enumerate(mean_mse.iloc[:, [0, 1,3]].columns):
    x_values = []
    y_values = []
    line_x=[]
    line_y=[]
    for k in range(4):
        if k == 4:
            1
        else:
            x = mean_mse[col].iloc[(k * 4):((k + 1) * 4)].mean()
            y = mean_de.loc[k * 4:(k + 1) * 4, col].mean()
            x_values.append(x)
            y_values.append(y)
            line_x.append(x)
            line_y.append(y)

    plt.plot(x_values, y_values, color=colors[i], linestyle='-', linewidth=0,marker='o',markersize=15,zorder=5)
    plt.plot(line_x, line_y, color=colors[i], linestyle='-', linewidth=2)
    for j in range(len(line_x) - 1):
        dx = (line_x[j + 1] - line_x[j])/2
        dy = (line_y[j + 1] - line_y[j])/2
        plt.annotate('', xy=(line_x[j]+dx, line_y[j]+dy), xytext=(line_x[j], line_y[j]),
                     arrowprops=dict(arrowstyle='->', color=colors[i]),size=20)
for i, col in enumerate(mean_mse.iloc[:, [0, 1,3]].columns):
    for k in range(5):
        if k == 4:
            plt.scatter(-1000, 0, 
                        color=colors[i], 
                        marker='o',
                        label=name_m[i], s=50)
        else:
            x = mean_mse[col].iloc[(k * 4):((k + 1) * 4)].mean()
            y = mean_de.loc[k * 4:(k + 1) * 4, col].mean()
            plt.scatter(x, y, 
                        color='white', 
                        marker='$'+letters[k]+'$', s=100,zorder=10)
plt.plot(0, 0, marker='o',linewidth=0, label='Compound',color='lightgrey',markersize=8,zorder=10)
plt.xlabel('Accuracy ratio (MSE reversed)')
plt.ylabel('Difference explained ratio (DE)')
plt.axhline(0, linestyle='--', alpha=0.4)
plt.axvline(0, linestyle='--', alpha=0.4)
plt.xlim(0.052, -0.008)
plt.ylim(-0.017, 0.007)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5,fontsize=15)
plt.savefig("out/horizon_test_short.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Select balancing factor k for DE formula
df_t = pd.DataFrame([0,0.2,0,0.5,0,0.96,0.2,0,0.15,0])
df_1 = pd.DataFrame([0.05,0.05,0.6,0.25,0,0.7,0.05,0,0,0])
df_2 = pd.DataFrame([1,0,1,0,1,0,1,0,1,0])
df_3 = pd.DataFrame([0.5,0.6,0.5,0.55,0.5,0.65,0.55,0.12,0,0.2])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
ax1.plot(df_1, label='Model 1',color='dimgray',linewidth=3,linestyle="solid")
ax1.plot(df_2, label='Model 2',color='darkgray',linewidth=2,linestyle="dashed")
ax1.plot(df_3, label='Model 3',color='darkgray',linewidth=2,linestyle="dotted")
ax1.plot(df_t, label='Actuals',color='black',linewidth=3)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,fontsize=12)
k_values = [1, 5, 10]
models = ['Model 1', 'Model 2', 'Model 3']
differences = {
    'Model 1': [diff_explained(df_t, df_1, k, horizon=10)[0] for k in k_values],
    'Model 2': [diff_explained(df_t, df_2, k, horizon=10)[0] for k in k_values],
    'Model 3': [diff_explained(df_t, df_3, k, horizon=10)[0] for k in k_values],
}
podiums = {k: sorted(models, key=lambda m: differences[m][i]) for i, k in enumerate(k_values)}
bar_width = 0.2
col=["gray","darkgray","lightgray","black"]
x = np.arange(len(k_values))
for i, model in enumerate(models):
    ax2.bar(x + i * bar_width, [differences[model][j] for j in range(len(k_values))], bar_width, label=model,color=col[i])
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(['k=1', 'k=5', 'k=10'],fontsize=15)
ax2.set_title('Difference explained (DE)',fontsize=20)
ax1.set_title('Actuals and predicted outcome',fontsize=20)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([])
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,fontsize=12)
plt.tight_layout()
plt.savefig("out/de_exp_fictional.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Select conditions for best and worst performance
mean_d=[]
std_d=[]
per_d=[]
mean_m=[]
std_m=[]
per_m=[]
sca=[]
ap_ent=[]
countries=[]
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        countries.append(f"{df_input.columns[i]}--2022")
        ser = (df_input.iloc[-34:-24,i] - df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        diff = ser.diff()
        mean_d.append(abs(diff).mean())
        std_d.append(abs(diff).std())
        per_d.append((diff>0).mean())
        mean_m.append(ser.mean())
        std_m.append(ser.std())
        per_m.append((ser>ser.mean()).mean())
        sca.append(df_input.iloc[-34:-24,i].sum())
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        countries.append(f"{df_input.columns[i]}--2023")
        ser = (df_input.iloc[-22:-12,i] - df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        diff = ser.diff()
        mean_d.append(abs(diff).mean())
        std_d.append(abs(diff).std())
        per_d.append((diff>0).mean()) 
        mean_m.append(ser.mean())
        std_m.append(ser.std())
        per_m.append((ser>ser.mean()).mean())
        sca.append(df_input.iloc[-22:-12,i].sum())        
df_var = pd.DataFrame([mean_d,std_d,per_d,sca,df_sel['log MSE'],mean_m,std_m,per_m,ap_ent]).T
X = df_var.iloc[:, [1,5]]
y = df_var.iloc[:, 4]
model = DecisionTreeRegressor(random_state=42, max_depth=2,min_samples_split=10,min_samples_leaf=10)
model.fit(X, y)

# Best
fig,ax=plt.subplots(figsize=(9,6))
plot_tree(model, feature_names=[ 'Diff SD','Mean'], rounded=True,impurity=False) 
for text in ax.texts:
    if 'Mean <= 0.395\nsamples = 111\nvalue = -0.076' in text.get_text():
        text.set_text('Mean $<=$ 0.395\nSamples = 111\nValue = -0.076')  
for text in ax.texts:
    if 'Mean <= 0.286\nsamples = 90\nvalue = -0.196' in text.get_text():
        text.set_text('Mean $<=$ 0.286\nSamples = 90\nValue = -0.196')  
for text in ax.texts:
    if 'samples = 71\nvalue = -0.063' in text.get_text():
        text.set_text('Samples = 71\nValue = -0.063')  
for text in ax.texts:
    if 'samples = 19\nvalue = -0.691' in text.get_text():
        text.set_text('Samples = 19\nValue = -0.691')                     
for text in ax.texts:
    if 'Diff SD <= 0.252\nsamples = 21\nvalue = 0.437' in text.get_text():
        text.set_text('Diff SD $<=$ 0.252\nSamples = 21\nValue = 0.437')                     
for text in ax.texts:
    if 'samples = 10\nvalue = 0.944' in text.get_text():
        text.set_text('Samples = 10\nValue = 0.944') 
        text.set_color('darkblue')
for text in ax.texts:
    if 'samples = 11\nvalue = -0.023' in text.get_text():
        text.set_text('Samples = 11\nValue = -0.023')  
plt.savefig("out/decision_tree_best.jpeg",dpi=400,bbox_inches="tight")
plt.show()

df_subr=df_var[(df_var.iloc[:,1]<0.2516) & (df_var.iloc[:,5]>0.3951)]
norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap('RdBu')
color = cmap(norm(df_subr))
fig,ax = plt.subplots(figsize=(12,8))
plt.fill_betweenx(y=[0.3951, 0.7], x1=0.2516, color="gray", alpha=0.2,linestyle='--')
plt.axhline(0.3951,xmax=0.3316,linestyle='--',color='black',alpha=0.2)
plt.axvline(0.2516,ymin=0.5551,linestyle='--',color='black',alpha=0.2)
plt.scatter(df_var.iloc[:,1],df_var.iloc[:,5],c=df_var.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_var.iloc[:,3])*20,alpha=0.2)
plot=plt.scatter(df_subr.iloc[:,1],df_subr.iloc[:,5],c=df_subr.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_subr.iloc[:,3])*20)
plt.axhline(0.3951,linestyle='--',color='black',alpha=0.2)
plt.axvline(0.2516,linestyle='--',color='black',alpha=0.2)
plt.xlim(0.1,0.55)
plt.ylim(0.04,0.67)
plt.xlabel('Standard deviation of the differentiated input series (Diff SD)',fontsize=15)
plt.ylabel('Mean of all data points in input series (Mean)',fontsize=15)
cax_cb = fig.add_axes([0.95, 0.23, 0.015, 0.4])  
cbar = plt.colorbar(plot, ax=ax,cax=cax_cb,orientation='vertical')
cbar.ax.set_title('Log-ratio', fontsize=15, pad=15)
size_ranges = [20, 80, 160, 240]
legend_labels = ['0', '4', '8', '12']
legend_handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(size), linestyle='', markeredgewidth=1) for size in size_ranges]
ax.legend(legend_handles, legend_labels,loc='center left', bbox_to_anchor=(1, 0.9), title='Fatalities (log)', title_fontsize='15', fontsize='15',frameon=False)
for i in range(len(df_var)):
    if df_var.iloc[i,1] < 0.2516 and df_var.iloc[i,5] > 0.3951 and df_var.iloc[i,4]>0.3:
        ax.annotate(countries[i], (df_var.iloc[i,1], df_var.iloc[i,5]-0.005),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9)
plt.savefig("out/scatter_best.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Worst
X = df_var.iloc[:, [6,5]]
y = df_var.iloc[:, 4]
w = pd.Series([1]*len(df_var),index = df_var.index)
w.loc[63] = 3
w.loc[22] = 2
model_2 = DecisionTreeRegressor(random_state=42, max_depth=2,max_features=1)
model_2.fit(X,y,w)
fig,ax=plt.subplots(figsize=(9,6))
plot_tree(model_2, feature_names=[ 'SD','Mean'], rounded=True,impurity=False)  
for text in ax.texts:
    if 'SD <= 0.375\nsamples = 111\nvalue = -0.102' in text.get_text():
        text.set_text('SD $<=$ 0.375\nSamples = 111\nValue = -0.102')
for text in ax.texts:
    if 'SD <= 0.29\nsamples = 96\nvalue = 0.035' in text.get_text():
        text.set_text('SD $<=$ 0.29\nSamples = 96\nValue = 0.035')
for text in ax.texts:
    if 'samples = 5\nvalue = 1.09' in text.get_text():
        text.set_text('Samples = 5\nValue = 1.09')
for text in ax.texts:
    if 'samples = 91\nvalue = -0.023' in text.get_text():
        text.set_text('Samples = 91\nValue = -0.023')
for text in ax.texts:
    if 'Mean <= 0.436\nsamples = 15\nvalue = -0.835' in text.get_text():
        text.set_text('Mean $<=$ 0.436\nSamples = 15\nValue = -0.835')        
for text in ax.texts:
    if 'samples = 14\nvalue = -1.139' in text.get_text():
        text.set_text('Samples = 14\nValue = -1.139')    
        text.set_color('darkred')
for text in ax.texts:
    if 'samples = 1\nvalue = 0.682' in text.get_text():
        text.set_text('Samples = 1\nValue = 0.682')                         
plt.savefig("out/decision_tree_worst.jpeg",dpi=400,bbox_inches="tight")
plt.show()

df_subr=df_var[(df_var.iloc[:,6]>0.37) & (df_var.iloc[:,5]<0.43)]
norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap('RdBu')
color = cmap(norm(df_subr))
fig,ax = plt.subplots(figsize=(12,8))
plt.fill_betweenx(y=[0.43, 0.05], x1=0.37,x2=0.45, color="gray", alpha=0.2,linestyle='--')
plt.axhline(0.43,xmin=0.6, linestyle='--', color='gray', alpha=1)
plt.axvline(0.37,ymax=0.62, linestyle='--', color='gray', alpha=1)
plt.scatter(df_var.iloc[:,6],df_var.iloc[:,5],c=df_var.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_var.iloc[:,3])*20,alpha=0.2)
plt.scatter(df_subr.iloc[:,6],df_subr.iloc[:,5],c=df_subr.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_subr.iloc[:,3])*20)
plt.axhline(0.43, linestyle='--', color='black', alpha=0.2)
plt.axvline(0.37, linestyle='--', color='black', alpha=0.2)
plt.xlim(0.25,0.45)
plt.ylim(0.05,0.67)
plt.xlabel('Standard deviation of the input series (SD)')
plt.ylabel('Mean of all data points in input series (Mean)')
cax_cb = fig.add_axes([0.95, 0.23, 0.015, 0.4])  
cbar = plt.colorbar(plot, ax=ax,cax=cax_cb,orientation='vertical')
cbar.ax.set_title('Log-ratio', fontsize=15, pad=15)
size_ranges = [20, 80, 160, 240]
legend_labels = ['0', '4', '8', '12']
legend_handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(size), linestyle='', markeredgewidth=1) for size in size_ranges]
ax.legend(legend_handles, legend_labels,loc='center left', bbox_to_anchor=(1, 0.9), title='Fatalities (log)', title_fontsize='15', fontsize='15',frameon=False)
for i in range(len(df_var)):
    if df_var.iloc[i,6] > 0.35 and df_var.iloc[i,5] < 0.43 and df_var.iloc[i,4]<-0.8:
        ax.annotate(countries[i], (df_var.iloc[i,6], df_var.iloc[i,5]-0.005),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9)
plt.savefig("out/scatter_worst.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Worst performing cases
def remove_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
        
fig, axs = plt.subplots(4, 1, figsize=(10, 13))
axs[0].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 119].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[0])
axs[0].set_xticks([])
axs[0].set_yticks([])
ax2 = axs[0].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 119].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 119].reset_index(drop=True),linestyle="dashed",color='black',linewidth=2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 119].reset_index(drop=True), linestyle="dotted",color='black',linewidth=2)
axs[0].set_title(f'{df_input.columns[119]} 2021---2022')
ax2.plot([9,10],[6,27], linestyle="solid",color='black',linewidth=2)
axs[0].axvline(x=10, color='black', linestyle='--', linewidth=1)

axs[1].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 123].reset_index(drop=True), linewidth=2, color='black')   
remove_box(axs[1])
axs[1].set_xticks([])
axs[1].set_yticks([])
ax2 = axs[1].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 123].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 123].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 123].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[1].set_title(f'{df_input.columns[123]} 2022---2023')
axs[1].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[0.009,0], linestyle="solid",color='black',linewidth=2)

axs[2].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 190].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[2])
axs[2].set_xticks([])
axs[2].set_yticks([])
ax2 = axs[2].twinx()
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 190].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 190].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 190].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
ax2.set_yticks([])
remove_box(ax2)
axs[2].set_title(f'{df_input.columns[190]} 2021---2022')
axs[2].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[38,0], linestyle="solid",color='black',linewidth=2)

axs[3].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 109].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[3])
axs[3].set_xticks([])
axs[3].set_yticks([])
ax2 = axs[3].twinx()
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 109].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 109].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 109].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
ax2.set_yticks([])
remove_box(ax2)
axs[3].set_title(f'{df_input.columns[109]} 2022---2023')
axs[3].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[0,0], linestyle="solid",color='black',linewidth=2)
plt.tight_layout()
plt.savefig("out/cases_worst2.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Best performing cases
fig, axs = plt.subplots(4, 1, figsize=(10, 13))
axs[0].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 177].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[0])
axs[0].set_xticks([])
axs[0].set_yticks([])
ax2 = axs[0].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 177].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 177].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 177].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[0].set_title(f'{df_input.columns[177]} 2021---2022')
ax2.plot([9,10],[6,85], linestyle="solid",color='black',linewidth=2)
axs[0].axvline(x=10, color='black', linestyle='--', linewidth=1)

axs[1].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 179].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[1])
axs[1].set_xticks([])
axs[1].set_yticks([])
ax2 = axs[1].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 179].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 179].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 179].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[1].set_title(f'{df_input.columns[179]} 2021---2022')
axs[1].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[68.3,35], linestyle="solid",color='black',linewidth=2)

axs[2].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 63].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[2])
axs[2].set_xticks([])
axs[2].set_yticks([])
ax2 = axs[2].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 63].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 63].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 63].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[2].set_title(f'{df_input.columns[63]} 2022---2023')
axs[2].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[57,19], linestyle="solid",color='black',linewidth=2)

axs[3].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 26].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[3])
axs[3].set_xticks([])
axs[3].set_yticks([])
ax2 = axs[3].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 26].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 26].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 26].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[3].set_title(f'{df_input.columns[26]} 2022---2023')
axs[3].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[9.2,10], linestyle="solid",color='black',linewidth=2)
plt.tight_layout()
plt.savefig("out/cases_best2.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Best grid
df_subr=df_var[(df_var.iloc[:,1]<0.27) & (df_var.iloc[:,5]>0.44)][4].mean()
df_input = df_input.fillna(0)
h_train=10
h=12
fig, axs = plt.subplots(5, 5, figsize=(19,12))
row=0
col=0
nume=30
flag=True
while flag==True:
    nume=nume+12
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
            ser=df_input.iloc[nume:nume+h_train,coun]
            ser=(ser-ser.min())/(ser.max()-ser.min())
            diff = ser.diff()
            if (abs(diff).std()<0.25) & (ser.mean()>0.44):
                axs[row, col].plot(ser.reset_index(drop=True), linewidth=3, color='black')
                remove_box(axs[row, col])
                axs[row, col].set_xticks([])
                axs[row, col].set_ylim(-0.1,1.1)
                axs[row, col].set_yticks([])
                axs[row, col].set_title(f"{df_input.columns[coun]}",size=40)
                row=row+1
                if row==5:
                    col=col+1
                    row=0
                if col==5:
                    flag=False
plt.tight_layout()
plt.savefig("out/cases_best_grid.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Worst grid
fig, axs = plt.subplots(5, 5, figsize=(19,12))
row=0
col=0
nume=1
flag=True
while flag==True:
    nume=nume+12
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
            ser=df_input.iloc[nume:nume+h_train,coun]
            ser=(ser-ser.min())/(ser.max()-ser.min())
            diff = ser.diff()
            if (ser.std()>0.37) & (ser.mean()<0.4):
                axs[row, col].plot(ser.reset_index(drop=True), linewidth=3, color='black')
                remove_box(axs[row, col])
                axs[row, col].set_xticks([])
                axs[row, col].set_ylim(-0.1,1.1)
                axs[row, col].set_yticks([])
                if df_input.columns[coun]=="United Kingdom":
                    axs[row, col].set_title(f"UK",size=40)
                elif df_input.columns[coun]=="Papua New Guinea":
                    axs[row, col].set_title(f"PNG",size=40)                   
                else:
                    axs[row, col].set_title(f"{df_input.columns[coun]}",size=40)
                row=row+1
                if row==5:
                    col=col+1
                    row=0
                if col==5:
                    flag=False
plt.tight_layout()
plt.savefig("out/cases_worst_grid.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Check characteristics of best performing cases
err_sf_pr_n=[]
we_n=[]
for i in range(len(df_input.columns)):  
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        true = df_input.iloc[-24:-24+horizon,i]
        true = (true-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        pred = df_sf_1.iloc[:,i]
        pred = (pred-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        err_sf_pr_n.append(mean_squared_error(true, pred))
        we_n.append(df_input.iloc[-24:-24+horizon,i].sum())
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        true = df_input.iloc[-12:,i]
        true = (true-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        pred = df_sf_2.iloc[:,i]
        pred = (pred-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        err_sf_pr_n.append(mean_squared_error(true, pred))
        we_n.append(df_input.iloc[-12:,i].sum())
err_sf_pr_n = [i+0.001 for i in err_sf_pr_n]
we_n = [i+1 for i in we_n]
df_norm = pd.DataFrame([err_sf_pr_n,we_n]).T
df_be=pd.read_csv('Datasets/df_be.csv',index_col=0)
mse_be_w = df_be.iloc[:,1]
mse_be_n = df_be.iloc[:,0]
df_norm=pd.read_csv('Datasets/df_norm.csv',index_col=0)
we_n = df_norm.iloc[:,1]
err_sf_pr_n = df_norm.iloc[:,0]
plt.figure(figsize=(14, 10))
plt.scatter(np.log(we_n),np.log(err_sf_pr_n),color='gray',alpha=0.6)
plt.scatter(np.log(mse_be_w),np.log(mse_be_n),color='black',marker='x',alpha=1)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log(we_n), np.log(err_sf_pr_n))
plt.plot(np.log(we_n), intercept1 + slope1 * np.log(we_n), color='gray', linestyle='--')
plt.xlabel('Sum of fatalies in forecasting window (log)')
plt.ylabel('Normalized MSE (log)')
plt.savefig("out/best_reg.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Validate reason for Shape Finder overpredictions
df_match_w=pd.DataFrame()
rea=[]
err_spe=[]
inde=[]
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        ser = (df_input.iloc[-34:-24,i] - df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        if (ser.std()>0.37) & (ser.mean()<0.4):
            seq = dict_m[df_input.columns[i]]
            mat = [i[0].sum() for i in seq]
            mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
            df_match_w=pd.concat([df_match_w,mat.value_counts(normalize=True)],axis=1)
            rea.append(df_input.iloc[-34:-24,i].sum())
            err_s=mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])
            err_s_v=mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i])
            err_spe.append(np.log((err_s_v+1)/(err_s+1)))
            inde.append(df_input.columns[i]+' 2022')

with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        ser = (df_input.iloc[-22:-12,i] - df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        if (ser.std()>0.37) & (ser.mean()<0.4):
            seq = dict_m[df_input.columns[i]]
            mat = [i[0].sum() for i in seq]
            mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
            df_match_w=pd.concat([df_match_w,mat.value_counts(normalize=True)],axis=1)
            rea.append(df_input.iloc[-22:-12,i].sum())
            err_s=mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])
            err_s_v=mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i])
            err_spe.append(np.log((err_s_v+1)/(err_s+1)))
            inde.append(df_input.columns[i]+' 2023')

df_match_w = df_match_w.T        
df_match_w = pd.concat([df_match_w.reset_index(drop=True),pd.cut(pd.Series(rea), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)],axis=1)
df_match_w.index=inde
df_match_w['MSE']=err_spe
latex_table = df_match_w.sort_values('MSE').iloc[:5, :].to_latex(index=True, caption='Top 5 Entries by Log Ratio MSE', label='tab:top5_mse', float_format="%.2f")
print(latex_table)
with open('out/sf_overpredictions.tex', 'w') as o:
     o.write(latex_table)

# Validate categorization into hihg and low complexity cases
ts_tot_l = []
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-34:-24, i] == 0).all() == False:
        ts_tot_l.append(df_input.iloc[-34:-24, i])
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12, i] == 0).all() == False:
        ts_tot_l.append(df_input.iloc[-22:-12, i]) 
high_complexity_cases = df_var[(df_var.iloc[:, 1] < 0.2516) & (df_var.iloc[:, 5] > 0.3951)].iloc[:4]
low_complexity_cases = df_var[(df_var.iloc[:, 6] > 0.37) & (df_var.iloc[:, 5] < 0.35)].iloc[:4]
fig, axes = plt.subplots(2, 5, figsize=(24, 12),gridspec_kw={'width_ratios': [1, 1, 0.2, 1, 1]})  
axes = axes.ravel()
for idx, i in enumerate(high_complexity_cases.index):
    if idx < 2:
        ax = axes[idx]
    else:
        ax = axes[idx + 3] 
    ax.plot(ts_tot_l[i],linewidth=3)
    ax.set_title(f"{ts_tot_l[i].name} \n Mean={df_var.iloc[i, 5]:.2f}, SD (diff)={df_var.iloc[i, 1]:.2f}",fontsize=35)
    ax.axis('off') 
for idx, i in enumerate(low_complexity_cases.index):
    if idx < 2:
        ax = axes[idx + 3]
    else:
        ax = axes[idx + 6] 
    ax.plot(ts_tot_l[i],linewidth=3)
    ax.set_title(f"{ts_tot_l[i].name}\nMean={df_var.iloc[i, 5]:.2f}, SD={df_var.iloc[i, 6]:.2f}",fontsize=35)
    ax.axis('off')
axes[2].axis('off')
axes[7].axis('off')
plt.tight_layout()
plt.savefig("out/low_high_com.jpeg",dpi=400,bbox_inches="tight")
plt.show()

#########################
### Robustness checks ###
#########################

# (a) Top 1 3 5 matches

# Original model
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]          
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())        
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()   
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
df_sf_1 = pd.concat(pred_tot_pr,axis=1)
df_sf_1.columns=country_list['name']

df_sf_1_tot=[]
for k_min in [1,3,5]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            if (df_input_sub.columns[coun] in ['Peru','Burkina Faso','Cameroon','Tunisia','Ukraine','Somalia','Armenia','Iran','Afghanistan','Kyrgyzstan','Myanmar','Thailand','Mozambique','Congo, RDC','Indonesia','Israel','Syria','India','Kenya','Sudan']) & k_min==1:
                k_min=2
            for col,last_date,mi,ma,somme in tot_seq[:k_min]:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            pred_ori=tot_seq.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)
    
# Original model
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]           
        pred_seq=[]                     
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
            if date+horizon<len(df_tot_m.iloc[:-12]):               
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)              
                pred_seq.append(seq.tolist())                
        tot_seq=pd.DataFrame(pred_seq)       
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:     
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
    
df_sf_2= pd.concat(pred_tot_pr,axis=1)
df_sf_2.columns=country_list['name']    
    
df_sf_2_tot=[]
for k_min in [1,3,5]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-12]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            if (df_input_sub.columns[coun] in ['Brazil','Peru','Mali','Iraq','Burkina Faso','Cameroon','Turkey','Uzbekistan','Yemen','Azerbaijan','Iran','Kyrgyzstan','Tajikistan','Burundi','South Africa','Mozambique','Papua New Guinea','Libya','Israel','Syria','Egypt','Morocco','South Sudan']) & k_min==1:
                k_min=2
            for col,last_date,mi,ma,somme in tot_seq[:k_min]:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-12]):                              
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            pred_ori=tot_seq.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))            
    df_sf_2_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']
    df_sf_2_tot.append(df_sf_2_sub)  
 
# Evaluate 
mse_list_sub=[]
for n_k,k_min in enumerate([1,3,5]):
    err_sf_pr=[]
    err_sub=[]
    for i in range(len(df_input.columns)):   
        err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_tot[n_k].iloc[:,i]))
        err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_tot[n_k].iloc[:,i]))
    err_sf_pr = np.array(err_sf_pr)
    err_sub = np.array(err_sub)
    mse_list_sub.append(np.log((err_sub+1)/(err_sf_pr+1)))   
d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1)
d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2)
d_nn = np.concatenate([d_nn,d_nn2])    
de_list_sub=[]
for n_k,k_min in enumerate([1,3,5]):
    d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_tot[n_k])
    d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_tot[n_k])
    d_sub = np.concatenate([d_sub,d_sub2])
    de_list_sub.append(np.log((d_nn+1)/(d_sub+1)))
    
# (b) Change value for cut-off
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
horizon=12
h_train=10
for cut in [2,4,5]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/cut, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)           
                      
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
horizon=12
for cut in [2,4,5]:
    df_input_sub=df_input.iloc[:-12]
    pred_tot_min=[]
    pred_tot_pr=[]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
            l_find=dict_m[df_input.columns[coun]]
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]           
            pred_seq=[]                     
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
                if date+horizon<len(df_tot_m.iloc[:-12]):               
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)              
                    pred_seq.append(seq.tolist())                
            tot_seq=pd.DataFrame(pred_seq)       
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/cut, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()
            pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:     
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
    df_sf_2_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']
    df_sf_2_tot.append(df_sf_2_sub)  
       
# (c) Change value for distance
with open('Results/test1_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
for thres in [0.2,0.3,0.4,0.5,0.75,1]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]     
            tot_seq = []
            for series, weight in l_find:
                if weight<thres:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            if len(tot_seq)<5:
                tot_seq = []
                for series, weight in l_find[:5]:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            tot_seq = tot_seq.dropna()
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)
    
with open('Results/test2_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
for thres in [0.2,0.3,0.4,0.5,0.75,1]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-12]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]     
            tot_seq = []
            for series, weight in l_find:
                if weight<thres:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            if len(tot_seq)<5:
                tot_seq = []
                for series, weight in l_find[:5]:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-12]):                              
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            tot_seq = tot_seq.dropna()
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))            
    df_sf_2_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']
    df_sf_2_tot.append(df_sf_2_sub)
    
# (g) Weighted clusters as prediction
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]          
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])        
        pred_seq=[]
        w_list=[]
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())     
                w_list.append(wei)                
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        tot_seq['weights'] = w_list
        val_sce = tot_seq.groupby('Cluster').apply(lambda x: (x.iloc[:, :12].T * x['weights']).sum(axis=1) / x['weights'].sum())
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
df_sf_1_w_clu = pd.concat(pred_tot_pr,axis=1)
df_sf_1_w_clu.columns=country_list['name']
df_sf_1_w_clu = df_sf_1_w_clu.fillna(0)

with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])        
        pred_seq=[]
        w_list=[]        
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
            if date+horizon<len(df_tot_m.iloc[:-12]):               
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)              
                pred_seq.append(seq.tolist()) 
                w_list.append(wei)                
        tot_seq=pd.DataFrame(pred_seq)       
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        tot_seq['weights'] = w_list
        val_sce = tot_seq.groupby('Cluster').apply(lambda x: (x.iloc[:, :12].T * x['weights']).sum(axis=1) / x['weights'].sum())
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:     
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))      
df_sf_2_w_clu = pd.concat(pred_tot_pr,axis=1)
df_sf_2_w_clu.columns=country_list['name']
df_sf_2_w_clu = df_sf_2_w_clu.fillna(0)

# (e) Original approach in Chadefaux
with open('Results/test1_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]     
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])
        pred_seq=[]
        w_list=[]
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())  
                w_list.append(wei)
        tot_seq=pd.DataFrame(pred_seq)
        weights = np.array(w_list).reshape(-1, 1)
        weights/weights.sum()
        pred_ori = np.average(tot_seq, axis=0, weights=weights.flatten())
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pd.Series(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()))
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))       
df_sf_1_chad = pd.concat(pred_tot_pr,axis=1)
df_sf_1_chad.columns=country_list['name']
df_sf_1_chad = df_sf_1_chad.fillna(0)
    
with open('Results/test2_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
df_input_sub=df_input.iloc[:-12]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]     
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])
        pred_seq=[]
        w_list=[]
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-12]):                              
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())        
                w_list.append(wei)
        tot_seq=pd.DataFrame(pred_seq)
        weights = np.array(w_list).reshape(-1, 1)
        weights/weights.sum()
        pred_ori = np.average(tot_seq, axis=0, weights=weights.flatten())
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pd.Series(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()))
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))        
df_sf_2_chad = pd.concat(pred_tot_pr,axis=1)
df_sf_2_chad.columns=country_list['name']
df_sf_2_chad = df_sf_2_chad.fillna(0)

# (d) Change window size    
for k_win,wind in enumerate([0,1,3]):
    with open(f'Results/test1_wind{wind}.pkl', 'rb') as f:
        dict_m = pickle.load(f)
    pred_tot_min=[]
    pred_tot_pr=[]
    horizon=12
    h_train=10
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))          
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)
    
    with open(f'Results/test2_wind{wind}.pkl', 'rb') as f:
        dict_m = pickle.load(f) 
    df_input_sub=df_input.iloc[:-12]
    pred_tot_min=[]
    pred_tot_pr=[]
    horizon=12
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
            l_find=dict_m[df_input.columns[coun]]
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]           
            pred_seq=[]                     
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
                if date+horizon<len(df_tot_m.iloc[:-12]):               
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)              
                    pred_seq.append(seq.tolist())                
            tot_seq=pd.DataFrame(pred_seq)       
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()
            pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:     
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))        
    df_sf_2_sub= pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']    
    df_sf_2_tot.append(df_sf_2_sub) 
    
# (f) Random matches
with open('Results/test1_random.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]          
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())        
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq=tot_seq.dropna()
        tot_seq = tot_seq[np.isfinite(tot_seq).all(1)]
        tot_seq=tot_seq.iloc[:5]
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()   
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))        
df_sf_1_random = pd.concat(pred_tot_pr,axis=1)
df_sf_1_random.columns=country_list['name']

with open('Results/test2_random.pkl', 'rb') as f:
    dict_m = pickle.load(f)
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]           
        pred_seq=[]                     
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
            if date+horizon<len(df_tot_m.iloc[:-12]):               
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)              
                pred_seq.append(seq.tolist())                
        tot_seq=pd.DataFrame(pred_seq)   
        tot_seq=tot_seq.dropna()
        tot_seq = tot_seq[np.isfinite(tot_seq).all(1)]
        tot_seq=tot_seq.iloc[:5]
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:     
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))     
df_sf_2_random= pd.concat(pred_tot_pr,axis=1)
df_sf_2_random.columns=country_list['name']    

# Evaluation
df_sf_1_tot.append(df_sf_1_chad)
df_sf_2_tot.append(df_sf_2_chad)
df_sf_1_tot.append(df_sf_1_random)
df_sf_2_tot.append(df_sf_2_random)
df_sf_1_tot.append(df_sf_1_w_clu)
df_sf_2_tot.append(df_sf_2_w_clu)
mse_list_sub=[]
for n_k in range(len(df_sf_2_tot)):
    err_sf_pr=[]
    err_sub=[]
    for i in range(len(df_input.columns)):   
        err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_tot[n_k].iloc[:,i]))
        err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_tot[n_k].iloc[:,i]))
    err_sf_pr = np.array(err_sf_pr)
    err_sub = np.array(err_sub)
    mse_list_sub.append(np.log((err_sub+1)/(err_sf_pr+1)))
de_list_sub=[]
for n_k in range(len(df_sf_2_tot)):
    d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_tot[n_k])
    d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_tot[n_k])
    d_sub = np.concatenate([d_sub,d_sub2])
    de_list_sub.append(np.log((d_nn+1)/(d_sub+1)))
    
    means = [arr.mean() for arr in mse_list_sub]
cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in mse_list_sub]
de_means = [arr.mean() for arr in de_list_sub]
de_cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in de_list_sub]

x_ticks = ["(a) Top match", "Top 3 matches", "Top 5 matches","(b) cut=2","cut=4","cut=5","(c) dist=0.2","dist=0.3","dist=0.4","dist=0.5","dist=0.75","dist=1","(d) win=0","win=1","win=3","(e) Chadefaux (2022)","(f) Random","(g) Weighted cluster"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=24)
ax1.axhline(0, linestyle='--', color='gray')
ax2.errorbar(x=x_ticks, y=de_means, yerr=np.squeeze(de_cis), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
ax2.axhline(0, linestyle='--', color='gray')
ax1.tick_params(axis='both', which='major')
ax2.tick_params(axis='both', which='major')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("out/robust.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Include null model in VIEWS
df_p_views_1 = pd.read_csv('Datasets/views1.csv',index_col=0)
df_p_views_2 = pd.read_csv('Datasets/views2.csv',index_col=0)
c_y=0
c_n=0
l_mse_zero_not=[]
l_mse_zero=[]
pair_pred=[]
for coun in range(len(df_input.columns)): 
    if (df_input.iloc[-34:-24,coun]==0).all():
        if not (df_input.iloc[-24:-12,coun]==0).all():
            mse_v = mean_squared_error(df_input.iloc[-24:-12,coun], df_p_views_1.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-24:-12,coun], pd.Series([0]*12))
            l_mse_zero_not.append(np.log((mse_v+1)/(mse_z+1)))
            pair_pred.append([df_input.iloc[-24:-12,coun].sum(),df_p_views_1.iloc[:,coun].sum()])
            c_y+=1
        else:
            c_n+=1
            mse_v = mean_squared_error(df_input.iloc[-24:-12,coun], df_p_views_1.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-24:-12,coun], pd.Series([0]*12))
            l_mse_zero.append(np.log((mse_v+1)/(mse_z+1)))
            
for coun in range(len(df_input.columns)): 
    if (df_input.iloc[-22:-12,coun]==0).all():
        if not (df_input.iloc[-12:,coun]==0).all():
            mse_v = mean_squared_error(df_input.iloc[-12:,coun], df_p_views_2.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-12:,coun], pd.Series([0]*12))
            l_mse_zero_not.append(np.log((mse_v+1)/(mse_z+1)))
            pair_pred.append([df_input.iloc[-12:,coun].sum(),df_p_views_2.iloc[:,coun].sum()])
            c_y+=1
        else:
            c_n+=1
            mse_v = mean_squared_error(df_input.iloc[-24:-12,coun], df_p_views_1.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-24:-12,coun], pd.Series([0]*12))
            l_mse_zero.append(np.log((mse_v+1)/(mse_z+1)))
            
means_z = [arr.mean() for arr in [np.array(l_mse_zero_not),np.array(l_mse_zero),np.array(l_mse_zero_not+l_mse_zero)]]
cis_z = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in [np.array(l_mse_zero_not),np.array(l_mse_zero),np.array(l_mse_zero_not+l_mse_zero)]]
x_ticks = ['Non-flat future (14\%)', 'Flat future (86\%)','All']
plt.figure(figsize=(10, 5))
plt.errorbar(x=x_ticks, y=means_z, yerr=np.squeeze(cis_z), fmt='o', markersize=8, color='black', ecolor='black')
plt.ylabel("Mean squared error (log-ratio)")
plt.axhline(0,linestyle='--')
plt.xlim(-0.5,2.5)
plt.yticks([-0.025,-0.02,-0.015,-0.01,-0.005,0,0.005],[-0.025,-0.02,-0.015,-0.01,-0.005,0,0.005])
ax1.tick_params(axis='both', which='major')
ax2.tick_params(axis='both', which='major')
plt.savefig("out/flat.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Tune hyperparameters of the Shape finder
err_sf_pr_tot=[]   
de_list=[]  
for file in ['Results/test_0.1.pkl','Results/test_0.5.pkl','Results/test_1dist.pkl']: 
    with open(file, 'rb') as f:
        dict_m = pickle.load(f) 
    for thres_hold in [12,6,4]:
        pred_tot_min=[]
        pred_tot_pr=[]
        horizon=12
        h_train=10
        df_input_sub=df_input.iloc[:-36]
        cluster_dist=[]
        for coun in range(len(df_input_sub.columns)):
            if not (df_input_sub.iloc[-h_train:,coun]==0).all():
                inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
                l_find=dict_m[df_input.columns[coun]]
                tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
                pred_seq=[]
                co=[]
                deca=[]
                scale=[]
                for col,last_date,mi,ma,somme in tot_seq:
                    date=df_tot_m.iloc[:-36].index.get_loc(last_date)
                    if date+horizon<len(df_tot_m.iloc[:-36]):
                        seq=df_tot_m.iloc[:-36].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                        seq = (seq - mi) / (ma - mi)
                        pred_seq.append(seq.tolist())
                        co.append(df_conf[col])
                        deca.append(last_date.year)
                        scale.append(somme)
                tot_seq=pd.DataFrame(pred_seq)
                linkage_matrix = linkage(tot_seq, method='ward')
                clusters = fcluster(linkage_matrix, thres_hold, criterion='distance')
                tot_seq['Cluster'] = clusters
                val_sce = tot_seq.groupby('Cluster').mean()
                pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
                cluster_dist.append(pd.Series(clusters).value_counts().max())
                pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
                pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
                pred_ori=val_sce.loc[pr==pr.min(),:]
                pred_ori=pred_ori.mean(axis=0)
                preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
                pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
            else:
                pred_tot_min.append(pd.Series(np.zeros((horizon,))))
                pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        err_sf_pr=[]
        for i in range(len(df_input.columns)):   
            pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
            err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
        err_sf_pr = np.array(err_sf_pr)
        err_sf_pr_tot.append(err_sf_pr)
        df_sf = pd.concat(pred_tot_pr,axis=1)
        df_sf.columns=country_list['name']
        d_nn = diff_explained(df_input.iloc[-36:-24],df_sf)
        de_list.append(d_nn)

# DE
means = [x.mean() for x in de_list]
std_error = [1.993*x.std()/np.sqrt(len((x))) for x in de_list]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

# MSE
means = [x.mean() for x in err_sf_pr_tot]
std_error = [1.993*x.std()/np.sqrt(len((x))) for x in err_sf_pr_tot]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['gray','lightgray','black',"royalblue","lightskyblue","aliceblue","mediumturquoise","limegreen","forestgreen"])}
plt.rcParams.update(plot_params)
name = ['dist=' + str(i) + ', cut=' + str(j) for i in ['0.1', '0.5', '1'] for j in ['12/1=12','12/2=6','12/3=4']]
fig,ax = plt.subplots(figsize=(12,8))
for i in range(9):
    if name[i]=="dist=0.1, cut=12/3=4":
        plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],label=name[i],s=200,marker="x")
    else: 
        plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],label=name[i],s=150)
plt.xlabel("Accuracy (MSE reversed)")
plt.ylabel("Difference explained (DE)")
plt.xlim(350000000,0)
plt.legend()
plt.savefig("out/valide.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Validate calibration of predictions
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
    
horizon=12
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
check_inside=[]
check_out=[]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr_old = pd.Series(clusters).value_counts(normalize=True).sort_index()
        tot_seq['Cluster'] = clusters
        pred_seq.append((df_input.iloc[-24:-12,coun]-df_input_sub.iloc[-h_train:,coun].min())  /(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min()))
        tot_seq_s=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq_s, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        if clusters[-1] in pr[pr==pr.max()].index.tolist():
            check_inside.append(pr_old.max())
        else:
            check_out.append(pr_old.max())     
    else:
        pass
        
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr_old = pd.Series(clusters).value_counts(normalize=True).sort_index()
        tot_seq['Cluster'] = clusters
        pred_seq.append((df_input.iloc[-12:,coun]-df_input_sub.iloc[-h_train:,coun].min())  /(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min()))
        tot_seq_s=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq_s, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        if clusters[-1] in pr[pr==pr.max()].index.tolist():
            check_inside.append(pr_old.max())
        else:
            check_out.append(pr_old.max())  
    else:
        pass

bins = [0.2, 0.4, 0.6, 0.8, 1]
inside_hist, _ = np.histogram(check_inside, bins=bins)
out_hist, _ = np.histogram(check_out, bins=bins)
inside_percentage = inside_hist / (inside_hist + out_hist + 1e-9) * 100  # Prevent division by zero
out_percentage = out_hist / (inside_hist + out_hist + 1e-9) * 100
width = 0.1
fig,ax = plt.subplots(figsize=(12,8))
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
plt.bar(bin_centers, inside_percentage, width=width, color='gray', alpha=0.5, label='Outcome Inside')
plt.xlabel('Size of majority cluster')
plt.ylabel('Futures (\%) assigned to majority cluster')
plt.xticks(bins)
plt.savefig("out/hist.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# Validate whether size of cluster is associated with prediction performance  
with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

# Get MSE
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)
mse_list_raw = err_sf_pr.copy()
 
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  

# Get MSE
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)
mse_list_true = np.concatenate([mse_list_raw,err_sf_pr],axis=0)

with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.median(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

# Get MSE
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)
mse_list_raw = err_sf_pr.copy()
 
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.median(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  

# MSE
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)
mse_list_med = np.concatenate([mse_list_raw,err_sf_pr],axis=0)

with open('Results/test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.min(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

# Get MAE
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_absolute_percentage_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)
mse_list_raw = err_sf_pr.copy()
 
with open('Results/test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr==pr.min(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
        
# Get MAE
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_absolute_percentage_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)
mse_list_min = np.concatenate([mse_list_raw,err_sf_pr],axis=0)

mse_list_true = np.log(mse_list_true+1)
mse_list_med = np.log(mse_list_med+1)
mse_list_min = np.log(mse_list_min+1)

means = [mse_list_true.mean(),mse_list_med.mean(),mse_list_min.mean()]
std_error = [1.96*mse_list_true.std()/np.sqrt(len(mse_list_true)),1.96*mse_list_med.std()/np.sqrt(len(mse_list_med)),1.96*mse_list_min.std()/np.sqrt(len(mse_list_min))]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

fig,ax = plt.subplots(figsize=(12,8))
for i in range(3):
    plt.scatter(i,mean_de["mean"][i],color="black",s=150)
    plt.plot([i,i],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="black")
plt.xticks([0,1,2],['Maximum cluster size','Median cluster size','Minimum cluster size'],fontsize=20)
plt.ylabel('Mean absolute percentage error (log-ratio)',fontsize=20)
plt.xlim(-0.5,2.5)
plt.savefig("out/mape.jpeg",dpi=400,bbox_inches="tight")
plt.show()







