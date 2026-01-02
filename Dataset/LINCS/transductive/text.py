# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 05:29:10 2021

@author: Administrator
"""
import numpy as np
import pandas as pd
import csv
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import scipy as sp
import networkx as nx
import scipy.sparse as sp
from operator import itemgetter

import networkx as nx
from networkx.algorithms.bipartite import biadjacency_matrix
#from openpyxl import load_workbook

y_true1=pd.read_csv('C:/Users/Administrator/Desktop/陈国俊/1与-1.csv',header=None)
np.savetxt("C:/Users/Administrator/Desktop/陈国俊/1与-1.txt", y_true1,fmt='%s', newline='\n')

#txt = np.loadtxt('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/GCN/feature[3508, 847].txt') 
#txtDF = pd.DataFrame(txt) 
#txtDF.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/GCN/feature[3508, 847].csv',index=False)


#txt = np.loadtxt('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/MDA-GCNGS/label[334].txt') 
#txtDF = pd.DataFrame(txt) 
#txtDF.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/MDA-GCNGS/label[334].csv',index=False)

#print(txt)




'''
#####￥￥￥￥￥￥￥￥￥匹配￥￥￥￥￥￥￥￥######

label = pd.read_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/GCN/label[702].csv')
feature = pd.read_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/GCN/feature[3508, 847].csv')
#ff = pd.read_csv('text/ID.csv')

print(label)
print(feature)
print(type(label))
print(type(feature))

#print(df)
data = pd.merge(label,feature,left_on=['label'], 
                right_on = ['label'])##right_on

print('data',data)
print('data',data.shape)
#print('stp_ID',data)

#lastdata = pd.merge(data,cf,left_on=['disease'], 
#                right_on = ['id'])##right_on

#print('lastdata:',lastdata)
data.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/分类散点图数据/GCN/emb_subg[702,847].csv',index=False)

'''










#l = [1]
#print(len(l))  

#df = pd.read_csv('1.csv')##读取数据
#print(df.head(3))   ##查看前几行数据
#cgi = df["cgi"]   ###提取cgi
#cgi.to_csv('C:/Users/Administrator/Desktop/cgilie.csv',index=False) ##保存cgi列数据


#data = pd.read_csv('关联数据2744＋enst＋统一格式.csv')
#res=data['0'].str.split("'",expand = True)
#print(res)
#res.to_csv('assoication2.csv',index=False)
#data.to_csv('mutation.csv',index=False)
#ces=res['0'].str.split(":'",expand = True)
#print(ces)


'''
#####￥￥￥￥￥￥￥￥￥匹配￥￥￥￥￥￥￥￥######

df = pd.read_csv('/Users/komatsu/Desktop/图采样tcbb/审稿意见/gene-pathway/association 1754(1).csv')
cf = pd.read_csv('/Users/komatsu/Desktop/图采样tcbb/审稿意见/gene-pathway/chem_gene_3.csv')
#ff = pd.read_csv('text/ID.csv')

#print(df)
data = pd.merge(df,cf,left_on=['Drug'],
                right_on = ['# ChemicalName'])##right_on

print('stp_ID',data)

#lastdata = pd.merge(data,cf,left_on=['disease'], 
#                right_on = ['id'])##right_on

#print('lastdata:',lastdata)
data.to_csv('/Users/komatsu/Desktop/图采样tcbb/审稿意见/gene-pathway/chem_gene_4.csv',index=False)

'''



#********数据提取**********
#data = pd.read_csv('/Users/komatsu/Desktop/图采样tcbb/审稿意见/gene-pathway/association 1754.csv')
#res=data['Mutation'].str.split("_",expand = True)
#print(res)
#res = pd.read_csv('/Users/komatsu/Desktop/图采样tcbb/审稿意见/gene-pathway/association 1754(1).csv')
#res=data['1'].str.split("'",expand = True)

#cf=res['1'].str.split("', '",expand = True)
#print(cf)
#res.to_csv('/Users/komatsu/Desktop/图采样tcbb/审稿意见/gene-pathway/association 1754(1).csv',index=False)

###########错误的求余弦相似度以及相似度的代码！！！！！！！！！！######

#import numpy as np
#import pandas as pd
#import csv
#from pandas import DataFrame
#from sklearn.metrics.pairwise import cosine_similarity
#df= pd.read_csv('求相似性/last all_drug_feature_184x231.csv',)
#print(df)


#m1_similarity = cosine_similarity(df)
#print(m1_similarity)
#m1_similarity=(m1_similarity+1)/2
#print(m1_similarity) 

#np.savetxt("求相似性/M_SM归一化.txt", m1_similarity,fmt='%s', newline='\n')


##########归一化#######

#data=pd.read_csv('求相似性/jaccard_similarity/drug_similarity.csv')

#m1_similarity=(data - data.min())/(data.max() - data.min())

#print(m1_similarity) 

#np.savetxt("求相似性/jaccard_similarity/D_SD归一化.txt", m1_similarity,fmt='%s', newline='\n')


##########归一化csv 转换txt######
#data=pd.read_csv('C:/Users/Administrator/Desktop/图采样data/求相似性/cosine_similarity/drug_Similarity.csv',header=None)####一定要删除索引，只保留数据！！！！！！！！！！
#data=pd.read_csv('求相似性/new2_jaccard_Similarity/drug_similarity.csv',header=None)
#min_max_scaler = preprocessing.MinMaxScaler() 
#X_minMax = min_max_scaler.fit_transform(data)
#print('X_minMax',X_minMax)

#pd.DataFrame(X_minMax).to_csv("C:/Users/Administrator/Desktop/图采样data/求相似性/cosine_similarity/drug_Similarity1.csv")
#np.savetxt("求相似性/new2_jaccard_similarity/M_SM归一化.txt", X_minMax,fmt='%s', newline='\n')
#np.savetxt("求相似性/new2_jaccard_Similarity/D_SM归一化.txt", X_minMax,fmt='%s', newline='\n')


#########先归一化再求相似性##########
#data=pd.read_csv('求相似性/last all_drug_feature_184x231.csv',header=None)####一定要删除索引，只保留数据！！！！！！！！！！
#data=pd.read_csv('求相似性/all mutation feature 661x248.csv',header=None)
#min_max_scaler = preprocessing.MinMaxScaler() 
#X_minMax = min_max_scaler.fit_transform(data)
#print('X_minMax',X_minMax)
#pd.DataFrame(X_minMax).to_csv("求相似性/new2_jaccard_similarity/mutation_normalize.csv")

#np.savetxt("求相似性/new1_jaccard_similarity/M_SM归一化.txt", X_minMax,fmt='%s', newline='\n')
#np.savetxt("求相似性/new1_jaccard_Similarity/D_SM归一化.txt", X_minMax,fmt='%s', newline='\n')









#txt = np.loadtxt('C:/Users/Administrator/Desktop/图采样data/last data/D_SM.txt') 
#txtDF = pd.DataFrame(txt) 
#txtDF.to_csv('C:/Users/Administrator/Desktop/图采样data/last data/D_SM1.csv',index=False)








#########删除全0的列######

#data= pd.read_csv('all_drug_feature_186.csv')
#print(data)
#idx = np.argwhere(np.all(df[..., :] == 0, axis=0))
#a2 = np.delete(df, idx, axis=1)
#print(a2)

#idx=np.where(~data.any(axis=0))[0]
#print(idx)
#print(np.where(~data.any(axis=0))[0])
#data= np.delete(data,np.where(~data.any(axis=1))[0], axis=0)

#df.ix[~(df==0).all(axis=1), :]












#####去重#####

#data = pd.read_csv('5数据汇总/last data5 对齐1187.csv')
#print(data)
#cf=data.drop_duplicates(['drug'])
#print(cf)
#cf.to_csv('5数据汇总/药物格式统一/only_drug101.csv',index=False)







##3*********根据一列数据转换成多行*********
#data = pd.read_csv('5数据汇总/drug3.csv')
#data=data.set_index(['1'])
#data=data.set_index(['genes','feature_names','mutation_type','association','disease'])
#data=data.stack()
#data=data.reset_index()
#print(data)
#data.to_csv('5数据汇总/drug4.csv',index=False)



#data = pd.read_csv('last data/pairs/生成三列/661x184.csv')
#data=data.set_index(['0'])
#data=data.stack()###unstack()
#cf=data.reset_index()
#print(cf)


#print(data.stack().unstack())

#cf.to_csv('last data/pairs/生成三列/text1.csv',index=False)



#**************分裂***********

#data = pd.read_csv('association 步骤/对比数据/对比表1754.csv')
#res=data['Drug'].str.split("_",expand = True) ##分成多列
#data = df[df['18'].str.startswith('['']')]
#data = df[df['18'].str[0:3]!="['']"]   ##判断并提取
#print(res)
#print(res)

#res.to_csv('association 步骤/对比数据/对比表1754(2).csv',index=False)


#print(res)
#res=df["cgi"].str.split("':",expand = True)  ###,"'info':"  ##分为n列
#df1=df['cgi'] == 'info'
#print(res.head(4))
#print(res)
#res.to_csv('C:/Users/Administrator/Desktop/cgilie_fen.csv',index=False) ##保存cgi列数据

#print(df[df['18'].str.startswith('CSQM')])

#print(df1.shape)

#print(df)


# ************xlsx格式提取文件***********
#data=[]
#for row in range(2,df.max_row+1): #原工作表从第2行开始读取
#    model = df['A' + str(row)].value.split()['Alteration'] #直接分割，并取第一个单词，存入model变量
#    data.append(model) #将分割下来的单词存入data列表

#i=2
#for word in data:
#    df.cell(row=i,column=2).value=word #将分割出的单词从data列表中取出，从第二行开始写入Excel表第二列
#    i+=1
#df.save('cgilie_Alteration')






#cf=cgi[0]         ##cgi 第一个单元格中的元素
#cf["alteration"]



#############合并数据############

#df = pd.read_csv('last data/删除争议敏感抗性关联/all mutation feature 673x248.csv')
#cf=df['GENE'].str.cat(df['Code'],sep='_')
#print(cf)
#cf.to_csv('last data/删除争议敏感抗性关联/mutation_name.csv',index=False)



###############生成矩阵#######
#data=np.arange(1,121625).reshape(661,184)
#data_df = pd.DataFrame(data)
#print(data_df)
#data_df.to_csv('last data/pairs/661x184.csv',index=False)





#######混淆矩阵  seaborn的热度图########
 
#import seaborn as sns;
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#sns.set()
#y_true = ["cat", "dog", "cat", "cat", "dog", "rebit"]
#y_pred = ["dog", "dog", "rebit", "cat", "dog", "cat"]
#C2= confusion_matrix(y_true, y_pred, labels=["dog", "rebit", "cat"])
#sns.heatmap(C2,annot=True)


#txt = np.loadtxt('C:/Users/Administrator/Desktop/图采样有向图代码/test data/old labels.txt') 
#txtDF = pd.DataFrame(txt) 
#txtDF.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/old labels.csv',index=False)


#old

#txt = np.loadtxt('C:/Users/Administrator/Desktop/图采样有向图代码/test data/参数/class_arr.txt') 
#txtDF = pd.DataFrame(txt) 
#txtDF.to_csv('C:/Users/Administrator/Desktop/图采样有向图代码/test data/参数/class_arr.csv',index=False)

#one_hot = torch.tensor([[0,0,1],[0,0,1],[1,0,0]])
#print(one_hot)
#label = torch.topk(one_hot, 1)[1].squeeze(1)
#print(label)
    
#pt=[0.6651, 0.5334]
#print(np.argmax(pt, axis=1))




#######交叉表  邻接矩阵####

#mat = pd.read_csv('C:/Users/Administrator/Desktop/图采样data/last data/邻接矩阵/drug_mutation_pairs.csv')
#data=mat.values.tolist()
#print(data)
#print(df)
#G = nx.Graph(mat)
#print(G)

#print(data[183:])
#for i in data[2]:

    



#cf = open("C:/Users/Administrator/Desktop/图采样data/last data/邻接矩阵/drug2.txt", 'r')
#G = nx.Graph()
#print(type(mat))
#data=mat.values.tolist()
#print(data)
#print(type(data))




#data = [('a', 'developer'),
#         ('b', 'tester'),
#        ('b', 'developer'),
#         ('c','developer'),
#         ('c', 'architect')]
#print(type(data))






#A=nx.adjacency_matrix(df)
#print(A.todense())


#df = pd.crosstab(df.drug,df.mutation) 
#print(df)
#cf.to_csv('C:/Users/Administrator/Desktop/图采样data/last data/邻接矩阵/邻接矩阵drug1.csv',index=False)











