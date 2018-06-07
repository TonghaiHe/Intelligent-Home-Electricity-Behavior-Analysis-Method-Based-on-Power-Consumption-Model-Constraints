# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd


def read_csv(path):
    # 读取一个path，各个电器数据分成单独的文件存放，返回读取结果，读取后未做处理
    data = pd.read_csv(path, header=0, error_bad_lines=False)
    """
    use_feature = ["localminute", "dataid", "use"]
    air_feature = ["localminute", "dataid", "air"]
    dishwasher_feature = ["localminute", "dataid", "dishwasher"]
    furnace_feature = ["localminute", "dataid", "furnace"]
    refrigerator_feature = ["localminute", "dataid", "refrigerator"]
    others_feature = ["localminute", "dataid", "others"]

    use_data = data[use_feature]
    air_data = data[air_feature]
    dishwasher_data = data[dishwasher_feature]
    furnace_data = data[furnace_feature]
    refrigerator_data = data[refrigerator_feature]
    others_data = data[others_feature]

    use_data.to_csv("use_data.csv", index=False)
    air_data.to_csv("air_data.csv", index=False)
    dishwasher_data.to_csv("dishwasher_data.csv", index=False)
    furnace_data.to_csv("furnace_data.csv", index=False)
    refrigerator_data.to_csv("refrigerator_data.csv", index=False)
    others_data.to_csv("others_data.csv", index=False)
    """
    return data


def produce_train_data(raw_data, category):
    # 产生训练数据，输入原始数据和电器类别字符串，返回约70%数据构成的矩阵，包括总数据
    data = raw_data[category]
    data = np.array(data)
    data = data[0:54910]
    data = data.reshape(190, 289)
    data = data.T
    return data


def produce_test_data(raw_data, category):
    # 产生测试数据，输入原始数据和电器类别字符串，返回约100%数据构成的矩阵，包括总数据
    data = raw_data[category]
    data = np.array(data)
    data = data.reshape(264, 289)
    data = data.T
    return data

def cos(vector1,vector2):  
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return 1e-8  
    else:  
        return dot_product / ((normA*normB)**0.5)    #计算余弦相似度函数
        
 
def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab
 
def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den
    
    

def Initialization(G, X, n, k):
    # 参数中X为训练矩阵，即单独电器矩阵，G为总数据矩阵，n为分解电器矩阵的维度，k为矩阵数量
    # 该函数返回A,B,W,H H为近似的对角矩阵
    shape_X = np.shape(X)
    shape_G = np.shape(G)
    A = [0] * (k + 1)
    B = [0] * (k + 1)
    W = [0] * (k + 1)
    for i in range(1, k + 1):
        A[i] = np.random.rand(n, shape_X[1])
        B[i] = np.random.rand(shape_X[0], n)
        W[i] = np.random.rand(n, shape_G[1])
    H = np.zeros((shape_G[1], shape_X[1]))
    for j in range(0, min(shape_G[1], shape_X[1])):
        H[j, j] = 1
    return A, B, W, H


def train_model_withlaplace(A, B, W, H, G, X, Z, E, n, k, alpha, beta, gamma, lanmda):
    D=[0]*(k+1)
    for i in range(1,k+1):
        d=np.sum(Z[i],0)
        D[i]=np.diag(d)
    iter = 1
    while (iter <= 5000):
        for i in range(1, k + 1):
            temp = [0] * (k + 1)
            for j in range(1, k + 1):
                temp[j] = np.dot(B[j], W[j])
            temp_sum = sum(temp)
            W[i] = W[i] * np.sqrt((np.dot(B[i].T, G) + lanmda * np.dot(A[i], H.T)+E[i] * np.dot(W[i],Z[i])) / \
                                  (lanmda * np.dot(np.dot(W[i], H), H.T) + alpha * W[i] + np.dot(B[i].T, temp_sum) \
                                   +E[i]*np.dot(W[i],D[i])+ 1e-8))
            #print(W[i][2, 3])
            temp2 = [0] * (k + 1)
            for l in range(1, k + 1):
                temp2[l] = np.dot(B[l], W[l])
            temp2_sum = sum(temp2)
            B[i] = B[i] * np.sqrt((np.dot(G, W[i].T) + np.dot(X[i], A[i].T)) / \
                                  (np.dot(np.dot(B[i], A[i]), A[i].T) + beta * B[i] + np.dot(temp2_sum, W[i].T) \
                                   + 1e-8))
            #print(B[i][2, 3])
            A[i] = A[i] * np.sqrt((np.dot(B[i].T, X[i]) + np.dot(W[i], H)) / (
            np.dot(np.dot(B[i].T, B[i]), A[i]) + A[i] + gamma * A[i] + 1e-8))
            #print(A[i][2, 3])
        iter += 1
        print(iter)
        # for m in range(1,k+1):
        # np.savetxt("B_" + str(m), B[m], fmt='%.8f')
        # np.savetxt("A_" + str(m), A[m], fmt='%.8f')
        # np.savetxt("W_" + str(m), W[m], fmt='%.8f')
    return A, B, W


def prediction(B, W, k):
    new_G = [0] * (k + 1)
    for i in range(1, k + 1):
        new_G[i] = np.dot(B[i], W[i])
        np.savetxt("new_G_" + str(i), new_G[i], fmt="%.8f")
    return new_G


def performance(G, new_G, k):    #原论文中的评测标准 去掉训练集，从188列开始
    denominator = np.sum(G[0])
    numerator = 0
    for i in range(1, k + 1):
        shape = np.shape(G[i])
        for j in range(0, shape[1]):
            numerator = numerator + min(np.sum(G[i][:, j]), np.sum(new_G[i][:, j]))
    Acc = numerator / denominator
    return Acc

def performance2(G, new_G, k):  #去掉了训练集合的原论文方法评测
    denominator = np.sum(G[0][:, 188:])
    numerator = 0
    for i in range(1, k + 1):
        shape = np.shape(G[i])
        for j in range(188, shape[1]):
            numerator = numerator + min(np.sum(G[i][:, j]), np.sum(new_G[i][:, j]))
    Acc = numerator / denominator
    return Acc

def performance3(G, new_G, k):  #两个大矩阵相减，没有0去掉训练集
    denominator = np.sum(G[0])
    numerator = 0
    for i in range(1, k + 1):
        numerator = numerator + np.abs(np.sum(G[i] - new_G[i]))
    err = numerator / denominator
    Acc = 1 - err
    return Acc



def performance4(G, new_G, k): #去掉训练集的评测，减少了循环次数，去掉了训练集
    denominator = np.sum(G[0][:, 188:])
    numerator = 0
    for i in range(1, k + 1):
        numerator = numerator + abs(np.sum(G[i][:, 188:]) - np.sum(new_G[i][:, 188:]))
    Acc = 1 - numerator / denominator
    return Acc

def performance5(G,new_G,k):  #比例
    denominator = sum(G[0][:, 188:])
    difference=[0]*(k+1)
    ave_diff=[0]*(k+1)
    for i in range(1,k+1):
        raw_numerator=sum(G[i][:, 188:])
        new_numerator=sum(new_G[i][:,188:])
        raw_percent=list(map(lambda x:x[0]/x[1] ,zip(raw_numerator,denominator)))
        new_percent=list(map(lambda x:x[0]/x[1],zip(new_numerator,denominator)))
        file = open("percent_file", "a")
        file.write("applicance_"+str(i)+"_raw_percent="+str(raw_percent) + '\n')
        file.write("applicance_"+str(i)+"new_percent=" + str(new_percent) + '\n')
        file.close()
        difference[i]=list(map(lambda x:x[0]-x[1],zip(raw_percent,new_percent)))
    for i in range(1,k+1):
        ave_diff[i]=sum(difference[i])/len(difference[i])
    return ave_diff

def run(alpha,beta,gamma,lanmda,n,h_l): 
    #alpha=0.01 #迭代1的参数
    #beta=0.01 #迭代2的参数
    #gamma=0.01 #迭代3的参数
    #lanmda=0.01 #迭代1的参数
    #n=70 #单独电器分解后的维度
    file_path = "smart_grid_dataset.csv" 
    raw_data = read_csv(file_path)
    category = ["use", "air", "dishwasher", "furnace", "refrigerator", "others"]
    X = [0] * len(category)
    G = [0] * len(category)
    Z = [0] * len(category)
    E = [0] * len(category)
    k = len(category) - 1  # 单个电器的数量  本数据集为5
    for i in category:
        X[category.index(i)] = produce_train_data(raw_data, i)
        # X为公式中的X，X[0]为use X[1]为air 都是矩阵289x190
    for j in category:
        G[category.index(j)] = produce_test_data(raw_data, j)
    for m in range(1, k + 1):
        np.savetxt("G_" + str(m), G[m], fmt='%.8f')
        # X为公式中的G，X[0]为use X[1]为air 都是矩阵289x264
    init_A, init_B, init_W, init_H = Initialization(G[0], X[1], n, k)
    for nn in range(1,k+1):
        E[nn]=1
        #先把E设为全1矩阵
    E[1]=h_l[0]*E[1]
    E[3]=h_l[0]*E[3]
    E[2]=h_l[1]*E[2]
    E[4]=h_l[1]*E[4]
    E[5]=0.01*E[5]
	
    


    """
    总用电数据形成字典
    """
    data=pd.read_csv("smart_grid_dataset.csv",header=0,error_bad_lines=False)
    use_feature=["dataid","use"]
    air_feature=["dataid","air"]
    dishwasher_feature=["dataid","dishwasher"]
    furnace_feature=["dataid","furnace"]
    refrigerator_feature=["dataid","refrigerator"]
    others_feature=["dataid","others"]
    
    use_data=data[use_feature]
    air_data=data[air_feature]
    dishwasher_data=data[dishwasher_feature]
    furnace_data=data[furnace_feature]
    refrigerator_data=data[refrigerator_feature]
    others_data=data[others_feature]
    
    use_data.to_csv("use_data.csv",index=False)
    air_data.to_csv("air_data.csv",index=False)
    dishwasher_data.to_csv("dishwasher_data.csv",index=False)
    furnace_data.to_csv("furnace_data.csv",index=False)
    refrigerator_data.to_csv("refrigerator_data.csv",index=False)
    others_data.to_csv("others_data.csv",index=False)
    
    vec_id=np.array(use_data["dataid"])
    vec_id=vec_id.reshape(264,289)
    vec_id=vec_id[:,1]   #用户ID 列表
    
    use_only=use_data['use']
    use_only=np.array(use_only)
    use_only=use_only.reshape(264,289)
    #总用电数据矩阵 一列为一周
    
    use_dict=dict(zip(vec_id,use_only)) #用户ID和数据构成的字典 数据为一周的用电数据列表
    
    use_sum=sum(use_only.T) #列表 数据之和
    
    use_sum_dict=dict(zip(vec_id,use_sum)) #数据ID 和数据之和形成的字典
    
    sorted_use_sum_dict=sorted(use_sum_dict.items(),key=lambda asd: asd[1])
    #数据ID 和数据之和形成的字典 排序后  形成的列表 列表内是元组
    
    for i in range(len(sorted_use_sum_dict)):
        sorted_use_sum_dict[i]=list(sorted_use_sum_dict[i])
    #讲列表内的元组转换为列表
  
    
    
    """
    空调用电数据形成关系矩阵
    """
    
    vec_id_single=vec_id[:190]  #用户ID 列表
    air_only=air_data['air']
    air_only=np.array(air_only)
    
    air_only=air_only.reshape(264,289)
    air_only=air_only[190]
    
    #总用电数据矩阵 一行为一周
    
    air_dict=dict(zip(vec_id,air_only)) #用户ID和数据构成的字典 数据为一周的空调用电数据列表
    Z[1]=np.ones((np.shape(G[0])[1],np.shape(G[0])[1]))
    temp=np.shape(X[1])
    for i in range(0,temp[1]):
        for j in range(0,temp[1]):
            Z[1][i][j]=cos(X[1][:,i],X[1][:,j]) #先生成已知数据的关系矩阵
    for l in range(190,np.shape(G[0])[1]):
        temp2=use_sum_dict.get(vec_id[l]) #190个家庭后第L个用户的总用电数据
        for i in sorted_use_sum_dict: #在排序之后的总用电字典中找到新用户的位置
            if temp2==i[1]:
                while(sorted_use_sum_dict.index(i)==len(sorted_use_sum_dict)-1):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-2][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                    break  #如果位于列表的最后一项，找到该位置之前两个相邻的ID值
                while(sorted_use_sum_dict.index(i)==0):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+2][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+1][0]
                    break  #如果位于列表的第一项，找到该位置之后两个相邻的ID值
                while(sorted_use_sum_dict.index(i)>0 and sorted_use_sum_dict.index(i)<len(sorted_use_sum_dict)-1):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+1][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                    break  #找到该位置上下两个相邻的ID值
                break
        vec_id=list(vec_id)
        index1=vec_id.index(select_id_1) 
        index2=vec_id.index(select_id_2) #通过ID值在未排序的ID列表中找到索引位置
        Z[1][:,l]=(Z[1][:,index1]+Z[1][:,index2])/2
        Z[1][l]=(Z[1][index1]+Z[1][index2])/2 #在已经建立的关系矩阵中找到相应的位置，并为未知家庭赋值

    

    """
    洗碗机用电数据形成关系矩阵
    """
   
    vec_id_single=vec_id[:190]  #用户ID 列表
    dishwasher_only=dishwasher_data['dishwasher']
    dishwasher_only=np.array(dishwasher_only)
    dishwasher_only=dishwasher_only.reshape(264,289)
    dishwasher_only=dishwasher_only[:190]
    #总用电数据矩阵 一行为一周
    dishwasher_dict=dict(zip(vec_id,dishwasher_only)) #用户ID和数据构成的字典 数据为一周的空调用电数据列表
    Z[2]=np.ones((np.shape(G[0])[1],np.shape(G[0])[1]))
    temp=np.shape(X[2])
    for i in range(0,temp[1]):
        for j in range(0,temp[1]):
            Z[2][i][j]=cos(X[2][:,i],X[2][:,j]) #先生成已知数据的关系矩阵
    for l in range(190,np.shape(G[0])[1]):
        temp2=use_sum_dict.get(vec_id[l]) #190个家庭后第L个用户的总用电数据
        for i in sorted_use_sum_dict: #在排序之后的总用电字典中找到新用户的位置
            if temp2==i[1]:
                while(sorted_use_sum_dict.index(i)==len(sorted_use_sum_dict)-1):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-2][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                    break  #如果位于列表的最后一项，找到该位置之前两个相邻的ID值
                while(sorted_use_sum_dict.index(i)==0):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+2][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+1][0]
                    break  #如果位于列表的第一项，找到该位置之后两个相邻的ID值
                while(sorted_use_sum_dict.index(i)>0 and sorted_use_sum_dict.index(i)<len(sorted_use_sum_dict)-1):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+1][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                    break  #找到该位置上下两个相邻的ID值
        
        vec_id=list(vec_id)
        index1=vec_id.index(select_id_1) 
        index2=vec_id.index(select_id_2) #通过ID值在未排序的ID列表中找到索引位置
        Z[2][:,l]=(Z[2][:,index1]+Z[2][:,index2])/2
        Z[2][l]=(Z[2][index1]+Z[2][index2])/2 #在已经建立的关系矩阵中找到相应的位置，并为未知家庭赋值

   

    """
    冰箱用电数据形成关系矩阵
    """
    vec_id_single=vec_id[:190]  #用户ID 列表
    refrigerator_only=refrigerator_data['refrigerator']
    refrigerator_only=np.array(refrigerator_only)
    
    refrigerator_only=refrigerator_only.reshape(264,289)
    refrigerator_only=refrigerator_only[:190]
    #总用电数据矩阵 一行为一周
    refrigerator_dict=dict(zip(vec_id,refrigerator_only)) #用户ID和数据构成的字典 数据为一周的空调用电数据列表
    Z[4]=np.ones((np.shape(G[0])[1],np.shape(G[0])[1]))
    temp=np.shape(X[4])
    for i in range(0,temp[1]):
        for j in range(0,temp[1]):
            Z[4][i][j]=cos(X[4][:,i],X[4][:,j]) #先生成已知数据的关系矩阵
    """
    ave=sum(Z[3])/temp[1]
    for l in range(190,np.shape(G[0])[1]):
        Z[3][:,l]=ave
        Z[3][l]=ave
    """
    for l in range(190,np.shape(G[0])[1]):
        temp2=use_sum_dict.get(vec_id[l]) #190个家庭后第L个用户的总用电数据
        for i in sorted_use_sum_dict: #在排序之后的总用电字典中找到新用户的位置
            if temp2==i[1]:
                while(sorted_use_sum_dict.index(i)==len(sorted_use_sum_dict)-1):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-2][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                    break  #如果位于列表的最后一项，找到该位置之前两个相邻的ID值
                while(sorted_use_sum_dict.index(i)==0):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+2][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+1][0]
                    break  #如果位于列表的第一项，找到该位置之后两个相邻的ID值
                while(sorted_use_sum_dict.index(i)>0 and sorted_use_sum_dict.index(i)<len(sorted_use_sum_dict)-1):
                    select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))+1][0]
                    select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                    break  #找到该位置上下两个相邻的ID值
                break
                
        vec_id=list(vec_id)
        index1=vec_id.index(select_id_1) 
        index2=vec_id.index(select_id_2) #通过ID值在未排序的ID列表中找到索引位置
        Z[4][:,l]=(Z[4][:,index1]+Z[4][:,index2])/2
        Z[4][l]=(Z[4][index1]+Z[4][index2])/2 #在已经建立的关系矩阵中找到相应的位置，并为未知家庭赋值
    
    """
	火炉
	"""
    vec_id_single=vec_id[:190]  #用户ID 列表
    furnace_only=furnace_data['furnace']
    furnace_only=np.array(furnace_only)
    
    furnace_only=furnace_only.reshape(264,289)
    furnace_only=furnace_only[:190]
    #总用电数据矩阵 一行为一周
    furnace_dict=dict(zip(vec_id,furnace_only)) #用户ID和数据构成的字典 数据为一周的空调用电数据列表
    Z[3]=np.ones((np.shape(G[0])[1],np.shape(G[0])[1]))
    temp=np.shape(X[3])
    for i in range(0,temp[1]):
        for j in range(0,temp[1]):
            Z[3][i][j]=cos(X[3][:,i],X[3][:,j]) #先生成已知数据的关系矩阵
    for l in range(190,np.shape(G[0])[1]):
        temp2=use_sum_dict.get(vec_id[l]) #190个家庭后第L个用户的总用电数据
        for i in sorted_use_sum_dict: #在排序之后的总用电字典中找到新用户的位置
            if temp2==i[1]:
                select_id_1=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-2][0]
                select_id_2=sorted_use_sum_dict[int(sorted_use_sum_dict.index(i))-1][0]
                break  #找到该位置上下两个相邻的ID值
        vec_id=list(vec_id)
        index1=vec_id.index(select_id_1) 
        index2=vec_id.index(select_id_2) #通过ID值在未排序的ID列表中找到索引位置
        Z[3][:,l]=(Z[3][:,index1]+Z[3][:,index2])/2
        Z[3][l]=(Z[3][index1]+Z[3][index2])/2 #在已经建立的关系矩阵中找到相应的位置，并为未知家庭赋值
    
	

	
    """
    临时程序
    """    
    Z[5]=np.zeros((np.shape(G[0])[1],np.shape(G[0])[1]))
    
        
        
                                    
        
    

    
    """
    for l in range(1,k+1):
        E[l]=np.ones((np.shape(G[0])[1],np.shape(G[0])[1]))
        temp=np.shape(X[l])
        for i in range(0,temp[1]):
            for j in range(0,temp[1]):
                E[l][i][j]=cos(X[l][:,i],X[l][:,j]) #先生成已知数据的关系矩阵
    
    """
    
                
    
    





    print(n)
    A, B, W = train_model_withlaplace(init_A, init_B, init_W, init_H, G[0], X, Z, E, n, k, alpha, beta, gamma, lanmda)
    new_G = prediction(B, W, k)
    Acc = performance(G, new_G, k)
    Acc2 = performance2(G, new_G, k)
    Acc3 = performance3(G, new_G, k)
    Acc4 = performance4(G, new_G, k)
    Acc5 = performance5(G, new_G, k)
    filename='0.01_70_'+'Eh='+str(h_l[0])+'_'+'El='+str(h_l[1])+'.txt'
    file=open(filename,"a")
    file.write('Acc='+str(Acc)+'\n'+'Acc2='+str(Acc2)+'\n'+'Acc3='+str(Acc3)+'\n'+'Acc4='+str(Acc4)+'\n'+'Acc5='+str(Acc5)+'\n\n\n')
    file.close()
    return 0
