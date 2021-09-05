# -*- coding: utf-8 -*-

########################   Libraries ######################
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from random import randint
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
import itertools
import random
import json
import gzip
import re
import pickle
#from PIL import Image
#import requests
from io import BytesIO
from ast import literal_eval
import multiprocessing as mp
from sklearn.metrics import ndcg_score
user_encoder = preprocessing.LabelEncoder()
item_encoder = preprocessing.LabelEncoder()

num_users=None
num_items=None
print(tf.__version__)
print('*************** Men Dataset with pretrained ResNet Features LR=0.0001 and lambda=0.1*************************')
#!python3 --version

############################      Read Data     #####################################

def Read_Data(path1,path3,path4):
  #Reviews=pd.DataFrame()
  Reviews=pd.read_csv(path1,sep=',')
  Test=pd.read_csv(path3,sep=',')
  val=pd.read_csv(path4,sep=',')
  return Reviews,Test,val

def Read_Meta(path2):
  with open(path2, 'rb') as handle:
    meta_Data = pickle.load(handle)
  return meta_Data

###########################     Sampling training instances #########################
def Sampling_training_instances(Data,Num_of_Negatives,batch_size):
  users=[]
  items=[] 
  labels=[]
  #num_users=Data['userId'].nunique()
  #print('num_users: ',num_users)
  #num_items=Data['movieId'].nunique()
  #print('num_items: ',num_items)
  for m in range(0,batch_size):
    ############## Add positive sample ################
    i=randint(0, num_users-1) # generate new user randomly
    select=Data.loc[Data['reviewerID']==i]
    users.append(i)
    pos=randint(0, len(select)-1)
    items.append(select.iloc[pos,1])
    labels.append(1)
    ################### Add negative samples ##########
    for j in range(0,Num_of_Negatives):
      ########## generate negative sample item ID #####  
      Neg=randint(0, num_items-1)
      while Neg in select['productId']:
        Neg=randint(0, num_items-1)
      users.append(i)
      items.append(Neg)
      labels.append(0)
    #print('user: ',i,' item positive: ',select.iloc[pos,1],' item negative: ',Neg)
  return users, items, labels

############### Cut Data into minibatches ######################################
def cut_item_list(items,size_chunk):
  start=0
  next=0
  items_chunks=[]
  while (next+size_chunk)<len(items):
    next=next+size_chunk
    items_chunks.append(items[start:next])
    start=start+size_chunk
  items_chunks.append(items[next:len(items)]) # appending the remaining amount
  return items_chunks

################################### Get image features ########################
import re
def Get_Image_Features(items_Train):
  Images=[]
  for i in items_Train: 
    #print('i ....   ',i,'  ',New_Meta[i]) 
    element=New_Meta[i]
    #print('  f  ',len(element))
    Features=element[3][0]
    Images.append(Features)
  return Images

#######################  Prepare testing Data ##################################
def Prepare_Testing_Data(Data,Num_of_Negatives=100):
  users=[]
  items_All=[] 
  labels_All=[]
  #num_users=Data['userId'].nunique()
  print('num_users: ',num_users)
  #num_items=Data['movieId'].nunique()
  #print('num_items: ',num_items)
  for i in range(0,len(Data)):
    items=[] 
    labels=[]
    ############## Add positive sample ################
    test_userID=Data.iloc[i,0]
    select=Data.loc[Data['reviewerID']==test_userID]
    #print(select.head) 
    users.append(test_userID)
    #pos=randint(0, len(select)-1) 
    test_itemID=Data.iloc[i,1]
    items.append(test_itemID)
    #index = Data[ (Data['reviewerID']==i) & (Data['productId']==pos_item) ].index
    #Data.drop(index , inplace=True)
    #Data.drop(index )
    #print('item_dropped: ',index)
    labels.append(1)
    ################### Add negative samples ##########
    for j in range(0,Num_of_Negatives):
      ########## generate negative sample item ID #####  
      Neg=randint(0, num_items-1)
      while Neg in select['productId']:
        Neg=randint(0, num_items-1)
      items.append(Neg)
      labels.append(0) 
    items_All.append(items)
    labels_All.append(labels)
  #print('shape of items all: ',len(items_All),'  and element ',len(items_All[0]))
  #print(len(users))
  return users, items_All, labels_All

##############################       Main      #################################
path1='/ImageRec/Datasets/Men/Men_Training_Reviews.csv'
path2='/ImageRec/Datasets/Men/Men_Dict_WithResNetFeatures.pickle'
path3='/ImageRec/Datasets/Men/Men_test_Reviews.csv'
path4='/ImageRec/Datasets/Men/Men_validation_Reviews.csv'

Reviews, Test_Reviews,val_Reviews=Read_Data(path1,path3,path4)
#print('read data done ....', len(Reviews))
#print(Reviews.columns)
Meta=Read_Meta(path2)

Meta_keys=[]
Reviews['reviewerID']=user_encoder.fit_transform(Reviews['reviewerID']) 
Test_Reviews['reviewerID']=user_encoder.transform(Test_Reviews['reviewerID'])
val_Reviews['reviewerID']=user_encoder.transform(val_Reviews['reviewerID']) 
num_users=Reviews['reviewerID'].nunique()
print('num_users: ',num_users)
#for y in Meta.keys():
#  Meta_keys.append(y)
New_Keys=item_encoder.fit_transform(list(Meta.keys()))
#Meta['New_key'] = New_Keys
Reviews['productId']=item_encoder.transform(Reviews['productId'])
Test_Reviews['productId']=item_encoder.transform(Test_Reviews['productId'])
val_Reviews['productId']=item_encoder.transform(val_Reviews['productId'])
num_items_unique=Reviews['productId'].nunique()
print('num_itemss: ',num_items_unique)


num_items=len(Meta.keys())
print('num_itemss Meta: ',num_items)
Data_Training=Reviews

users_Test, items_Test, labels_Test=Prepare_Testing_Data(Test_Reviews)
users_val, items_val, labels_val=Prepare_Testing_Data(val_Reviews)

#print('Training amount ....',users_Test, items_Test)

#size_of_minibatch_Test=100
#users_Test, items_Test, labels_Test,Number_of_minibatches_Test=cut_minibatches(users_T, items_T, labels_T,size_of_minibatch_Test)
#print(users_Test[0])

######################## Create new Dictionary with the transformed keys #######
#print(Meta.keys())
New_Meta={}
counter=0
for x in Meta.values():
  for j in x:
    if New_Keys[counter] in New_Meta.keys():
      New_Meta[New_Keys[counter]].append([j])
    else:
      New_Meta[New_Keys[counter]]=[j]
  counter=counter+1
#f=literal_eval(New_Meta[0])
#print(len(New_Meta[0][3][0]))

#print(New_Meta[0])

##############################  Model structure ################################ 
embedding_u=20
embedding_i=20
image_embedding=150 # was 100
Number_of_users = num_users
Number_of_items = num_items
Features_Size=2048
reg_lambda=0.1
###############################################################################
#Number_of_items=tf.compat.v1.placeholder(tf.int32)
############################# users ###########################################
encoded_users = tf.compat.v1.placeholder(tf.int32, [None])
encoded_users_onehot=tf.one_hot(encoded_users,Number_of_users) # changing users to one hot encoded vectors
#encoded_users=tf.reshape(encoded_users,[-1,Number_of_users])
############################# items ###########################################
encoded_items = tf.compat.v1.placeholder(tf.int32, [None])
encoded_items_onehot=tf.one_hot(encoded_items,Number_of_items) # changing items to one hot encoded vectors
#encoded_items=tf.reshape(encoded_items,[-1,Number_of_items])

############################# Images ###########################################
images_features = tf.compat.v1.placeholder(tf.float32, [None,Features_Size])

list_of_labels = tf.compat.v1.placeholder(tf.float32, [None]) 
#***************************************************************************************************
############ image embedding #############################
#*****************************************************************
items_with_images=tf.concat([encoded_items_onehot,images_features],axis=1)
#*****************************************************************
############ user embedding #############################
user_embedding=tf.compat.v1.layers.dense(encoded_users_onehot,units=embedding_u,activation=tf.nn.crelu,kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))

############ item embedding #############################
item_embedding=tf.compat.v1.layers.dense(items_with_images,units=embedding_i,activation=tf.nn.crelu,kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))

dot_user_item = tf.multiply(user_embedding, item_embedding) 
logits=tf.compat.v1.reduce_sum(dot_user_item,1)
#scores=tf.reshape(scores, [-1,1 ])
#scores_sigmoid=tf.sigmoid(logits) # to return 
################ Loss function ########################## 
Loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=list_of_labels, logits=logits) 
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005).minimize(Loss)

#######################   Evaluation ###########################################
#######################   Hit rate   ###########################################
def Hit_Rate(users_scores,labels_Test,k): 
    hits=0
    for u in range(0,len(users_scores)):
        list_scores=users_scores[u]
        list_labels=labels_Test[u] 
        #print(list_scores.index(max(list_scores))) 
        list_scores_sorted, list_labels_sorted = zip(*sorted(zip(list_scores, list_labels), reverse=True))
        index_of_positive=list_labels_sorted.index(max(list_labels_sorted))
        #print('Label :  ',max(list_labels_sorted),'  index : ',index_of_positive)
        if index_of_positive<k:
          hits=hits+1
    print('Avg hit rate= ',(hits/num_users))

####################### AUC 2 ##############################################
####################### AUC Fast ###########################################
#Data_Training, Meta, Test_Reviews
'''
def AUC2():
  auc=0
  cc=0
  chunk_size=1000
  #All_items_embedding=[]
  All_items_embedding=np.empty((0,40))
  # run session for all items embeddings
  Meta_items=list(New_Meta.keys())
  #items_chunks=cut_item_list(Meta_items,chunk_size)
  print('total number of users: ',num_users)
  users_Ids=np.arange(num_users)
  All_Users_embeddings=session.run(user_embedding, feed_dict={encoded_users:np.asarray(users_Ids)}) # get items embedding
  for i in range(0,num_users):
    #print('currently in user: ',i)
    Images=Get_Image_Features(items_Test[i])
    items_embedding=session.run(item_embedding, feed_dict={encoded_items:items_Test[i],images_features:Images})
    # run session for user embedding 
    user_i_embedding=All_Users_embeddings[i]
    # get all user items (+ve items)
    user_i_embedding=np.transpose(user_i_embedding)
    all_user_i_scores=items_embedding.dot(user_i_embedding)
    cc=cc+1
    tmpans=0
    count=0
    for j in range(1,len(all_user_i_scores)): #sample
        if all_user_i_scores[0]>all_user_i_scores[j]:
            tmpans+=1
        count+=1       
    tmpans/=float(count)
    auc+=tmpans
  auc/=float(cc)
  return auc
'''
#######################   NDCG   ###########################################
def NDCG(users_scores,labels_Test):
  ndcg=ndcg_score(labels_Test,users_scores,k=10)
  print('NDCG = ',ndcg)
  return ndcg

#Index(['reviewerID', 'productId'], dtype='object')
#Index(['productId', 'title', 'related', 'Image_Features'], dtype='object')
############################ Training Session ##################################\
with tf.compat.v1.Session() as session:
  session.run(tf.compat.v1.initialize_all_variables())
  num_of_epochs=120
  size_of_minibatch_Train=512
  epochs_Train_losses=[] 
  for i in range(0,num_of_epochs):
    if i% 1==0:
       print('running epoch... ',i) 
    step_Train_losses=[] 
    Number_of_minibatches_Train=np.floor(len(Data_Training)/size_of_minibatch_Train)
    print('Number_of_minibatches_Train: ',Number_of_minibatches_Train)
    for j in range(0,int(Number_of_minibatches_Train)):
      users_Train, items_Train, labels_Train=Sampling_training_instances(Data_Training,1,size_of_minibatch_Train)
      Images_Features=Get_Image_Features(items_Train)
      loss,_ = session.run([Loss,train_op], feed_dict={encoded_users: users_Train, encoded_items:items_Train,images_features:Images_Features,
                                                       list_of_labels: labels_Train}) 
      #print('Finished minibatch...',j)
      step_Train_losses.append(np.mean(loss))                                                                   
    epochs_Train_losses.append(np.mean(step_Train_losses))
    print('training done ...',epochs_Train_losses[-1] ) 
    print("*** Validation ***")
    if i% 5==0:
      users_val_scores=[] 
      all_val_losses=[]
      for l in range(0,num_users): 
        val_Images_Features=Get_Image_Features(items_val[l])
        users_val_feed= [users_val[l]]* len(items_val[l])  
        val_loss,score = session.run([Loss,logits], feed_dict={encoded_users:users_val_feed , encoded_items:items_val[l],images_features:val_Images_Features,
                                                        list_of_labels: labels_val[l]})  
        all_val_losses.append(np.mean(val_loss))
        #print('loss :  ',val_loss) 
        users_val_scores.append(score)                                                                   
      print('Validation Done ... ',np.mean(all_val_losses))   
      Hit_Rate(users_val_scores,labels_val,10)   
      NDCG(users_val_scores,labels_val)
  print("*** Testing ***")
  users_scores=[] 
  all_test_losses=[]
  auc=0
  cc=0
  for l in range(0,num_users): 
     Test_Images_Features=Get_Image_Features(items_Test[l])
     users_Test_feed= [users_Test[l]]* len(items_Test[l])  
     test_loss,score = session.run([Loss,logits], feed_dict={encoded_users:users_Test_feed , encoded_items:items_Test[l],images_features:Test_Images_Features,
                                                        list_of_labels: labels_Test[l]})  
     all_test_losses.append(np.mean(test_loss))
     #print('loss :  ',test_loss) 
     cc=cc+1
     tmpans=0
     count=0
     for j in range(1,len(score)): #sample
        if score[0]>score[j]:
            tmpans+=1
        count+=1       
     tmpans/=float(count)
     auc+=tmpans
     users_scores.append(score)                                                                   
  print('Testing Done ... ',np.mean(all_test_losses))   
  Hit_Rate(users_scores,labels_Test,10)   
  NDCG(users_scores,labels_Test) 
  auc/=float(cc)
  print('AUC value =',auc)
#################################  plot training loss over epochs ##############
#fig = plt.figure(figsize = (11,8)) ###specify image size
#ax = fig.add_subplot(1,1,1)  
#ax.set_xticks(np.arange(1,num_of_epochs,1))
#ax.plot(epochs_Train_losses)

