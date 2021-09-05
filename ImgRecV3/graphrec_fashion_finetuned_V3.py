########################   Libraries ######################
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow  import keras
import itertools
import random
import json
import gzip
import re
import pickle
from PIL import Image
import requests
from io import BytesIO
from ast import literal_eval
import multiprocessing as mp
from sklearn.metrics import ndcg_score
user_encoder = preprocessing.LabelEncoder()
item_encoder = preprocessing.LabelEncoder()

num_users=None
num_items=None
print(tf.__version__)
print('*** Fashion Dataset vs DVBPR 40 epochs Fine tuned ResNet features LR 0.00005 reg 0.0001 load the weights savecheck6 with reg on image layer 0.00001***')
############################      Read Data     #####################################

def Read_Data(path1,path3,path4):
  #Reviews=pd.DataFrame()
  Reviews=pd.read_csv(path1,sep=',')
  Test=pd.read_csv(path3,sep=',')
  Valid=pd.read_csv(path4,sep=',')
  return Reviews,Test,Valid

def Read_Meta(path2):
  with open(path2, 'rb') as handle:
    meta_Data = pickle.load(handle)
  return meta_Data

###########################     Sampling training instances #########################
def Sampling_training_instances(Data,Num_of_Negatives,batch_size):
  users=[]
  items=[] 
  labels=[]
  for m in range(0,batch_size):
    ############## Add positive sample ################
    i=randint(0, num_users-1) # generate new user randomly
    select=Data.loc[Data['reviewerID']==i]
    users.append(i)
    pos=randint(0, len(select)-1)
    items.append(select.iloc[pos,1])
    labels.append(1.0)
    ################### Add negative samples ##########
    for j in range(0,Num_of_Negatives):
      ########## generate negative sample item ID #####  
      Neg=randint(0, num_items-1)
      while Neg in select['productId']:
        Neg=randint(0, num_items-1)
      users.append(i)
      items.append(Neg)
      labels.append(0.0)
    #print('user: ',i,' item positive: ',select.iloc[pos,1],' item negative: ',Neg)
  return np.asarray(users), np.asarray(items), np.asarray(labels)

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
def Get_Image_Features(items_Train): # ready Features 
  Images=[]
  for i in items_Train: 
    #print('i ....   ',i,'  ',New_Meta[i]) 
    element=New_Meta[i]
    #print('  f  ',len(element))
    Features=element[3][0]
    Images.append(Features)
  return np.asarray(Images)

################################### Get image features ########################
def Get_Items_Images(items_Train): # read raw images
  path='/ImageRec/Datasets/Fashion/Fashion_Raw_Images/'
  Images=[]
  for i in range(0,len(items_Train)): 
    img_ind=np.where(New_Keys == items_Train[i])
    #print(img_ind[0])
    string_name=String_item_names[img_ind[0][0]]
    try:
      img=Image.open(path+string_name+'.jpg')
    except IOError:
      img=Image.new('RGB', (244,244), (255, 255, 255))
    newsize=(244,244)
    img=img.resize(newsize)
    if (img.mode) =='RGB':
      img = np.expand_dims(img, axis=0)
    else:
      img=img.convert('RGB')
      img = np.expand_dims(img, axis=0) 
    image=preprocess_input(img)
    image=np.asarray(image)
    image=np.reshape(image,(244,244,3))
    Images.append(image)
    '''
    img=Image.new('RGB', (160,160), (255, 255, 255))
    img = np.expand_dims(img, axis=0)
    image=preprocess_input(img)
    image=np.asarray(image)
    image=np.reshape(image,(160,160,3))
    '''
  return np.asarray(Images)

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
    test_itemID=Data.iloc[i,1]
    items.append(test_itemID)
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
path1='/ImageRec/Datasets/Fashion/Fashion_Training_Reviews.csv'
path2='/ImageRec/Datasets/Fashion/Fashion_Dict_ResNetFeatures.pickle'
path3='/ImageRec/Datasets/Fashion/Fashion_test_Reviews.csv'
path4='/ImageRec/Datasets/Fashion/Fashion_validation_Reviews.csv'

Reviews, Test_Reviews, Validation_Reviews=Read_Data(path1,path3,path4)
#print('read data done ....', len(Reviews))
#print(Reviews.columns)
Meta=Read_Meta(path2)

Meta_keys=[]
Reviews['reviewerID']=user_encoder.fit_transform(Reviews['reviewerID']) 
Test_Reviews['reviewerID']=user_encoder.transform(Test_Reviews['reviewerID']) 
Validation_Reviews['reviewerID']=user_encoder.transform(Validation_Reviews['reviewerID']) 
num_users=Reviews['reviewerID'].nunique()
print('num_users: ',num_users)
#for y in Meta.keys():
#  Meta_keys.append(y)
String_item_names=list(Meta.keys())
New_Keys=item_encoder.fit_transform(list(Meta.keys()))
#Meta['New_key'] = New_Keys
Reviews['productId']=item_encoder.transform(Reviews['productId'])
Test_Reviews['productId']=item_encoder.transform(Test_Reviews['productId'])
Validation_Reviews['productId']=item_encoder.transform(Validation_Reviews['productId'])
num_items_unique=Reviews['productId'].nunique()
#print('num_itemss: ',num_items_unique)


num_items=len(Meta.keys())
print('num_itemss Meta: ',num_items)
Data_Training=Reviews

users_Test, items_Test, labels_Test=Prepare_Testing_Data(Test_Reviews)
users_Validation, items_Validation, labels_Validation=Prepare_Testing_Data(Validation_Reviews)
#print('Training amount ....',len(users_Validation), len(items_Validation))

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

########################## Functional API ######################################
embedding_u=30
embedding_i=30
image_embedding=150
Number_of_users = num_users
Number_of_items = num_items
Features_Size=2048
reg_lambda=0.0001
reg_img=0.00001
########################## Inputs ##############################################
############################# users ###########################################
encoded_users = keras.layers.Input(shape=(),dtype=tf.int32)
encoded_users_onehot=tf.one_hot(encoded_users,Number_of_users) # changing users to one hot encoded vectors
#encoded_users=tf.reshape(encoded_users,[-1,Number_of_users])
############################# items ###########################################
encoded_items = keras.layers.Input(shape=(),dtype=tf.int32)
encoded_items_onehot=tf.one_hot(encoded_items,Number_of_items) # changing items to one hot encoded vectors
############################# Images ###########################################
images = keras.layers.Input(shape=((244,244,3)),dtype=tf.float32)
#list_of_labels = keras.layers.Input(shape=(),dtype=tf.float32) 
#***************************** IMAGE Model Fine tuning *************************
#Image_model=tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,include_top=False,weights=None)
Image_model=tf.keras.models.load_model("/ImageRec/Datasets/ResNet_Modelsv")

Image_model.trainable = True
# Fine-tune from this layer onwards
fine_tune_at = 126
number_of_layers=len(Image_model.layers)
print('number of layers: ',len(Image_model.layers))
# Freeze all the layers before the `fine_tune_at` layer
count=0
for layer in range(0,fine_tune_at):
  Image_model.layers[layer].trainable =  False
  count=count+1
print('Make non trainable.....',count)

images_features=Image_model(images)
image_features_embedding=keras.layers.Dense(image_embedding,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg_img))(images_features)

#*******************************************************************************
###################### Layers ##################################################
items_with_images=keras.layers.concatenate([encoded_items_onehot,image_features_embedding],axis=1)
#*****************************************************************
############ user embedding #############################
user_embedding=keras.layers.Dense(embedding_u,kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))(encoded_users_onehot)
############ item embedding #############################
item_embedding=keras.layers.Dense(embedding_i,kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))(items_with_images)

##################### Model ####################################################
dot_user_item = tf.multiply(user_embedding, item_embedding) 
logits=tf.reduce_sum(dot_user_item,1)
model = keras.Model(inputs=[encoded_users,encoded_items,images],outputs=logits)

#print(model.summary())

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
########################   NDCG   ###########################################
def NDCG(users_scores,labels_Test):
  ndcg=ndcg_score(labels_Test,users_scores,k=10)
  print('NDCG = ',ndcg)
  return ndcg

def CustomLoss():
    def loss(y_true, y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return loss

###################### running the model #################
opt =keras.optimizers.Adam(0.00005)
#opt = tfa.optimizers.Lookahead(opt)

num_of_epochs=130
size_of_minibatch_Train=256
for i in range(0,num_of_epochs):
    if i% 1==0:
       print('running epoch... ',i) 
    step_Train_losses=[] 
    Number_of_minibatches_Train=np.floor(len(Data_Training)/size_of_minibatch_Train)
    #print('Number_of_minibatches_Train: ',Number_of_minibatches_Train)
    for j in range(0,int(Number_of_minibatches_Train)):
      users_Train, items_Train, labels_Train=Sampling_training_instances(Data_Training,1,size_of_minibatch_Train)
      #Images_Features=Get_Image_Features(items_Train) 
      Images=Get_Items_Images(items_Train) 
      #print(Images.shape)
      H = model.train_on_batch([users_Train, items_Train,Images],labels_Train)
      step_Train_losses.append(H)
    print('Train loss... ',np.mean(step_Train_losses))
    ############ validation ########################################
    
    if i% 15==0:
        users_val_scores=[] 
        all_val_losses=[]
        auc_val=0
        cc_val=0
        validation_predictions=[]
        for l in range(0,num_users):  
            Validation_Images=Get_Items_Images(items_Validation[l])
            users_Validation_feed= [users_Validation[l]]* len(items_Validation[l]) 
            #print(users_Validation_feed)
            #print(np.asarray(items_Validation[l]).shape)
            Val_batch_scores=model.predict_on_batch([np.asarray(users_Validation_feed),np.asarray(items_Validation[l]),Validation_Images])
            validation_predictions.append(Val_batch_scores)
            cc_val=cc_val+1
            tmpans_val=0
            count_val=0
            for j in range(1,len(Val_batch_scores)): #sample
               if Val_batch_scores[0]>Val_batch_scores[j]:
                  tmpans_val+=1
               count_val+=1       
            tmpans_val/=float(count_val)
            auc_val+=tmpans_val
            #print('loss :  ',val_loss) 
            users_val_scores.append(Val_batch_scores)                                                                   
        #print('Validation Done ... ',np.mean(all_val_losses))   
        Hit_Rate(users_val_scores,labels_Validation,10)   
        NDCG(users_val_scores,labels_Validation)
        auc_val/=float(cc_val)
        print('validation AUC= ',auc_val)
        print("**...Validation...**") 
        
model.save_weights('/ImageRec/Fashion/my_checkpoint_vs_DVBPR')
print('model saved...')
#print(model.summary()) 
##### Testing ########

Test_predictions=[]
auc=0
cc=0
for h in range(0,num_users): 
      Test_Images=Get_Items_Images(items_Test[h])
      users_Test_feed= [users_Test[h]]* len(items_Test[h]) 
      Test_batch_scores=model.predict_on_batch([np.asarray(users_Test_feed),np.asarray(items_Test[h]),Test_Images])
      cc=cc+1
      tmpans=0
      count=0
      for j in range(1,len(Test_batch_scores)): #sample
         if Test_batch_scores[0]>Test_batch_scores[j]:
             tmpans+=1
         count+=1       
      tmpans/=float(count)
      auc+=tmpans
      Test_predictions.append(Test_batch_scores)
print("**...Test...**")
Hit_Rate(Test_predictions,labels_Test,10)   
NDCG(Test_predictions,labels_Test) 
auc/=float(cc)
print('Test AUC= ',auc)
