#In this we aim to compute mean feature vectors for a business Id
#We have feature vectors for each image
#All the feature vectors of the images that correspond to a businedd id are used to compute the mean feature vector
#We executed this on [train/test]_biz_fc[6,7,8]features.csv
import numpy as np
import pandas as pd 
import h5py
import time

#Data and Caffe paths
path_to_data = '/Users/samantha/yelp/'
path_to_caffe = '/Users/samantha/dev/caffe/'

train_photo_to_biz = pd.read_csv(path_to_data+'train_photo_to_biz_ids.csv')
train_labels = pd.read_csv(path_to_data+'train.csv').dropna()
#Processing labels
train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
train_labels.set_index('business_id', inplace=True)
bids = train_labels.index.unique()

## Get the image features
file6 = h5py.File(path_to_data+'train_image_fc6features.h5','r')
train_image_features = np.copy(f['feature'])
file6.close()

## Normalizing FC6 Vectors
normalized_fc6 = preprocessing.normalize(train_image_features_fc6)

## Load FC7 image features
file7 = h5py.File(data_root+'train_image_fc7features.h5','r')
train_image_features_fc7 = np.copy(file7['feature'])
file7.close()

## Normalizing FC7 Vectors
normalized_fc7 = preprocessing.normalize(train_image_features_fc7)

#concatenating normalized fc6 and fc7
concatenated67 = np.concatenate((normalized_fc6,normalized_fc7),axis=1)

t= time.time()
## For each business, compute a feature vector 
df = pd.DataFrame(columns=['business','label','feature vector'])
index = 0
for bid in bids:  
    
    label = train_labels.loc[bid]['labels']
    image_index = train_photo_to_biz[train_photo_to_biz['business_id']==bid].index.tolist()
    folder = path_to_data+'train_photo_folders/'  
#mean calculation    
    features = concatenated[image_index]
    mean_feature =list(np.mean(features,axis=0))

    df.loc[index] = [bid, label, mean_feature]
    index+=1
    if index%100==0:
        print "Done with Business : ", index, "Elapsed Time : ", "{0:.1f}".format(time.time()-t), "sec"

with open(path_to_data+"train_biz_fc6_norm_features.csv",'w') as f:  
    df.to_csv(f, index=False)

#Similar process for test features 
test_photo_to_biz = pd.read_csv(path_to_data+'test_photo_to_biz.csv')
bids = test_photo_to_biz['business_id'].unique()

## Load fc6 image features
fc6 = h5py.File(data_root+'test_image_fc6features.h5','r')
image_filenames_fc6 = list(np.copy(fc6['photo_id']))
image_filenames_fc6 = [name.split('/')[-1][:-4] for name in image_filenames_fc6]  
image_features_fc6 = np.copy(fc6['feature'])
fc6.close()

## Normalizing FC6 Vectors
normalized_fc6 = preprocessing.normalize(image_features_fc6)

## Load fc7 image features
fc7 = h5py.File(data_root+'test_image_fc7features.h5','r')
image_filenames_fc7 = list(np.copy(fc7['photo_id']))
image_filenames_fc7 = [name.split('/')[-1][:-4] for name in image_filenames_fc7]  
image_features_fc7 = np.copy(fc7['feature'])
fc7.close()

## Normalizing FC7 Vectors
normalized_fc7 = preprocessing.normalize(image_features_fc7)

#concatenating normalized fc6 and fc7 
concatenated67 = np.concatenate((normalized_fc6,normalized_fc7),axis=1)
del normalized_fc6
del normalized_fc7

df = pd.DataFrame(columns=['business','feature vector'])
index = 0
t = time.time()

for biz in bids:     
    
    image_ids = test_photo_to_biz[test_photo_to_biz['business_id']==biz]['photo_id'].tolist()  
    image_index = [image_filenames.index(str(x)) for x in image_ids]
     
    folder = path_to_data+'test_photo_folders/'            
#mean calculation    
    features = concatenated[image_index]
    mean_feature =list(np.mean(features,axis=0))

    df.loc[index] = [biz, mean_feature]
    index+=1
    if index%1000==0:
        print "Done with Buisness : ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

with open(path_to_data+"test_biz_fc67_norm_features.csv",'w') as f:  
    df.to_csv(f, index=False)


