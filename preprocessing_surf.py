import numpy as np
import cv2
import os
import sklearn.metrics as sm
from surf_image_processing import func
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import random
from sklearn.naive_bayes import GaussianNB as nb

path="train"
label=0
img_descs=[]
y=[]


def perform_data_split(X, y, training_idxs, test_idxs):

    X_train = X[training_idxs]
    X_test = X[test_idxs]


    y_train = y[training_idxs]
    y_test = y[test_idxs]


    return X_train, X_test, y_train , y_test

def train_test_val_split_idxs(total_rows, percent_test):


    row_range = range(total_rows)
    no_test_rows = int(total_rows*(percent_test))
    test_idxs = np.random.choice(row_range, size=no_test_rows, replace=False)
    # remove test indexes
    row_range = [idx for idx in row_range if idx not in test_idxs]
    # remove validation indexes
    training_idxs = [idx for idx in row_range]

    print('Train-test-val split: %i training rows, %i test rows' % (len(training_idxs), len(test_idxs)))
    return training_idxs, test_idxs





def cluster_features(img_descs, training_idxs, cluster_model):

    n_clusters = cluster_model.n_clusters

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)
    print ('%i descriptors before clustering' % all_train_descriptors.shape[0])

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    return X, cluster_model





def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))

def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    print("svm started")
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    calc_accuracy("SVM",y_test,y_pred)
    

def predict_nb(X_train, X_test, y_train, y_test):
    clf = nb()
    print("nb started")
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    calc_accuracy("Naive Bayes",y_test,y_pred)
   
    




#creating desc for each file with label
for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
            for file in files:
                actual_path=path+"\\\\"+dirname+"\\\\"+file
                print(actual_path)
                des=func(actual_path)
                img_descs.append(des)
                y.append(label)
        label=label+1





#finding indexes of test train
y=np.array(y)
training_idxs, test_idxs = train_test_val_split_idxs(len(img_descs), 0.4)

#creating histogram using kmeans minibatch cluster model
X, cluster_model = cluster_features(img_descs, training_idxs, MiniBatchKMeans(n_clusters=150))

#splitting data into test, train using the indexes
X_train, X_test, y_train, y_test = perform_data_split(X, y, training_idxs, test_idxs)





#using classification methods
predict_svm(X_train, X_test,y_train, y_test)
predict_nb(X_train, X_test,y_train, y_test)