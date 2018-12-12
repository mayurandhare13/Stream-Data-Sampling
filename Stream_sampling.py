import keras
import numpy as np
from sklearn.preprocessing import scale
import math
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sl
from pandas import DataFrame, Series
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
import scipy.io as sio
from itertools import cycle
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import time
import random

################### Sport and Daily Activity dataset ####################
##### dataset path ######
fn = 'data/'

### activity labels
acts = ['a01', 'a02', 'a03', 'a04', 'a05','a06', 'a07','a08','a09','a10', 'a11','a12','a13','a14','a15','a16', 'a17', 'a18', 'a19']

### segment labels
segs = ['s01', 's02', 's03', 's04', 's05','s06', 's07','s08','s09','s10', 's11','s12','s13','s14','s15','s16', 's17', 's18', 's19', 's20',
        's21', 's22', 's23', 's24', 's25','s26', 's27','s28','s29','s30', 's31','s32','s33','s34','s35','s36', 's37', 's38', 's39', 's40',
        's41', 's42', 's43', 's44', 's45','s46', 's47','s48','s49','s50', 's51','s52','s53','s54','s55','s56', 's57', 's58', 's59', 's60',]

### sensor names
sensors = ['acc','gyro', 'magnet']

#### the axis index in each file for different sensor locations
locs = {"T": 0, "RA" : 9, "LA" : 18,"RL" : 27,"LL":36}

### user ids in the dataset format
users = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']

###############################################################


def selected_locations(nodes, locs, num_sensors):
    start = 1
    for n in nodes:
        if start == 1:
            start = 0
            idx = np.arange(locs[n], locs[n] + 9)
        else:
            idx = np.concatenate((idx, np.arange(locs[n], locs[n] + 3 * num_sensors)))
    return idx

def extract_features(x):
    numrows = len(x)    # 3 rows in your example
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)
    var = np.var(x, axis = 0)
    median = np.median(x, axis = 0)
    xmax = np.amax(x, axis = 0)
    xmin =np.amax(x, axis = 0)
    p2p = xmax - xmin
    amp = xmax - mean
    s2e = x[numrows-1,:] - x[0,:]
    features = np.concatenate([mean, std, var, median, xmax, xmin, p2p, amp, s2e])#, morph]);
    return features


def data_generate(users_idx, acts_idx, segs_idx, fn, nodes, num_sensors):
    start = 0
    idx = selected_locations(nodes, locs, num_sensors)

    for u in users_idx:
        for s in segs_idx:
            for a in acts_idx:
                filename = fn + acts[a] + '/' + users[u] + '/' + segs[s] + '.txt'
                raw_data = np.genfromtxt(filename, delimiter=',', skip_header=0)
                raw_data = raw_data[:,idx]
                f = extract_features(raw_data)
                if start == 0:
                    fsp = f
                    label = a + 1
                    start = 1
                else: 
                    fsp = np.vstack((fsp, f))
                    label = np.vstack((label, a+1))
    return scale(fsp, axis = 0), label


acts_idx = range(0,19)
users_t = range(0,8)
segs_idx = range(0,60)
nodes = ["T", "LL", "RL", "LA", "RA"]

count = 1

for user in users_t:
    print("read the data for user %d" %(user+1))
    fspace, labels = data_generate([user], acts_idx, segs_idx, fn, nodes, num_sensors = 3)
    fspace

    cache_data_new = pd.DataFrame(fspace)
    cache_label_new = pd.DataFrame(labels)

    if count == 1:
        data_new = pd.DataFrame(fspace)
        label_new = pd.DataFrame(labels)
        count += 1

    else:
        data_new = data_new.append(cache_data_new, ignore_index=True)
        label_new = label_new.append(cache_label_new, ignore_index=True)


label_new = label_new[0]
result = pd.concat([data_new, label_new], axis=1, join='inner')

x_train, x_test, y_train, y_test = train_test_split(data_new, label_new, test_size = 0.9, random_state = 139)


def b_sampling_perc(X, Y, x_test, y_test):
    
    #print(X)
    start = time.process_time()
    count = 0
    beta = {0.001 : 0.01, 0.01 : 0.1, 0.1 : 1}
    
    classifier = Perceptron()
    classifier.fit(X, Y)
    print(np.abs(classifier.decision_function(X)).shape)
    #print(np.argmin(np.abs(classifier.decision_function(X))))

    #sample = X[np.argmin(np.abs(classifier.decision_function(X)), axis=0)]
    #print(sample.shape)
    #print(X.shape)
    
   
    arr = classifier.decision_function(x_test)
  #  print(arr)
    score_train = classifier.score(X, Y)
    print(score_train)
    
#     y_train = pd.DataFrame(y_train)
#     y_test = pd.DataFrame(y_test)

    beta_t_array = []
    beta_score_array = []
    
    for key, value in beta.items():
        print(key, value)
        t = 0
        count_for = 0
        x_train = X.copy()
        y_train = Y.copy()
        
        
        for index, row in x_test.iterrows():

            if (count < 20):
                y_hat = (classifier.predict(x_test.iloc[count_for:(count_for+1),:]))
            
                z = ((random.random()*value) < (key / (key + abs(y_hat))))

                if (z == 1):
                    count += 1   
                    t += 1
                    a = x_test.iloc[count_for:count_for+1,:]       
                    x_train = x_train.append(a)
                    lab = y_test[index]
                    lab_series = pd.Series(lab)
                    y_train = y_train.append(lab_series,ignore_index=True)
            else:
                classifier.fit(x_train, y_train)
                count = 0

            count_for +=1
    
        score_test = classifier.score(x_test, y_test)
        beta_t_array.append(t)
        beta_score_array.append(score_test)
        

    end = time.process_time()
    print("time takes for Perceptron= {} min".format((end - start) / 60))
    
    return(beta_t_array, beta_score_array)

perc_beta_t_array, perc_beta_score_array = b_sampling_perc(x_train, y_train, x_test, y_test)


def sampling_svm(X, Y, x_test, y_test):

    start = time.process_time()
    count = 0
    
    classifier = LinearSVC(C=10**-1, multi_class='crammer_singer')
    classifier.fit(X, Y)
    
    score_train = classifier.score(X, Y)
    print(score_train)
    
    beta = {0.001 : 0.01, 0.01 : 0.1, 0.1 : 1}
    beta_t_array = []
    beta_score_array = []
    
    for key, value in beta.items():
        print(key, value)
        t = 0
        count_for = 0
        x_train = X.copy()
        y_train = Y.copy()
    
        for index, row in x_test.iterrows():
            if (count < 10):
                y_hat = (classifier.predict(x_test.iloc[count_for:(count_for+1),:]))

                z = ((random.random() * value) < (key / (key + abs(y_hat))))

                if (z == True):
                    count += 1
                    t += 1
                    a = x_test.iloc[count_for:count_for+1,:]
                    x_train = x_train.append(a)

                    lab = y_test[index]
                    lab_series = pd.Series(lab)
                    y_train = y_train.append(lab_series,ignore_index=True)
            else:
                classifier.fit(x_train, y_train)
                count = 0

            count_for +=1
    
        score_test = classifier.score(x_test, y_test)
        beta_t_array.append(t)
        beta_score_array.append(score_test)

    end = time.process_time()
    print("time takes = {} min".format((end - start) / 60))

    return(beta_t_array, beta_score_array)

svm_beta_t_array, svm_beta_score_array = sampling_svm(x_train, y_train, x_test, y_test)

plt.plot([0.001, 0.01, 0.1], perc_beta_t_array, color = "red", marker="o", label = "Perceptron")
plt.plot([0.001, 0.01, 0.1], svm_beta_t_array, color = "blue", marker="o", label = "Linear SVM")
plt.title("b-sampling", loc = 'center')
plt.xlabel("b parameters")
plt.ylabel("label requests")
plt.legend(loc='upper left')
plt.savefig("b-sampling.png")
plt.close()


def logistic_sampling_perc(X, Y, x_test, y_test):
    start = time.process_time()
    
    count = 0
    gamma = [1, 2, 4, 8]
    classifier = Perceptron()
    classifier.fit(X, Y)

    score_train = classifier.score(X, Y)
    print(score_train)

    gamma_t_array = []
    gamma_score_array = []
    
    for key, value in enumerate(gamma):
        print(key, value)
        
        t = 0
        count_for = 0
        x_train = X.copy()
        y_train = Y.copy()
        
        for index, row in x_test.iterrows():
            if (count < 10):
                y_hat = (classifier.predict(x_test.iloc[count_for:(count_for+1),:]))
                z = ((random.random()) < math.exp(-value*abs(y_hat)))

                if (z == 1):
                    count += 1   
                    t += 1
                    a = x_test.iloc[count_for:count_for+1,:]       
                    x_train = x_train.append(a)
                    lab = y_test[index]
                    lab_series = pd.Series(lab)
                    y_train = y_train.append(lab_series,ignore_index=True)
                    
                else:
                    classifier.fit(x_train, y_train)
                    count = 0

            count_for +=1

        score_test = classifier.score(x_test, y_test)
        gamma_t_array.append(t)
        gamma_score_array.append(score_test)

    end = time.process_time()
    print("time takes = {} min".format((end - start) / 60))

    return(gamma_t_array, gamma_score_array)

perc_gamma_t_array, perc_gamma_score_array = logistic_sampling_perc(x_train, y_train, x_test, y_test)


def logistic_sampling_svm(X, Y, x_test, y_test):

    start = time.process_time()
    count = 0
    
    classifier = LinearSVC(C=10**-1, multi_class='crammer_singer')
    classifier.fit(X, Y)
    
    score_train = classifier.score(X, Y)
    print(score_train)
    
    gamma = [1, 2, 4, 8]
    gamma_t_array = []
    gamma_score_array = []
    
    for key, value in enumerate(gamma):
        print(key, value)
        t = 0
        count_for = 0
        x_train = X.copy()
        y_train = Y.copy()
    
        for index, row in x_test.iterrows():
            if (count < 10):
                y_hat = (classifier.predict(x_test.iloc[count_for:(count_for+1),:]))

                z = ((random.random()) < math.exp(-value*abs(y_hat)))

                if (z == True):
                    count += 1
                    t += 1
                    a = x_test.iloc[count_for:count_for+1,:]
                    x_train = x_train.append(a)

                    lab = y_test[index]
                    lab_series = pd.Series(lab)
                    y_train = y_train.append(lab_series,ignore_index=True)
            else:
                classifier.fit(x_train, y_train)
                count = 0

            count_for +=1
    
        score_test = classifier.score(x_test, y_test)
        gamma_t_array.append(t)
        gamma_score_array.append(score_test)

    end = time.process_time()
    print("time takes = {} min".format((end - start) / 60))

    return(gamma_t_array, gamma_score_array)

svm_gamma_t_array, svm_gamma_score_array = logistic_sampling_svm(x_train, y_train, x_test, y_test)

plt.plot([1, 2, 4, 8], perc_gamma_t_array, color = "red", marker="o", label = "Perceptron")
plt.plot([1, 2, 4, 8], svm_gamma_t_array, color = "blue", marker="o", label = "Linear SVM")
plt.title("logistic-sampling", loc = 'center')
plt.xlabel("gamma parameters")
plt.ylabel("label requests")
plt.legend(loc='upper left')
plt.savefig("logistic-sampling.png")
plt.close()


def fixed_sampling_perc(X, Y, x_test, y_test):
    start = time.process_time()
    count = 0

    classifier = Perceptron()
    classifier.fit(X, Y)
    score_train = classifier.score(X, Y)
    print(score_train)

    thresholds = [0.04, 0.05, 0.06]
    threshold_t = []
    threshold_t_score = []
    
    for thold in thresholds:
        print(thold)
        t = 0
        count_for = 0
        x_train = X.copy()
        y_train = Y.copy()
        
        for index, row in x_test.iterrows():
            if (count < 10):
                y_hat = (classifier.predict(x_test.iloc[count_for:(count_for+1),:]))
                z = (math.exp(-1*abs(y_hat)) < thold)

                if (z == 1):
                    count += 1   
                    t += 1
                    a = x_test.iloc[count_for:count_for+1,:]       
                    x_train = x_train.append(a)
                    lab = y_test[index]
                    lab_series = pd.Series(lab)
                    y_train = y_train.append(lab_series,ignore_index=True)
            else:
                classifier.fit(x_train, y_train)
                count = 0

            count_for +=1
    
        score_test = classifier.score(x_test, y_test)
        threshold_t.append(t)
        threshold_t_score.append(score_test)

    end = time.process_time()
    print("time takes for Perceptron= {} min".format((end - start) / 60))
    return(threshold_t, threshold_t_score)

threshold_t, threshold_t_score = fixed_sampling_perc(x_train, y_train, x_test, y_test)

plt.plot([0.04, 0.05, 0.06], threshold_t_perc, color = "red", marker="o", label = "Perceptron")
plt.title("fixed margin sampling", loc = 'center')
plt.xlabel("thresholds")
plt.ylabel("label requests")
plt.legend(loc='upper left')
plt.savefig("fixed margin sampling perceptron.png")
plt.close()


def fixed_sampling_svm(X, Y, x_test, y_test):
    start = time.process_time()
    count = 0

    classifier = LinearSVC(C=10**-1, multi_class='crammer_singer')
    classifier.fit(X, Y)
    score_train = classifier.score(X, Y)
    print(score_train)

    thresholds = [0.04, 0.05, 0.06]
    threshold_t = []
    threshold_t_score = []
    
    for thold in thresholds:
        print(thold)
        t = 0
        count_for = 0
        x_train = X.copy()
        y_train = Y.copy()
        
        for index, row in x_test.iterrows():
            if (count < 10):
                y_hat = (classifier.predict(x_test.iloc[count_for:(count_for+1),:]))
                z = (math.exp(-1*abs(y_hat)) < thold)
                
                if (z == 1):
#                     print("value of z", math.exp(-1*abs(y_hat)))
                    count += 1   
                    t += 1
                    a = x_test.iloc[count_for:count_for+1,:]       
                    x_train = x_train.append(a)
                    lab = y_test[index]
                    lab_series = pd.Series(lab)
                    y_train = y_train.append(lab_series,ignore_index=True)
            else:
                classifier.fit(x_train, y_train)
                count = 0

            count_for +=1
    
        score_test = classifier.score(x_test, y_test)
        threshold_t.append(t)
        threshold_t_score.append(score_test)

    end = time.process_time()
    print("time takes for Perceptron= {} min".format((end - start) / 60))
    return(threshold_t, threshold_t_score)

threshold_t_svm, threshold_t_score_svm = fixed_sampling_svm(x_train, y_train, x_test, y_test)

plt.plot([0.04, 0.05, 0.06], threshold_t_svm, color = "blue", marker="o", label = "Linear SVM")
plt.title("fixed margin sampling", loc = 'center')
plt.xlabel("thresholds")
plt.ylabel("label requests")
plt.legend(loc='upper left')
plt.savefig("fixed margin sampling SVM.png")
plt.close()


def train_perc(x_train, y_train, x_test, y_test):
    start = time.process_time()

    classifier = Perceptron()
    classifier.fit(x_train, y_train)
    
    score_train = classifier.score(x_train, y_train)
    print(score_train)
    
    score_test = classifier.score(x_test, y_test)
    print(score_test)

    end = time.process_time()
    print("time takes for Perceptron= {} min".format((end - start) / 60))

train_perc(x_train, y_train, x_test, y_test)


def train_svm(x_train, y_train, x_test, y_test):

    start = time.process_time()

    linear_kernel = LinearSVC(C=10**-1, multi_class='crammer_singer')
    linear_kernel.fit(x_train, y_train)
    
    score_train = linear_kernel.score(x_train, y_train)
    print(score_train)
    
    score_test = linear_kernel.score(x_test, y_test)
    print(score_test)

    end = time.process_time()
    print("time takes = {} min".format((end - start) / 60))

train_svm(x_train, y_train, x_test, y_test)