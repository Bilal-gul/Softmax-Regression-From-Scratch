import numpy as np


def shuffle_data(X,y):

    total_line = X.shape[0]

    np.random.seed(42)

    indexes = np.arange(total_line)
    np.random.shuffle(indexes)

    X_mixed = X[indexes]
    y_mixed = y[indexes]

    return X_mixed,y_mixed,total_line

def train_test_split(df):

    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values.reshape(-1,1)

    X_mixed,y_mixed,total_line = shuffle_data(X,y)

    cutting = int(0.8*total_line)

    X_train = X_mixed[:cutting]
    y_train = y_mixed[:cutting]

    X_test = X_mixed[cutting:]
    y_test = y_mixed[cutting:]

    return  X_train,y_train,X_test,y_test

def train_mean_std(train):
    X_train_mean = np.mean(train,axis=0)
    X_train_std = np.std(train,axis=0)
    X_train_std[X_train_std == 0] = 1

    return X_train_mean,X_train_std

def normalize_train(train,mean,std):
    return (train - mean)/std

def normalize_test(test,mean,std):
    return (test - mean)/std

def hot_encoding(y):
    m = y.shape[0]

    n_classes = len(np.unique(y))

    one_hot = np.zeros((m,n_classes))

    for i in range(m):
        one_hot[i,y[i,0]-1] = 1

    return one_hot
