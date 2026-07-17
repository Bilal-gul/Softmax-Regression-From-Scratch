import numpy as np 
import pandas as pd 
import time
import model.softmax_regression_model as srm
import model.preprocessing as prc
import model.metrics as met

# region read data
df = pd.read_csv("CoverType Prediction ai\\data\\covtype.data",header=None)
# endregion

# region set and normalize data
X_train,y_train,X_test,y_test = prc.train_test_split(df)

X_train_mean,X_train_std = prc.train_mean_std(X_train)

X_train_norm = prc.normalize_train(X_train,X_train_mean,X_train_std)
X_test_norm = prc.normalize_test(X_test,X_train_mean,X_train_std)
# endregion

# region one hot
one_hot = prc.hot_encoding(y_train)
# endregion

# region epoch
model = srm.softmaxRegression()

start = time.time()

model.fit(X_train_norm,one_hot)
model.mean = X_train_mean
model.std = X_train_std

end = time.time()
print(f"\nTraining Time : {(end-start)/60:.2f} minutes")
# endregion

# region model save
model.save("CoverType Prediction ai\\Covertype_parameters")
# endregion

# region test data set 
basic_matrix_test = model.linear(X_test_norm)

prediction_matrix = model.softmax_proba(basic_matrix_test,1,True)

predict_test = model.predict_matrix(prediction_matrix,1,True)
# endregion

# region evaluation metrics
confusion_matrix,TP,FP,FN,TN = met.confusion_matrix(y_test,predict_test)

accuracy = met.accuracy(confusion_matrix)
precision = met.precision(TP,FP)
recall = met.recall(TP,FN)
f1 = met.f1(precision,recall)
# endregion

# region chart show
model.chart_show(model.loss_history)
met.confusion_matrix_chart(confusion_matrix)
# endregion

# region test result
print("\n" + "="*60)
print("        SOFTMAX REGRESSION EVALUATION RESULTS")
print("="*60)

print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {np.mean(precision):.4f}")
print(f"Macro Recall      : {np.mean(recall):.4f}")
print(f"Macro F1 Score    : {np.mean(f1):.4f}")

print("\nConfusion Matrix")
print("-"*60)
print(confusion_matrix)
print("-"*60)

print("\nPer-Class Metrics")
print("-"*60)

for i in range(7):
    print(f"Class {i+1}")
    print(f"  TP : {int(TP[0,i])}")
    print(f"  FP : {int(FP[0,i])}")
    print(f"  FN : {int(FN[0,i])}")
    print(f"  TN : {int(TN[0,i])}")
    print(f"  Precision : {precision[0,i]:.4f}")
    print(f"  Recall    : {recall[0,i]:.4f}")
    print(f"  F1 Score  : {f1[0,i]:.4f}")
    print("-"*60)
# endregion





























































































# df = df.sample(n=30000,random_state=42)

# X = df.iloc[:,:-1].values
# y = df.iloc[:,-1].values.reshape(-1,1)
# # endregion

# # region normalize
# total_line = X.shape[0]

# np.random.seed(42)

# indexes = np.arange(total_line)
# np.random.shuffle(indexes)

# X_mixed = X[indexes]
# y_mixed = y[indexes]

# cutting = int(0.8*total_line)

# X_train = X_mixed[:cutting]
# y_train = y_mixed[:cutting]

# X_test = X_mixed[cutting:]
# y_test = y_mixed[cutting:]

# X_train_mean = np.mean(X_train,axis=0)
# X_train_std = np.std(X_train,axis=0)
# X_train_std[X_train_std == 0] = 1

# X_train_norm = (X_train - X_train_mean)/X_train_std
# X_test_norm = (X_test - X_train_mean)/X_train_std

# # endregion

# # region hot encoding
# m = y_train.shape[0]
# n = X_train_norm.shape[1]

# n_classes = len(np.unique(y))

# one_hot = np.zeros((m,n_classes))

# for i in range(m):
#     one_hot[i,y_train[i,0]-1] = 1

# # endregion


# w = np.zeros((n,7))
# b = np.zeros((1,7))
# learning_rate = 0.1
# epoch = 5000
# loss_history = []

# start = time.time()

# for i in range(epoch):

#     basic_matrix_QTx = np.dot(X_train_norm,w) + b
#     basic_matrix_QTx = basic_matrix_QTx - np.max(basic_matrix_QTx,axis=1,keepdims=True)

#     basic_exp_matrix_QTx = np.exp(basic_matrix_QTx)

#     probability_matrix = basic_exp_matrix_QTx / np.sum(basic_exp_matrix_QTx,axis=1,keepdims=True)

#     error_matrix = one_hot - probability_matrix

#     gradient = np.dot(X_train_norm.T,error_matrix)/m

#     w += learning_rate*gradient
#     b += learning_rate * np.sum(error_matrix,axis=0,keepdims=True) / m

#     if i % 100 == 0:
#         cross_entropy = - np.sum(one_hot * np.log(probability_matrix)) / m
#         loss_history.append(cross_entropy)
#         print(cross_entropy)

# end = time.time()

# np.savez("CoverType Prediction ai\\Covertype_parameters", weights = w, bias = b, X_mean = X_train_mean, X_std = X_train_std)
# print(f"\nTraining Time : {(end-start)/60:.2f} minutes")

# plt.plot(loss_history)
# plt.grid()
# plt.show()

# # test


# confusion_matrix = np.zeros((7,7),dtype=int)
# TP = np.zeros((1,7))
# FP = np.zeros((1,7))
# TN = np.zeros((1,7))
# FN = np.zeros((1,7))


# basic_matrix_test = np.dot(X_test_norm,w) + b

# basic_matrix_test = basic_matrix_test - np.max(basic_matrix_test,axis=1,keepdims=True)

# exp_test_matrix = np.exp(basic_matrix_test)

# prediction_matrix = exp_test_matrix / np.sum(exp_test_matrix,axis=1,keepdims=True)

# predict_test = np.argmax(prediction_matrix,axis=1,keepdims=True) + 1 

# for i in range(y_test.shape[0]):

#     confusion_matrix[y_test[i,0]-1,predict_test[i,0]-1] += 1 

# total = np.sum(confusion_matrix)

# for i  in range(7):
#     TP[0,i] = confusion_matrix[i,i] 

#     FP[0,i] = np.sum(confusion_matrix[:,i]) - TP[0,i]

#     FN[0,i] = np.sum(confusion_matrix[i,:]) - TP[0,i]

#     TN[0,i] = total - TP[0,i] - FP[0,i] - FN[0,i]


# accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
# precision = TP / (TP+FP+1e-10)
# recall = TP / (TP+FN+1e-10)
# f1 = 2*precision*recall / (precision+recall+1e-10)

# print("\n" + "="*60)
# print("        SOFTMAX REGRESSION EVALUATION RESULTS")
# print("="*60)

# print(f"\nAccuracy          : {accuracy:.4f}")
# print(f"Macro Precision   : {np.mean(precision):.4f}")
# print(f"Macro Recall      : {np.mean(recall):.4f}")
# print(f"Macro F1 Score    : {np.mean(f1):.4f}")

# print("\nConfusion Matrix")
# print("-"*60)
# print(confusion_matrix)
# print("-"*60)

# print("\nPer-Class Metrics")
# print("-"*60)

# for i in range(7):
#     print(f"Class {i+1}")
#     print(f"  TP : {int(TP[0,i])}")
#     print(f"  FP : {int(FP[0,i])}")
#     print(f"  FN : {int(FN[0,i])}")
#     print(f"  TN : {int(TN[0,i])}")
#     print(f"  Precision : {precision[0,i]:.4f}")
#     print(f"  Recall    : {recall[0,i]:.4f}")
#     print(f"  F1 Score  : {f1[0,i]:.4f}")
#     print("-"*60)