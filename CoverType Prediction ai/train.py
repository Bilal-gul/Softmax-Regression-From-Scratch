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
