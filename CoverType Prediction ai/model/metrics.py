import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_test,predict_test):

    confusion_matrix = np.zeros((7,7),dtype=int)
    TP = np.zeros((1,7))
    FP = np.zeros((1,7))
    TN = np.zeros((1,7))
    FN = np.zeros((1,7))

    for i in range(y_test.shape[0]):

        confusion_matrix[y_test[i,0]-1,predict_test[i,0]-1] += 1 

    total = np.sum(confusion_matrix)

    for i  in range(7):
        TP[0,i] = confusion_matrix[i,i] 

        FP[0,i] = np.sum(confusion_matrix[:,i]) - TP[0,i]

        FN[0,i] = np.sum(confusion_matrix[i,:]) - TP[0,i]

        TN[0,i] = total - TP[0,i] - FP[0,i] - FN[0,i]

    return confusion_matrix,TP,FP,FN,TN

def accuracy(confusion_matrix):
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

def precision(TP,FP):
    return TP / (TP+FP+1e-10)

def recall(TP,FN):
    return TP / (TP+FN+1e-10)

def f1(pre,rcl):
    return 2*pre*rcl / (pre+rcl+1e-10)

def confusion_matrix_chart(confusion_matrix):

    n = confusion_matrix.shape[0]

    plt.figure(figsize=(8,6))

    plt.imshow(confusion_matrix,cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(range(n),range(1,n+1))
    plt.yticks(range(n),range(1,n+1))

    plt.colorbar()

    threshold = confusion_matrix.max()/2

    for i in range(n):
        for j in range(n):

            plt.text(
                j,
                i,
                int(confusion_matrix[i,j]),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i,j] > threshold else "black"
            )

    plt.tight_layout()
    plt.show()