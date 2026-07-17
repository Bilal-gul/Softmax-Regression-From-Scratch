import numpy as np
import matplotlib.pyplot as plt
from model.preprocessing import normalize_train as nrm


class softmaxRegression:

    def __init__(self,learning_rate = 0.1,epoch=5000):
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.loss_history = []

    def linear(self,matrix):
        return np.dot(matrix,self.weights) + self.bias

    def softmax_proba(self,matrix,axis,keepdims):
        
        basic_exp_matrix_QTx = np.exp(matrix - np.max(matrix,axis=axis,keepdims=keepdims))

        return basic_exp_matrix_QTx / np.sum(basic_exp_matrix_QTx,axis=axis,keepdims=keepdims)

    def save(self,path):
        np.savez(path, weights = self.weights, bias = self.bias, X_mean = self.mean, X_std = self.std)

    def load(self,path):
        model = np.load(path)

        self.weights = model["weights"]
        self.bias = model["bias"]
        self.mean = model["X_mean"]
        self.std = model["X_std"]

    def chart_show(self,array):
        plt.plot(array)
        plt.grid()
        plt.show()

    def predict_matrix(self,probability,axis,keepdims):
        return np.argmax(probability,axis=axis,keepdims=keepdims) + 1

    def fit(self,X_train,one_hot):

        m,n = X_train.shape
        n_classes = one_hot.shape[1]

        self.weights = np.zeros((n,n_classes))
        self.bias = np.zeros((1,n_classes))

        for i in range(self.epoch):

            basic_matrix_QTx = self.linear(X_train)
           
            probability_matrix = self.softmax_proba(basic_matrix_QTx,1,True)

            error_matrix = one_hot - probability_matrix

            gradient = np.dot(X_train.T,error_matrix)/m

            self.weights += self.learning_rate*gradient
            self.bias += self.learning_rate * np.sum(error_matrix,axis=0,keepdims=True) / m

            if i % 100 == 0:
                cross_entropy = - np.sum(one_hot * np.log(probability_matrix)) / m
                self.loss_history.append(cross_entropy)
                print(cross_entropy)

    def predict(self):
        soil = np.zeros((40))
        area = np.zeros((4))

        try:

            while True:
                
                elevation = int(input("Elevation : "))
                aspect = int(input("Aspect : "))
                slope = int(input("Slope : "))
                horizontal_distance_to_hydrology = int(input("Horizontal Distance to Hydrology : "))
                vertical_distance_to_hydrology = int(input("Vertical Distance to Hydrology : "))
                horizontal_distance_to_roadways = int(input("Horizontal Distance to Roadways : "))
                hillshade_9am = int(input("Hillshade 9am : "))
                hillshade_noon = int(input("Hillshade Noon : "))
                hillshade_3pm = int(input("Hillshade 3pm : "))
                horizontal_distance_to_fire_points = int(input("Horizontal Distance to Fire Points : "))
                wilderness_area = int(input("Wilderness Area (1-4) : "))
                area[wilderness_area-1] = 1
                soil_type = int(input("Soil Type (1-40) : "))
                soil[soil_type-1] = 1

                numerical = np.array([elevation,aspect,slope,horizontal_distance_to_hydrology,vertical_distance_to_hydrology,horizontal_distance_to_roadways,hillshade_9am,hillshade_noon,hillshade_3pm,horizontal_distance_to_fire_points])
                new_data = np.concatenate((numerical,area,soil))

                data_norm = nrm(new_data,self.mean,self.std)

                basic_matrix_QTx = self.linear(data_norm)

                probability_matrix = self.softmax_proba(basic_matrix_QTx,None,False)

                predict = self.predict_matrix(probability_matrix,None,False)

                probability = float(probability_matrix[0][predict-1])

                selection = input("Would you like to continue guessing? (yes / no) :").lower().strip()
                
                if selection == "yes":
                    continue
                elif selection == "no":
                    break
                else:
                    raise ValueError("Please, enter a valid value")
        except ValueError as e:
            print(f"System Error : {e}")
        except Exception as e:
            print(f"System Error : {e}")

        return predict,probability

        