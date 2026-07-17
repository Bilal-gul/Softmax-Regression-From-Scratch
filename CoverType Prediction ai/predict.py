from model.softmax_regression_model import softmaxRegression as srm

# region create object
model = srm()
# endregion

# region load model
model.load("CoverType Prediction ai\\Covertype_parameters")
# endregion

# region prediction
predict,probability = model.predict()
# endregion

# region prediction result
print("\n" + "=" * 60)
print("          FOREST COVER TYPE PREDICTION")
print("=" * 60)
print(f"Predicted Class : {predict}")
print(f"Probability      : %{probability * 100:.2f}")
print("-" * 60)

cover_types = {
1: "Spruce/Fir",
2: "Lodgepole Pine",
3: "Ponderosa Pine",
4: "Cottonwood/Willow",
5: "Aspen",
6: "Douglas-fir",
7: "Krummholz"
}

print(f"Forest Cover Type : {cover_types[predict]}")

print("=" * 60)
# endregion


















































































# model = np.load("CoverType Prediction ai\\Covertype_parameters.npz")

# w = model["weights"]
# b = model["bias"]
# X_train_mean = model["X_mean"]
# X_train_std = model["X_std"]

# soil = np.zeros((40))
# area = np.zeros((4))

# while True:
    
#     elevation = int(input("Elevation : "))
#     aspect = int(input("Aspect : "))
#     slope = int(input("Slope : "))
#     horizontal_distance_to_hydrology = int(input("Horizontal Distance to Hydrology : "))
#     vertical_distance_to_hydrology = int(input("Vertical Distance to Hydrology : "))
#     horizontal_distance_to_roadways = int(input("Horizontal Distance to Roadways : "))
#     hillshade_9am = int(input("Hillshade 9am : "))
#     hillshade_noon = int(input("Hillshade Noon : "))
#     hillshade_3pm = int(input("Hillshade 3pm : "))
#     horizontal_distance_to_fire_points = int(input("Horizontal Distance to Fire Points : "))
#     wilderness_area = int(input("Wilderness Area (1-4) : "))
#     area[wilderness_area-1] = 1
#     soil_type = int(input("Soil Type (1-40) : "))
#     soil[soil_type-1] = 1

#     numerical = np.array([elevation,aspect,slope,horizontal_distance_to_hydrology,vertical_distance_to_hydrology,horizontal_distance_to_roadways,hillshade_9am,hillshade_noon,hillshade_3pm,horizontal_distance_to_fire_points])

#     new_data = np.concatenate((numerical,area,soil))

#     data_norm = (new_data - X_train_mean) / X_train_std

#     basic_matrix_QTx = np.dot(data_norm,w) + b

#     basic_matrix_QTx = basic_matrix_QTx - np.max(basic_matrix_QTx)

#     basic_exp_matrix_QTx = np.exp(basic_matrix_QTx)

#     probability_matrix = basic_exp_matrix_QTx / np.sum(basic_exp_matrix_QTx)

#     predict = np.argmax(probability_matrix) + 1
#     probability = float(probability_matrix[0][predict-1])

#     break

# print("\n" + "=" * 60)
# print("          FOREST COVER TYPE PREDICTION")
# print("=" * 60)

# print(f"Predicted Class : {predict}")
# print(f"Probability      : %{probability * 100:.2f}")

# print("-" * 60)

# cover_types = {
#     1: "Spruce/Fir",
#     2: "Lodgepole Pine",
#     3: "Ponderosa Pine",
#     4: "Cottonwood/Willow",
#     5: "Aspen",
#     6: "Douglas-fir",
#     7: "Krummholz"
# }

# print(f"Forest Cover Type : {cover_types[predict]}")

# print("=" * 60)
