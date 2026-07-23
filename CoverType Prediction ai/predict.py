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
