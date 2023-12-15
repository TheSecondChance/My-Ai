from sklearn import tree

# Sample dataset (features and labels)
features = [[0, 0], [1, 1], [1, 0], [0, 1]]
labels = [0, 1, 1, 0]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

new_data = [[0, 0], [1, 0]]
predictions = clf.predict(new_data)

for i in range(len(new_data)):
    print(f"Prediction for {new_data[i]}: {predictions[i]}")
