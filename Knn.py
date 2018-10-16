from os.path import join
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import time

start_time = time.time()
cat_img = "Dataset/Cat"
dog_img = "Dataset/Dog"
X = []
y = []
cnt = 0
for i in os.listdir(cat_img):
	f = open(join(cat_img, i), "rb")
	feature = np.load(f)
	feature_np = np.array(feature).flatten()
	X.append(feature_np)
	y.append("Cat")
	f.close()


for i in os.listdir(dog_img):
	f = open(join(dog_img, i), "rb")
	feature = np.load(f)
	feature_np = np.array(feature).flatten()
	X.append(feature_np)
	y.append("Dog")
	f.close()

X_np = np.array(X)
y_np = np.array(y)
print(X_np.shape)
print(y_np.shape)
print("Creating train and test set!")
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size = 0.33)
print("Done!")
model = KNeighborsClassifier(n_neighbors = 3)
print("Training model!")
model.fit(X_train, y_train)
print("Starting predict test set!")
y_pred = model.predict(X_test)
print("Accuracy of KNN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print("Writing result to file!")
result = []
y_test = y_test.T
y_pred = y_pred.T
result.append(y_test)
result.append(y_pred)
result_np = np.array(result)
result_np = result_np.T
if (os.path.isfile("result with k = 3.csv") == False):
	with (open("result with k = 3.csv","w")) as f:
		 writer = csv.writer(f)
		 writer.writerow(("ground truth", "predict"))
		 writer.writerows(result_np)
f.close()
print("All done!")
end_time = time.time()

print("Program excutes in %s second.", end_time - start_time)