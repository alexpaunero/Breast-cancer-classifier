import codecademylib3_seaborn

# Task 1
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

# Task 2
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

# Task 3
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# Task 4
from sklearn.model_selection import train_test_split

# Task 5 & 6
training_data, validation_data, training_labels, validation_labels =train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

# Task 7
print(len(training_data))
print(len(training_labels))

# Task 8
from sklearn.neighbors import KNeighborsClassifier

# Task 9
classifier = KNeighborsClassifier(n_neighbors = 3)

# Task 10
classifier.fit(training_data, training_labels)

# Task 11
print(classifier.score(validation_data, validation_labels))

# Task 12
for i in range(1, 100):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels)
  print(classifier.score(validation_data, validation_labels))

# Task 13
import matplotlib.pyplot as plt

# Task 14
k_list = []
for i in range(1, 100):
  k_list.append(i)

# Task 15
accuracies = []
for i in range(1, 100):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

# Task 16 & 17
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
