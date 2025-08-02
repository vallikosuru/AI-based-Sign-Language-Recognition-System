import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('C:/suprajaHackathon/sign-language-detector-python-master/sign-language-detector-python-master/data.pickle', 'rb'))

# Keep only samples with exactly 42 features
filtered_data = [sample for sample in data_dict['data'] if len(sample) == 42]
filtered_labels = [label for i, label in enumerate(data_dict['labels']) if len(data_dict['data'][i]) == 42]

# Convert to numpy arrays
data = np.array(filtered_data)
labels = np.array(filtered_labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
