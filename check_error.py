import pickle

data_dict = pickle.load(open('C:/suprajaHackathon/sign-language-detector-python-master/sign-language-detector-python-master/data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Find any samples with inconsistent data
for i, sample in enumerate(data):
    if len(sample) != 42:  # Each hand should have 21 landmarks (x, y) = 42 values
        print(f"Inconsistent sample at index {i}, label: {labels[i]}, length: {len(sample)}")

print("Total samples:", len(data))
