import pickle

data_dict= pickle.load(open('C:/suprajaHackathon/sign-language-detector-python-master/sign-language-detector-python-master/data.pickle','rb'))
print(data_dict.keys())
print(data_dict)