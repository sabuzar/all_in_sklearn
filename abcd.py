import pickle
with open('model_1.0','rb') as f:
    mp=pickle.load(f)
print(mp.predict([[5000]]))