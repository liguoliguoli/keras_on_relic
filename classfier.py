import os
import pickle
import numpy as np

pkz_dir = r"F:\jupyter-notebook\deeplearning\features"
f_pkzs = [os.path.join(pkz_dir, x) for x in os.listdir(pkz_dir) if x.startswith("rawpixel_dpm_10_64")]
print(f_pkzs)

x, y = [], []

for f_pkz in f_pkzs:
    with open(f_pkz, 'rb') as f:
        print("load ", f_pkz)
        batch_data = pickle.load(f)
        if len(x) == 0:
            x = batch_data['x']
            y = batch_data['y']
        else:
            x = np.concatenate((x, batch_data['x']))
            y = np.concatenate((y, batch_data['y']))

slices = np.arange(len(x))
np.random.shuffle(slices)
X = x[slices]
Y = y[slices]

X /= 255
X = np.reshape(X, (X.shape[0], -1))

from sklearn import svm
clf = svm.SVC(gamma="scale", kernel="linear")
print("train...")
clf.fit(X[:3000], Y[:3000])
print("predict...")
pre = clf.predict(X[3000:])
print("score...")
res = [1 if pre[i] == Y[i+3000] else 0 for i in range(len(pre))]
score = sum(res) / len(res)
print(score)
