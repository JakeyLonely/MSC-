import numpy as np

data = np.loadtxt('data/housing.data', dtype= float)
X = data[..., :-1]
Y = data[..., -1:]
rowNum = X.shape[0]
colOne = np.ones((rowNum, 1))
x = np.column_stack((colOne, X[0:rowNum, :]))

a = np.dot(x.T, x)
b = np.asmatrix(a)
c = np.linalg.inv(b)
d = np.dot(c, x.T)
e = np.dot(d, Y)

i = 0
for w in e:
    print("w%d = %lf" % (i, w[0]))
    i = i + 1