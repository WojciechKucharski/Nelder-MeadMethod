from algorithm import NMMethod

fun = "x1^2+x2^2"
x0 = [[5, 5], [-2, -2], [2, -2]]

NMM = NMMethod()
x = NMM.runAlgorithm(fun, x0, alfa=0.5, beta=0.1, gamma=0.5, epsilon=1e-6)
print(x)
