

class MySGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x1, x2, y):
        learningRate = 0.001
        noEpochs = 1000
        self.coef_ = [0.0 for _ in range(len(x1) + 1)]
        for epoch in range(noEpochs):
            crtError=0
            for i in range(len(x1)):
                ycomputed = self.eval(x1[i], x2[i])
                crtError += ycomputed - y[i]
            self.coef_[0] = self.coef_[0] - learningRate * crtError * x1[i]
            self.coef_[1] = self.coef_[1] - learningRate * crtError * x2[i]
            self.coef_[len(x1)] = self.coef_[len(x1)] - learningRate * crtError * 1
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]


    def eval(self, x1, x2):
        yi = self.coef_[-1]
        yi += self.coef_[0] * x1
        yi += self.coef_[1]*x2
        return yi

    def predict(self, x1, x2):
        return [self.intercept_ + self.coef_[0] * val + self.coef_[1] * valx for (val, valx) in zip(x1, x2)]