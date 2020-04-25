
class MyLinearUnivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef1_ = 0.0
        self.coef2_ = 0.0

    # learn a linear univariate regression model by using training inputs (x) and outputs (y)
    def fit(self, x1, x2, y):
        sx1 = sum(x1)
        sx2=sum(x2)
        sy = sum(y)
        sx21 = sum(i * i for i in x1)
        sx22=sum(i * i for i in x2)
        sxy = sum(i * j * k for (i, j, k) in zip(x1, x2, y))
        w1 = (len(x1) * sxy - sx1 * sy) / (len(x1) * sx21 - sx1 * sx1)
        w2 = (len(x2) * sxy - sx2 * sy) / (len(x2) * sx22 - sx2 * sx2)
        w0 = (sy - w1 * sx1 - w2 * sx2) / (len(x1)+len(x2))
        self.intercept_, self.coef1_, self.coef2_ = w0, w1, w2


    # predict the outputs for some new inputs (by using the learnt model)
    def predict(self, x1, x2):
        return [self.intercept_ + self.coef1_ * val +self.coef2_ * valx for (val, valx) in zip(x1, x2)]