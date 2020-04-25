import csv
import numpy as np

def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName1)
    selectedVariable2 = dataNames.index(inputVariabName2)
    inputs1 = [float(data[i][selectedVariable]) for i in range(len(data))]
    inputs2 = [float(data[i][selectedVariable2]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs1, inputs2, outputs


import os

crtDir = os.getcwd()
filePath = os.path.join(crtDir, '2017.csv')

inputs1, inputs2, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')

''''''
#normalizarea datelor de tipul min max
#incadram datele intre 0 si 1
max=0
for i in inputs1:
    x = 1
    c = 0
    while i>=x:
        x*=10
        c+=1
    if c>max:
        max=c
x=1
for i in range(max):
    x*=10
for i in range(0,len(inputs1)):
    inputs1[i] /= x


max=0
for i in inputs2:
    x = 1
    c = 0
    while i>=x:
        x*=10
        c+=1
    if c>max:
        max=c
x=1
for i in range(max):
    x*=10
for i in range(0,len(inputs2)):
    inputs2[i] /= x
'''
max=0
for i in outputs:
    x = 1
    c = 0
    while i>=x:
        x*=10
        c+=1
    if c>max:
        max=c
x=1
for i in range(max):
    x*=10
for i in range(0,len(outputs)):
    outputs[i] /= x
'''
print('inEconomy:  ', inputs1[:5])
print('inFreedom:  ', inputs2[:5])
print('out: ', outputs[:5])


np.random.seed(5)
indexes = [i for i in range(len(inputs1))]
trainSample1 = np.random.choice(indexes, int(0.8 * len(inputs1)), replace = False)
testSample1 = [i for i in indexes  if not i in trainSample1]
trainInputs1 = [inputs1[i] for i in trainSample1]
testInputs1 = [inputs1[i] for i in testSample1]


np.random.seed(5)
indexes = [i for i in range(len(inputs2))]
trainSample2 = np.random.choice(indexes, int(0.8 * len(inputs1)), replace = False)
testSample2 = [i for i in indexes  if not i in trainSample2]
trainInputs2 = [inputs2[i] for i in trainSample2]
testInputs2 = [inputs2[i] for i in testSample2]

trainOutputs = [outputs[i] for i in trainSample1]
testOutputs = [outputs[i] for i in testSample1]
'''
from myRegression import MyLinearUnivariateRegression

regressor = MyLinearUnivariateRegression()
regressor.fit(trainInputs1, trainInputs2, trainOutputs)
w0, w1, w2 = regressor.intercept_, regressor.coef1_, regressor.coef2_
print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1', ' + ', w2, ' * x2')

for i in range (0,len(testInputs2)):
    computedTestOutputs = regressor.predict([testInputs1[i]], [testInputs2[i]])



error = 0.0
for t1, t2 in zip(computedTestOutputs, testOutputs):
    error += (t1 - t2) ** 2
error = error / len(testOutputs)
print('prediction error (manual): ', error)

'''

from mySGDregression import MySGDRegression

regressor = MySGDRegression()
regressor.fit(trainInputs1, trainInputs2, trainOutputs)
w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1', ' + ', w2, ' * x2')

for i in range (0,len(testInputs2)):
    computedTestOutputs = regressor.predict([testInputs1[i]], [testInputs2[i]])



error = 0.0
for t1, t2 in zip(computedTestOutputs, testOutputs):
    error += (t1 - t2) ** 2
error = error / len(testOutputs)
print('prediction error (manual): ', error)