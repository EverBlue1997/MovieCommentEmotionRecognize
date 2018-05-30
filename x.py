'''
Created on 2018/5/11

@author: apple
'''

result = open("D:/input/result.txt", 'br')
data = open("D:/input/labeledTrainData.tsv", 'br')
aa = open("D:/input/aa.csv", 'w', errors = 'ignore')
x = result.readline().decode('utf-8').split('[')
x = x[1].split(']')
x = x[0].split(',')
y = data.readline()
aa.write("id,sentiment\n")
for a in x:
    y = data.readline().decode('utf-8').split('\t')
    k = y[0]
    y = k.split('"')[1]
    aa.write("{},{}\n".format(y, a))
    
