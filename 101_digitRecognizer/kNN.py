#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
@author:buildall
@date:2016年 01月 08日 星期五 15:27:09 CST
@content: kNN - digit recognize
'''
from numpy import *
import operator
from os import listdir
import csv

def TrainImageToVector(srcfile):
    fp = open(srcfile, 'rb')                    #get rows
    lines = csv.reader(fp)
    myrowcount = -1                             #first row is index
    for line in lines: myrowcount += 1

    returnVect = zeros((myrowcount, 784))       #get return Vectors
    returnHwLabels = []                         #get labels

    fp = open(srcfile, 'rb')                    #read csv file and get Vectors and labels      
    lines = csv.reader(fp)
    line = lines.next()

    for i in range(myrowcount):
        line = lines.next()
        returnHwLabels.append(int(line[0]))

        for k in range(784):
            returnVect[i, k] = int(line[k+1])

    return (returnVect, returnHwLabels)

def TestImageToVector(srcfile):
    fp = open(srcfile, 'rb')                    #get rows
    lines = csv.reader(fp)
    myrowcount = -1                             #first row is index
    for line in lines: myrowcount += 1

    returnVect = zeros((myrowcount, 784))       #get return Vector

    fp = open(srcfile, 'rb')                    #read csv file and get Vectors
    lines = csv.reader(fp)
    line = lines.next()

    for i in range(myrowcount):
        line = lines.next()

        for k in range(784):
            returnVect[i, k] = int(line[k])

    return returnVect

def kNNClassify(thisVector, trainingMat, hwLabels, num = 3):
    trainDataSize = trainingMat.shape[0]                            #get trainData rows
    diffMat = tile(thisVector, (trainDataSize, 1)) - trainingMat    #tile(A, (m, n)) A = element m rows n cols matrix
    sqdiffMat = diffMat ** 2                                        #square
    sqDistances = sqdiffMat.sum(axis=1)                             #axis=1 add cols axis=0 add rows
    distances = sqDistances ** 0.5                                  
    sortedDistIndices = distances.argsort()
    classCount = {}                                                 #vote
    for i in range(num):
        voteLabel = hwLabels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]                                   #most vote category
 
def handwritingClassTest():
    #Train data to Vector
    trainfile = 'train.csv'
    trainresult = TrainImageToVector(trainfile)
    #Test data to Vector
    testfile = 'test.csv'
    testresult = TestImageToVector(testfile)

    #classify 
    writer = csv.writer(file('result.csv', 'wb'))
    writer.writerow(['ImageId', 'Label'])
    for i in range(len(testresult)):
        resultNum = kNNClassify(testresult[i], trainresult[0], trainresult[1], 6) # old = 3
        writer.writerow([str(i+1), str(resultNum)])


if __name__ == '__main__':
    handwritingClassTest()
       
    
