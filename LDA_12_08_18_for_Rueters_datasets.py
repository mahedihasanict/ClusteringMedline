# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:31:56 2018

@author: Mahedi Hasan
"""
import gensim
import pyodbc
import ast
import json
import math
import random

from math import log
from sklearn.cluster import SpectralClustering
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora

#**********parameters*************
number_of_passes=2
#*********************************

#**********Global variales*************
accuracy_dictionary={}
#*********************************

#*********database_credentials************
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=MAHEDIHASAN;DATABASE=Medline;UID=sa;PWD=0000')
cursor = cnxn.cursor()
cursor1 = cnxn.cursor()
cursor2 = cnxn.cursor()
cursor3 = cnxn.cursor()
cursor4 = cnxn.cursor()
cursor5 = cnxn.cursor()
#*********************************

#***************************global_space_ends_here***************************

def data_set_creation(link):
    with open(link+'7.1.training_doc_set_Reuters.txt', 'r') as f33:
            s33 = f33.read()
            training_doc_set1 = ast.literal_eval(s33)
    
    dataSetDictionary={}
    for i in range(0,100):
        nameOfDataset=''
        a=random.sample([3,4,5,6],1)
        b=random.sample(training_doc_set1.keys(),a[0])
        print b
        nameOfDataset='R2BCAV'+str(i)
        print nameOfDataset
        dataSetDictionary[nameOfDataset]=b
    
    with open(link+'100_Dataset_Rueters.txt','w') as f34:
        f34.write(str(dataSetDictionary))



def document_filtering_and_spliting_into_training_and_testing_sets(link):
    training_doc_set={}
    testing_doc_set={}
    with open(link+'7.0.catagoryIDReuters.txt', 'r') as f30:
        s30 = f30.read()
        catagory_id = ast.literal_eval(s30)


    for key, value in catagory_id.items():
        cursor2.execute("SELECT [SERIAL_NO],[CATAGORY_ID],[DOCUMENT_ID],[TRAIN_OR_TEST],[CATAGORY_NAME],[DOCUMENT_BODY] FROM [Medline].[dbo].[RuetersDataSet] where CATAGORY_ID="+str(value)+" and TRAIN_OR_TEST='test';")
        resultsetTest= cursor2.fetchall()

        cursor5.execute("SELECT [SERIAL_NO],[CATAGORY_ID],[DOCUMENT_ID],[TRAIN_OR_TEST],[CATAGORY_NAME],[DOCUMENT_BODY] FROM [Medline].[dbo].[RuetersDataSet] where CATAGORY_ID="+str(value)+" and TRAIN_OR_TEST='training';")
        resultsetTraining= cursor5.fetchall()
        
        numberOfDocsInResultsetTest=len(resultsetTest)
        numberOfDocsInResultsetTraining=len(resultsetTraining)
        if numberOfDocsInResultsetTraining>=15 and numberOfDocsInResultsetTest>=15:
            testing_doc_set[value]=[]
            training_doc_set[value]=[]
            numberOfDocsForTraining=numberOfDocsInResultsetTraining
            numberOfDocsForTest=numberOfDocsInResultsetTest
            counterTraining=0
            for i in resultsetTraining:
                document_id_training=i.DOCUMENT_ID
                if counterTraining<numberOfDocsForTraining:
                    training_doc_set[value].append(document_id_training)
                    counterTraining=counterTraining+1
            
            counterTest=0
            for j in resultsetTest:
                document_id_test=j.DOCUMENT_ID
                if counterTest<numberOfDocsForTest:
                    testing_doc_set[value].append(document_id_test)
                    counterTest=counterTest+1
        
        with open(link+'7.1.training_doc_set_Reuters.txt', 'w') as f31:
            f31.write(str(training_doc_set))
        
        with open(link+'7.2.testing_doc_set_Reuters.txt', 'w') as f32:
            f32.write(str(testing_doc_set))

#****************************global_space_starts_here****************************
link='F:\\publication work\\Data\\temp\\'
#document_filtering_and_spliting_into_training_and_testing_sets(link)
data_set_creation(link)
cursor.close()
cursor1.close()
cursor2.close()
cursor3.close()
cursor4.close()
cursor5.close()
cnxn.close()
#****************************global_space_ends_here******************************