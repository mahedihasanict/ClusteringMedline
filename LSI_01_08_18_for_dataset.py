# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 17:41:53 2018

@author: Mahedi Hasan
"""

#***************************global_space_starts_here***************************
import gensim
import pyodbc
import ast
import json
import math
import random

from sklearn.cluster import SpectralClustering
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora


from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np

#from SimilarityFromJSD import similarityFromJSD
#from ReplaceBrackets import replaceBrackets
#from DistanceFromJSDFromOnlyProb import distanceFromJSD
#from DistanceFromJSDFromReplaceBrackets import distanceFromJSD


#**********parameters*************
number_of_passes=2
#*********************************

#**********Global variales*************
accuracy_dictionary={}
#*********************************

#*********database_credentials************
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-P61DTNE;DATABASE=Medline;UID=sa;PWD=0000')
cursor = cnxn.cursor()
cursor1 = cnxn.cursor()
cursor2 = cnxn.cursor()
cursor3 = cnxn.cursor()
cursor4 = cnxn.cursor()
#*********************************

#***************************global_space_ends_here***************************



def model_training_and_testing_also_main_function():
    return_dictionary=corpus_creation('F:\\publication work\\Data\\temp\\100_Dataset.txt','F:\\publication work\\Data\\temp\\7.1.training_doc_set.txt','F:\\publication work\\Data\\temp\\7.2.testing_doc_set.txt')
    
    for key,value in return_dictionary.items():
        dataset_name=key
        print 'Number of documents for Training: '+str(value[0]) 
        number_of_topics_produced=value[1]
        print 'Number of topics produced: '+str(number_of_topics_produced)
        number_of_clusters=value[2]
        print 'Number of clusters: '+str(number_of_clusters)
        print 'Number of passes: '+str(number_of_passes)
        
        #pubmed_identifier_list_Training=value[3]
        #topic_doc_dictionary_Training=value[4]
        pubmed_identifier_list_Testing=value[8]
        topic_doc_dictionary_Testing=value[9]
        trainingCorpus=value[11]
        dictionary=value[10]

        lsiModel=gensim.models.lsimodel.LsiModel(corpus=trainingCorpus, num_topics=number_of_topics_produced, id2word=dictionary, chunksize=100, decay=1.0, distributed=False, onepass=True, power_iters=2, extra_samples=100)
        #index = similarities.MatrixSimilarity(lsiModel[trainingCorpus])
        
        #ldaModel = gensim.models.ldamodel.LdaModel(corpus=trainingCorpus, num_topics=number_of_topics_produced, id2word=dictionary, distributed=False, chunksize=100, passes=number_of_passes, update_every=1, alpha='symmetric', eta='auto', decay=0.5, offset=1.0, eval_every=10, iterations=10, gamma_threshold=0.00000001, minimum_probability=0.0000000000001)
        
        testingCorpus=value[12]
        
        list_of_topics=[]
        dist_of_docs_over_topics=[]
        #similarity_list=[]
        
        for iii in testingCorpus:
            dist_of_docs_over_topics.append(lsiModel[iii])


        for i in range(0, lsiModel.num_topics):
            list_of_topics.append(lsiModel.print_topic(topicno=i, topn=10))
            

        with open('F:\\publication work\\Data\\temp\\8.pubmed_identifier'+dataset_name+'.txt', 'w') as f2:
            f2.write(str(pubmed_identifier_list_Testing))
        
        with open('F:\\publication work\\Data\\temp\\9.topics_list'+dataset_name+'.txt', 'w') as f3:
            f3.write(str(list_of_topics))
        
        #print dist_of_new_docs_over_topics
        with open('F:\\publication work\\Data\\temp\\10.distribution_of_topics_in_docs'+dataset_name+'.txt', 'w') as f4:
            f4.write(str(dist_of_docs_over_topics))
        
        with open('F:\\publication work\\Data\\temp\\17.topic_doc_dictionary'+dataset_name+'.txt','w') as f5:
            f5.write(str(topic_doc_dictionary_Testing))



        replaceBrackets(dataset_name)
        print 'Bracket replacing completed'
        
        keepingOnlyProbability(number_of_topics_produced,dataset_name)
        print 'Keeping only probability completed'
        
        cosineSimilarity(dataset_name)
        print 'Measuring similarity completed'
        
        spectralClustering(number_of_clusters,dataset_name)
        print 'Spectral clustering completed'
        
        combiningClusterResult(dataset_name)
        print 'Combining clustering result completed'
        
        common_doc_in_class_and_counter(dataset_name)
        print 'Common document in class and cluster completed'
        
        findMaximumSimilarityClusterWise(dataset_name)
        print 'Finding maximum similarity completed'
                
        accuracyMeasure(dataset_name)
        print 'Measuring accuracy completed'
        
        
    with open('F:\\publication work\\Data\\temp\\18.NMI_dictionary.txt','w') as f6:
        f6.write(str(accuracy_dictionary))


    del return_dictionary




def corpus_creation(datasetName_topicName_dic,topicName_pmid_dicTraining,topicName_pmid_dicTesting ):
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    
    with open(datasetName_topicName_dic, 'r') as f1:
            s1 = f1.read()
            all_dataset = ast.literal_eval(s1)

    with open(topicName_pmid_dicTraining, 'r') as f35:
            s35 = f35.read()
            training_doc_set = ast.literal_eval(s35)

    with open(topicName_pmid_dicTesting, 'r') as f36:
            s36 = f36.read()
            testing_doc_set = ast.literal_eval(s36)    
    return_dic={}
    
    
    
    
    for key,value in all_dataset.items():
        dataset_name=key
        return_dic[dataset_name]=[]
        
        #Document set collection for training corpus starts here
        striTraining='where'
        j_Training=0
        for i_Training in value:
            tempTraining=training_doc_set[i_Training]
            for k_Training in tempTraining:
                if (j_Training==0):
                    striTraining=striTraining+' '+'PUBMED_IDENTIFIER='+str(k_Training)
                    j_Training=j_Training+1
                else:
                    striTraining=striTraining+' or '+'PUBMED_IDENTIFIER='+str(k_Training)


        doc_set_Training = list()
        pubmed_identifier_list_Training=list()
        topic_doc_dictionary_Training={}
        cursor1.execute("SELECT [TOPIC_NO],[SERIAL_NO],[PUBMED_IDENTIFIER],[ABSTRACT] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005]"+striTraining+";")
        for row1 in cursor1.fetchall():
            abstract_Training= row1.ABSTRACT.strip()
            serial_no_Training= row1.SERIAL_NO
            topic_no_Training=row1.TOPIC_NO
            pubmed_identifier_Training=row1.PUBMED_IDENTIFIER
            pubmed_identifier_list_Training.append(pubmed_identifier_Training)
            if topic_no_Training in topic_doc_dictionary_Training.keys():
                topic_doc_dictionary_Training[topic_no_Training].append(pubmed_identifier_Training)
            else:
                topic_doc_dictionary_Training[topic_no_Training]= list()
                topic_doc_dictionary_Training[topic_no_Training].append(pubmed_identifier_Training)
            number_of_topics_in_a_dataset_Training=len(topic_doc_dictionary_Training.keys())
            if not abstract_Training:
                cursor2.execute("SELECT [TITLE] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005] where SERIAL_NO='"+str(serial_no_Training)+"';")
                for row2 in cursor2.fetchall():
                   abstract_Training=row2.TITLE.strip()
            doc_set_Training.append(abstract_Training)
        
        number_of_documents_Training=len(doc_set_Training)
        return_dic[dataset_name].append(number_of_documents_Training)
        
        number_of_topics_to_be_produced_Training=len(doc_set_Training)/24
        return_dic[dataset_name].append(number_of_topics_to_be_produced_Training)
        
        number_of_clusters_to_be_produced_Training=number_of_topics_in_a_dataset_Training
        return_dic[dataset_name].append(number_of_clusters_to_be_produced_Training)
        
        return_dic[dataset_name].append(pubmed_identifier_list_Training)
        return_dic[dataset_name].append(topic_doc_dictionary_Training)

        #print 'Number of documents: '+str(len(doc_set))  
        #print 'Number of topics to be produced: '+str(number_of_topics_to_be_produced)
        #print 'Number of passes: '+str(number_of_passes)
        #print 'Number of clusters: '+str(number_of_topics_in_a_dataset)

        # Declaring list for tokenized documents in loop
        texts_Training = []
        
        # loop through document list
        for i in doc_set_Training:
            
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            #print(tokens)
        
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            
            # add tokens to list
            texts_Training.append(stemmed_tokens)




        #Document set collection for testing corpus starts here
        striTesting='where'
        j_Testing=0
        for i_Testing in value:
            tempTesting=testing_doc_set[i_Testing]
            for k_Testing in tempTesting:
                if (j_Testing==0):
                    striTesting=striTesting+' '+'PUBMED_IDENTIFIER='+str(k_Testing)
                    j_Testing=j_Testing+1
                else:
                    striTesting=striTesting+' or '+'PUBMED_IDENTIFIER='+str(k_Testing)


        doc_set_Testing = list()
        pubmed_identifier_list_Testing=list()
        topic_doc_dictionary_Testing={}
        cursor3.execute("SELECT [TOPIC_NO],[SERIAL_NO],[PUBMED_IDENTIFIER],[ABSTRACT] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005]"+striTesting+";")
        for row3 in cursor3.fetchall():
            abstract_Testing= row3.ABSTRACT.strip()
            serial_no_Testing= row3.SERIAL_NO
            topic_no_Testing=row3.TOPIC_NO
            pubmed_identifier_Testing=row3.PUBMED_IDENTIFIER
            pubmed_identifier_list_Testing.append(pubmed_identifier_Testing)
            if topic_no_Testing in topic_doc_dictionary_Testing.keys():
                topic_doc_dictionary_Testing[topic_no_Testing].append(pubmed_identifier_Testing)
            else:
                topic_doc_dictionary_Testing[topic_no_Testing]= list()
                topic_doc_dictionary_Testing[topic_no_Testing].append(pubmed_identifier_Testing)
            number_of_topics_in_a_dataset_Testing=len(topic_doc_dictionary_Testing.keys())
            if not abstract_Testing:
                cursor4.execute("SELECT [TITLE] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005] where SERIAL_NO='"+str(serial_no_Testing)+"';")
                for row4 in cursor4.fetchall():
                   abstract_Testing=row4.TITLE.strip()
            doc_set_Testing.append(abstract_Testing)
        
        number_of_documents_Testing=len(doc_set_Testing)
        return_dic[dataset_name].append(number_of_documents_Testing)
        
        number_of_topics_to_be_produced_Testing=len(doc_set_Testing)/24
        return_dic[dataset_name].append(number_of_topics_to_be_produced_Testing)
        
        number_of_clusters_to_be_produced_Testing=number_of_topics_in_a_dataset_Testing
        return_dic[dataset_name].append(number_of_clusters_to_be_produced_Testing)
        
        return_dic[dataset_name].append(pubmed_identifier_list_Testing)
        return_dic[dataset_name].append(topic_doc_dictionary_Testing)

        #print 'Number of documents: '+str(len(doc_set))  
        #print 'Number of topics to be produced: '+str(number_of_topics_to_be_produced)
        #print 'Number of passes: '+str(number_of_passes)
        #print 'Number of clusters: '+str(number_of_topics_in_a_dataset)

        # Declaring list for tokenized documents in loop
        texts_Testing = []
        
        # loop through document list
        for i in doc_set_Testing:
            
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            #print(tokens)
        
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            
            # add tokens to list
            texts_Testing.append(stemmed_tokens)




        
        texts_training_and_testing_merged=texts_Training+texts_Testing
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts_training_and_testing_merged)
        return_dic[dataset_name].append(dictionary)
            
        # convert tokenized documents into a document-term matrix
        Corpus_Training = [dictionary.doc2bow(text) for text in texts_Training]
        return_dic[dataset_name].append(Corpus_Training)
        
        Corpus_Testing = [dictionary.doc2bow(text) for text in texts_Testing]
        return_dic[dataset_name].append(Corpus_Testing )

        del doc_set_Training
        del pubmed_identifier_list_Training
        del texts_Training
        del Corpus_Training
        del topic_doc_dictionary_Training

        del doc_set_Testing
        del pubmed_identifier_list_Testing
        del texts_Testing
        del Corpus_Testing
        del topic_doc_dictionary_Testing
        
        del dictionary
        del tokens
        del stopped_tokens
        del stemmed_tokens

    return (return_dic)





def document_filtering_and_spliting_into_training_and_testing_sets():
    training_doc_set={}
    testing_doc_set={}
    with open('F:\\publication work\\Data\\temp\\7.onlyDeeplyRelavenceDocsPythonNoDuplicteNosmallTopic=2403.txt', 'r') as f30:
        s30 = f30.read()
        pmid_topicwise = ast.literal_eval(s30)


    for key,value in pmid_topicwise.items():
        topic_no=key
        cursor2.execute("SELECT [TOPIC_NO],[SERIAL_NO],[PUBMED_IDENTIFIER],[ABSTRACT] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005] where TOPIC_NO="+str(topic_no)+";")
        resultset= cursor2.fetchall()
        numberOfDocsInResultset=len(resultset)
        if numberOfDocsInResultset>=30:
            testing_doc_set[topic_no]=[]
            training_doc_set[topic_no]=[]
            numberOfDocsForTraining=int(numberOfDocsInResultset/2)
            counter1=0
            for i in resultset:
                pubmed_identifier=i.PUBMED_IDENTIFIER
                if counter1<numberOfDocsForTraining:
                    training_doc_set[topic_no].append(pubmed_identifier)
                    counter1=counter1+1
                else:
                    testing_doc_set[topic_no].append(pubmed_identifier)
                    counter1=counter1+1
    
    with open('F:\\publication work\\Data\\temp\\7.1.training_doc_set.txt', 'w') as f31:
        f31.write(str(training_doc_set))
    
    with open('F:\\publication work\\Data\\temp\\7.2.testing_doc_set.txt', 'w') as f32:
        f32.write(str(testing_doc_set))



def data_set_creation():
    with open('F:\\publication work\\Data\\temp\\7.1.training_doc_set.txt', 'r') as f33:
            s33 = f33.read()
            training_doc_set1 = ast.literal_eval(s33)
    
    dataSetDictionary={}
    for i in range(0,100):
        nameOfDataset=''
        a=random.sample([3,4,5,6],1)
        b=random.sample(training_doc_set1.keys(),a[0])
        print b
        nameOfDataset='G2005'+str(i)
        print nameOfDataset
        dataSetDictionary[nameOfDataset]=b
    
    with open('F:\\publication work\\Data\\temp\\100_Dataset.txt','w') as f34:
        f34.write(str(dataSetDictionary))


        

def spectralClustering(number_of_topics_in_a_dataset,dataset_name): 
    f7 = open('F:\\publication work\\Data\\temp\\14.Similarity_from_cosine'+dataset_name+'.txt','r')
    df=json.load(f7)
    spectral = SpectralClustering(n_clusters=number_of_topics_in_a_dataset,affinity="precomputed")
    spectral.fit(df)
    
    spectral_label=list()
    
    for lab2 in spectral.labels_:
        spectral_label.append(lab2)
    with open('F:\\publication work\\Data\\temp\\15.spectral_labels'+dataset_name+'.txt','w') as f8:
        f8.write(str(spectral_label))
    
    del spectral_label
    del df
    del spectral
    f7.close()



def keepingOnlyProbability(number_of_topics_produced,dataset_name):
    dist_of_new_docs_over_topics_OnlyProb=list()
    dist_of_new_docs_indexer=-1
    f9 = open('F:\\publication work\\Data\\temp\\11.distribution_of_topics_in_docs_bracket_replaced'+dataset_name+'.txt','r')
    #f2 = open('F:\\Masters SPBSU\\4th semester\\Information retrival\\Output\\dist_of_docs_over_topics200OnlyProb.txt','w')
    
    df=json.load(f9)
    for i in df:
        dist_of_new_docs_over_topics_OnlyProb.append([])
        dist_of_new_docs_indexer+=1
        #print(i)
        for j in range(0,number_of_topics_produced):
            dist_of_new_docs_over_topics_OnlyProb[dist_of_new_docs_indexer].append(0)
        for x in i:
            dist_of_new_docs_over_topics_OnlyProb[dist_of_new_docs_indexer][x[0]]=x[1];
            
    with open('F:\\publication work\\Data\\temp\\12.distribution_of_topics_in_docs_bracket_replaced_only_prob'+dataset_name+'.txt','w') as f10:
        f10.write(str(dist_of_new_docs_over_topics_OnlyProb))
    
    del dist_of_new_docs_over_topics_OnlyProb
    del df
    f9.close()


def common_doc_in_class_and_counter(dataset_name):
    number_of_documents_common_in_class_and_cluster={}
    
    with open('F:\\publication work\\Data\\temp\\17.topic_doc_dictionary'+dataset_name+'.txt', 'r') as f11:
            s11 = f11.read()
            class_dictionary = ast.literal_eval(s11)    
    
    with open('F:\\publication work\\Data\\temp\\16.spectral_clustering_results'+dataset_name+'.txt', 'r') as f12:
            s12 = f12.read()
            cluster_dictionary = ast.literal_eval(s12)
    
    for key_class,value_class in class_dictionary.items():
        for key_cluster,value_cluster in cluster_dictionary.items():
          common_doc_counter=0
          for i in value_class:
              if i in value_cluster:
                  common_doc_counter=common_doc_counter+1
          number_of_documents_common_in_class_and_cluster[key_class,key_cluster]=common_doc_counter
    
    with open('F:\\publication work\\Data\\temp\\19.(Extra)number_of_documents_common_in_class_and_cluster'+dataset_name+'.txt','w') as f13:
        f13.write(str(number_of_documents_common_in_class_and_cluster))


def findMaximumSimilarityClusterWise(dataset_name):
    PairedClassClusterDicClusterWise={}
    with open('F:\\publication work\\Data\\temp\\16.spectral_clustering_results'+dataset_name+'.txt', 'r') as f14:
            s14 = f14.read()
            spectral_clustering_result = ast.literal_eval(s14)
    
    with open('F:\\publication work\\Data\\temp\\19.(Extra)number_of_documents_common_in_class_and_cluster'+dataset_name+'.txt', 'r') as f15:
            s15 = f15.read()
            number_of_documents_common_in_class_and_cluster = ast.literal_eval(s15)
        
    for i in spectral_clustering_result.keys():
        maximum=0
        classes=0    
        for j,k in number_of_documents_common_in_class_and_cluster.items():
            if j[1]==i:
                if k>maximum:
                    maximum=k
                    classes=j[0]
        PairedClassClusterDicClusterWise[classes,i]=maximum
    with open('F:\\publication work\\Data\\temp\\21.PairedClassClusterDicPrimaryClusterWise'+dataset_name+'.txt','w') as f17:
        f17.write(str(PairedClassClusterDicClusterWise))


def accuracyMeasure(dataset_name):
    number_of_documents_in_dataset=0
    number_of_documents_in_cluster={}
    number_of_documents_in_class={}
    sum_class=0
    sum_cluster=0
    sum_common=0

    with open('F:\\publication work\\Data\\temp\\16.spectral_clustering_results'+dataset_name+'.txt', 'r') as f18:
            s18 = f18.read()
            cluster_dictionary = ast.literal_eval(s18)
    for key, value in cluster_dictionary.items():
         number_of_documents_in_cluster[key]=len(value)
    number_of_documents_in_dataset=sum(number_of_documents_in_cluster.values())
    for i in number_of_documents_in_cluster.values():
        sum_cluster=sum_cluster+i*(math.log(i,2)-math.log(number_of_documents_in_dataset,2))
        

    with open('F:\\publication work\\Data\\temp\\17.topic_doc_dictionary'+dataset_name+'.txt', 'r') as f20:
            s20 = f20.read()
            class_dictionary = ast.literal_eval(s20)
    for key, value in class_dictionary.items():
         number_of_documents_in_class[key]=len(value)
    for i in number_of_documents_in_class.values():
        sum_class=sum_class+i*(math.log(i,2)-math.log(number_of_documents_in_dataset,2))


    with open('F:\\publication work\\Data\\temp\\19.(Extra)number_of_documents_common_in_class_and_cluster'+dataset_name+'.txt', 'r') as f19:
        s19 = f19.read()
        number_of_documents_common_in_class_and_cluster = ast.literal_eval(s19)    
    for key_class_cluster,value_class_cluster in number_of_documents_common_in_class_and_cluster.items():
        if value_class_cluster:
            sum_common=sum_common+value_class_cluster*(math.log(number_of_documents_in_dataset,2)+math.log(value_class_cluster,2)-math.log(number_of_documents_in_class[key_class_cluster[0]],2)-math.log(number_of_documents_in_cluster[key_class_cluster[1]],2))


    NMI=sum_common/math.sqrt(sum_class*sum_cluster)
    accuracy_dictionary[dataset_name]=NMI
    print 'The NMI of the dataset '+dataset_name+' is: '+str(NMI)

    
    del number_of_documents_in_cluster
    del number_of_documents_in_class
    del class_dictionary
    del cluster_dictionary




def combiningClusterResult(dataset_name):
    clusterResult={}
    with open('F:\\publication work\\Data\\temp\\8.pubmed_identifier'+dataset_name+'.txt', 'r') as f21:
            s21 = f21.read()
            whip = ast.literal_eval(s21)
    
    with open('F:\\publication work\\Data\\temp\\15.spectral_labels'+dataset_name+'.txt', 'r') as f22:
            s22 = f22.read()
            df = ast.literal_eval(s22)
    
    for index, items in enumerate(df):
        if items in clusterResult:
            clusterResult[items].append(whip[index])
        else:
            clusterResult[items]= list()
            clusterResult[items].append(whip[index])
    
    with open('F:\\publication work\\Data\\temp\\16.spectral_clustering_results'+dataset_name+'.txt','w') as f23:
        f23.write(str(clusterResult))
    
    del clusterResult
    del whip
    del df





def replaceBrackets(dataset_name):
    f26 = open('F:\\publication work\\Data\\temp\\10.distribution_of_topics_in_docs'+dataset_name+'.txt','r')
    f27 = open('F:\\publication work\\Data\\temp\\11.distribution_of_topics_in_docs_bracket_replaced'+dataset_name+'.txt','w')
    for i in f26:
        x=i.replace("(","[")
        y=x.replace(")","]")
    f27.write(y)
    
    f26.close()
    f27.close()




def cosineSimilarity(dataset_name):
    f = open('F:\\publication work\\Data\\temp\\12.distribution_of_topics_in_docs_bracket_replaced_only_prob'+dataset_name+'.txt','r')
    df=json.load(f)
    #p=[[0,1], [1,1]]
    A =  np.array(df)
    A_sparse = sparse.csr_matrix(A)
    
    similarities = cosine_similarity(A_sparse)
    #print('pairwise dense output:\n {}\n'.format(similarities))
    similaritiesToWrite=[]
    similaritiesToWriteindexer=-1
    for i in similarities:
        similaritiesToWriteindexer=similaritiesToWriteindexer+1
        similaritiesToWrite.append([])
        for j in i:
            similaritiesToWrite[similaritiesToWriteindexer].append(j)
    with open('F:\\publication work\\Data\\temp\\14.Similarity_from_cosine'+dataset_name+'.txt','w') as f2:
        f2.write(str(similaritiesToWrite))



#****************************global_space_starts_here****************************
model_training_and_testing_also_main_function()
cursor.close()
cursor1.close()
cursor2.close()
cursor3.close()
cursor4.close()
cnxn.close()
#****************************global_space_ends_here******************************
