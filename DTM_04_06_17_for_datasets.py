# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 10:12:24 2017

@author: Mahedi Hasan
"""


from gensim import corpora
from gensim.models.wrappers.dtmmodel import DtmModel

import pyodbc
import ast
import json
import math

from sklearn.cluster import SpectralClustering
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from SimilarityFromJSD import similarityFromJSD
#from ReplaceBrackets import replaceBrackets
from DistanceFromJSDFromOnlyProb import distanceFromJSD
#from DistanceFromJSDFromReplaceBrackets import distanceFromJSD


#**********parameters*************
number_of_passes=55
#*********************************

#**********Global variales*************
accuracy_dictionary={}
#*********************************

def DTMimplementForDatasets():
    cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-P61DTNE;DATABASE=Medline;UID=sa;PWD=0000')
    cursor = cnxn.cursor()
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    
    with open('F:\\publication work\\Data\\temp\\sample_dataset.txt', 'r') as f2:
            s2 = f2.read()
            all_dataset = ast.literal_eval(s2)
    
    for key,value in all_dataset.items():
        dataset_name=key
        stri='where'
        j=0
        for i in value:
            if (j==0):
                stri=stri+' '+'TOPIC_NO='+str(i)
                j=j+1
            else:
                stri=stri+' or '+'TOPIC_NO='+str(i)
    # Declare list to create a list of the whole document set
        doc_set = list()
        list_of_topics=list()
        temp_dist_of_docs_over_topics=list()
        dist_of_docs_over_topics=list()
        pubmed_identifier_list=list()
        topic_doc_dictionary={}
        cursor.execute("SELECT [TOPIC_NO],[SERIAL_NO],[PUBMED_IDENTIFIER],[ABSTRACT] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005]"+stri+";")
        for row1 in cursor.fetchall():
            abstract= row1.ABSTRACT.strip()
            serial_no= row1.SERIAL_NO
            topic_no=row1.TOPIC_NO
            pubmed_identifier=row1.PUBMED_IDENTIFIER
            pubmed_identifier_list.append(pubmed_identifier)
            if topic_no in topic_doc_dictionary.keys():
                topic_doc_dictionary[topic_no].append(pubmed_identifier)
            else:
                topic_doc_dictionary[topic_no]= list()
                topic_doc_dictionary[topic_no].append(pubmed_identifier)
            number_of_topics_in_a_dataset=len(topic_doc_dictionary.keys())
            if not abstract:
                cursor1 = cnxn.cursor()
                cursor1.execute("SELECT [TITLE] FROM [Medline].[dbo].[OnlyDeeplyRelaGeno2005] where SERIAL_NO='"+str(serial_no)+"';")
                for row in cursor1.fetchall():
                   abstract=row.TITLE.strip()
                   #print k
            doc_set.append(abstract)
        #number_of_topics_produced=10
        number_of_topics_produced=len(doc_set)/24
        #number_of_topics_produced=len(doc_set)/70
        print 'Number of documents: '+str(len(doc_set))  
        print 'Number of topics produced: '+str(number_of_topics_produced)
        print 'Number of passes: '+str(number_of_passes)
        print 'Number of clusters: '+str(number_of_topics_in_a_dataset)
        
        
        
        # Declaring list for tokenized documents in loop
        texts = []
        
        # loop through document list
        for i in doc_set:
            
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            #print(tokens)
        
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            
            # add tokens to list
            texts.append(stemmed_tokens)
        
        
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        #print(dictionary.token2id)
            
        # convert tokenized documents into a document-term matrix
        Corpus = [dictionary.doc2bow(text) for text in texts]
        #print(corpus[0])
        
        """
        class DTMcorpus(corpora.textcorpus.TextCorpus):
        
            def get_texts(self):
                return self.input
        
            def __len__(self):
                return len(self.input)
        
        corpus = DTMcorpus(texts)
        
        """
        #if len(doc_set)*30%100:  
        #    section30=len(doc_set)*30/100+1
        #else:
        #   section30=len(doc_set)*30/100
        #section70=len(doc_set)*70/100
        #time_seq = [section30, section70]
        time_seq = [len(doc_set), 0]
        dtm_path = 'C:\Program Files\DTM\dtm-win64.exe'
        dtmModel = DtmModel(dtm_path, Corpus, time_seq, num_topics=number_of_topics_produced,id2word=dictionary, initialize_lda=True)
        
        for i in range(0,number_of_topics_produced):
            list_of_topics.append(dtmModel.show_topic(i,1,10))
        
        for i in range(0,len(doc_set)):
            temp_dist_of_docs_over_topics.append(dtmModel.gamma_[i])
            
        dist_of_docs_over_topics=[]
        dist_of_docs_over_topicsindexer=-1
        for i in temp_dist_of_docs_over_topics:
            dist_of_docs_over_topicsindexer=dist_of_docs_over_topicsindexer+1
            dist_of_docs_over_topics.append([])
            for j in i:
                dist_of_docs_over_topics[dist_of_docs_over_topicsindexer].append(j)
        

            
        with open('F:\\publication work\\Data\\temp\\8.pubmed_identifier.txt', 'w') as f3:
            f3.write(str(pubmed_identifier_list))
        
        with open('F:\\publication work\\Data\\temp\\9.topics_list.txt', 'w') as f1:
            f1.write(str(list_of_topics))
        
        #print dist_of_new_docs_over_topics
        with open('F:\\publication work\\Data\\temp\\12.distribution_of_topics_in_docs_bracket_replaced_only_prob.txt', 'w') as f5:
            f5.write(str(dist_of_docs_over_topics))
        
        with open('F:\\publication work\\Data\\temp\\17.topic_doc_dictionary.txt','w') as f4:
            f4.write(str(topic_doc_dictionary))
        
        del doc_set
        del list_of_topics
        del dist_of_docs_over_topics
        del pubmed_identifier_list
        del stopped_tokens
        del stemmed_tokens
        del texts
        del dictionary
        del Corpus
        del tokens
        del dtmModel
        del topic_doc_dictionary
        

        
        
        #replaceBrackets()
        #print 'Bracket replacing completed'
        
        #keepingOnlyProbability(number_of_topics_produced)
        #print 'Keeping only probability completed'
        
        distanceFromJSD()
        print 'Measuring distance completed'
        
        similarityFromJSD()
        print 'Measuring similarity completed'
        
        spectralClustering(number_of_topics_in_a_dataset)
        print 'Spectral clustering completed'
        
        combiningClusterResult(dataset_name)
        print 'Combining clustering result completed'
        
        accuracyMeasure(dataset_name)
        print 'Measuring accuracy completed'
        
    with open('F:\\publication work\\Data\\temp\\18.NMI_dictionary.txt','w') as f6:
        f6.write(str(accuracy_dictionary))
        
        
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    cursor.close()
    cursor1.close()
    cnxn.close()



def spectralClustering(number_of_topics_in_a_dataset): 
    f6 = open('F:\\publication work\\Data\\temp\\14.document_similarity_from_jsd.txt','r')
    df=json.load(f6)
    spectral = SpectralClustering(n_clusters=number_of_topics_in_a_dataset,affinity="precomputed")
    spectral.fit(df)
    
    spectral_label=list()
    
    for lab2 in spectral.labels_:
        spectral_label.append(lab2)
    with open('F:\\publication work\\Data\\temp\\15.spectral_labels.txt','w') as f7:
        f7.write(str(spectral_label))
    
    del spectral_label
    del df
    del spectral
    f6.close()
    f7.close()



def keepingOnlyProbability(number_of_topics_produced):
    dist_of_new_docs_over_topics_OnlyProb=list()
    dist_of_new_docs_indexer=-1
    f8 = open('F:\\publication work\\Data\\temp\\11.distribution_of_topics_in_docs_bracket_replaced.txt','r')
    #f2 = open('F:\\Masters SPBSU\\4th semester\\Information retrival\\Output\\dist_of_docs_over_topics200OnlyProb.txt','w')
    
    df=json.load(f8)
    for i in df:
        dist_of_new_docs_over_topics_OnlyProb.append([])
        dist_of_new_docs_indexer+=1
        #print(i)
        for j in range(0,number_of_topics_produced):
            dist_of_new_docs_over_topics_OnlyProb[dist_of_new_docs_indexer].append(0)
        for x in i:
            dist_of_new_docs_over_topics_OnlyProb[dist_of_new_docs_indexer][x[0]]=x[1];
            
    with open('F:\\publication work\\Data\\temp\\12.distribution_of_topics_in_docs_bracket_replaced_only_prob.txt','w') as f9:
        f9.write(str(dist_of_new_docs_over_topics_OnlyProb))
    
    del dist_of_new_docs_over_topics_OnlyProb
    del df
    f8.close
    f9.close




def accuracyMeasure(dataset_name):
    number_of_documents_in_dataset=0
    number_of_documents_in_cluster={}
    number_of_documents_in_class={}
    number_of_documents_common_in_class_and_cluster={}
    sum_class=0
    sum_cluster=0
    sum_common=0
    
    with open('F:\\publication work\\Data\\temp\\17.topic_doc_dictionary.txt', 'r') as f2:
            s2 = f2.read()
            class_dictionary = ast.literal_eval(s2)
    for key, value in class_dictionary.items():
         number_of_documents_in_class[key]=len(value)
    number_of_documents_in_dataset=sum(number_of_documents_in_class.values())
    for i in number_of_documents_in_class.values():
        sum_class=sum_class+i*(math.log(i)-math.log(number_of_documents_in_dataset))
        
    
    with open('F:\\publication work\\Data\\temp\\16.spectral_clustering_results'+dataset_name+'.txt', 'r') as f1:
            s1 = f1.read()
            cluster_dictionary = ast.literal_eval(s1)
    for key, value in cluster_dictionary.items():
         number_of_documents_in_cluster[key]=len(value)
    for i in number_of_documents_in_cluster.values():
        sum_cluster=sum_cluster+i*(math.log(i)-math.log(number_of_documents_in_dataset))
    
    for key_class,value_class in class_dictionary.items():
        for key_cluster,value_cluster in cluster_dictionary.items():
          common_doc_counter=0
          for i in value_class:
              if i in value_cluster:
                  common_doc_counter=common_doc_counter+1
          number_of_documents_common_in_class_and_cluster[key_class,key_cluster]=common_doc_counter
          a=math.log(number_of_documents_in_dataset)
          b=0.000000000000000000
          if common_doc_counter:
              b=math.log(common_doc_counter)
          c=math.log(number_of_documents_in_class[key_class])
          d=math.log(number_of_documents_in_cluster[key_cluster])
          sum_common=sum_common+common_doc_counter*(a+b-c-d)
    
    NMI=sum_common/math.sqrt(sum_class*sum_cluster)
    accuracy_dictionary[dataset_name]=NMI
    print 'The NMI is: '+str(NMI)
    
    #with open('F:\\publication work\\Data\\temp\\18.NMI_dictionary.txt','w') as f3:
     #   f3.write(str(accuracy_dictionary))
    
    with open('F:\\publication work\\Data\\temp\\19.(Extra)number_of_documents_common_in_class_and_cluster.txt','w') as f4:
        f4.write(str(number_of_documents_common_in_class_and_cluster))
    
    del number_of_documents_in_cluster
    del number_of_documents_in_class
    #del number_of_documents_common_in_class_and_cluster
    del class_dictionary
    del cluster_dictionary
    
    f1.close()
    f2.close()
    #f3.close()
    f4.close()



def combiningClusterResult(dataset_name):
    clusterResult={}
    with open('F:\\publication work\\Data\\temp\\8.pubmed_identifier.txt', 'r') as f:
            s = f.read()
            whip = ast.literal_eval(s)
    
    with open('F:\\publication work\\Data\\temp\\15.spectral_labels.txt', 'r') as f1:
            s1 = f1.read()
            df = ast.literal_eval(s1)
    
    for index, items in enumerate(df):
        if items in clusterResult:
            clusterResult[items].append(whip[index])
        else:
            clusterResult[items]= list()
            clusterResult[items].append(whip[index])
    
    with open('F:\\publication work\\Data\\temp\\16.spectral_clustering_results'+dataset_name+'.txt','w') as f2:
        f2.write(str(clusterResult))
    
    del clusterResult
    del whip
    del df
    f.close()
    f1.close()
    f2.close()


DTMimplementForDatasets()