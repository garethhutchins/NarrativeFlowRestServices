#math
import math
import numpy as np
import pandas as pd
#NLTK for speach
import nltk
#nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
#nltk.download('wordnet')
#download pos
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
import re
#Scikit for learning mechanisms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#A function to perform porter stemming
def remove_stop_stem(text):
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keepWords = []
    for r in words:
        if not r in stop_words:
            if r.isalpha():
                stemmed = porter.stem(r.lower())
                keepWords.append(stemmed)
    #convert it into a flat string
    kwstring = ' '.join(keepWords)
    return kwstring

#Now create a function to lemmatise and tokenize the text
def remove_stop_lem(text):
    stop_words = set(stopwords.words('english'))
    
    words = word_tokenize(text)
    keepWords = []
    for r in words:
        if not r in stop_words:
            if r.isalpha():
                lemmatized = lemmatizer.lemmatize(r.lower())
                keepWords.append(lemmatized)
    #convert it into a flat string
    kwstring = ' '.join(keepWords)
    return kwstring 

#This example came from the SKlearn tutorial
#https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

def plot_tfidf(lables,word_scores):
    num_lables = len(lables)
    rows = math.ceil(num_lables/5)
    fs = 15*rows
    fig, axes = plt.subplots(rows, 5, figsize=(fs, fs), sharex=True)
    axes = axes.flatten()
    y = 0
    title = 'TF-IDF Topics'
    for lable in lables:
        df = word_scores[y]
        ax = axes[y]
        ax.barh(df['Words'], df['Score'], height=0.7)
        ax.set_title(lable,
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
        y += 1
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return plt  

def plot_top_words(model, feature_names, n_top_words, title,num_topics,topic_names):
    rows = math.ceil(num_topics/5)
    fs = 15*rows
    fig, axes = plt.subplots(rows, 5, figsize=(fs, fs), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        if len(topic_names) == 0:
            ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        else:
            topic_title = topic_names[str(topic_idx +1)]
            ax.set_title(topic_title,
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return plt   
     
#Train the NMF model
def train_nmf(df,num_topics):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(df)
    tfidf_feature_names=tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=num_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
    n_top_words = 10
    plot_image = plot_top_words(nmf, tfidf_feature_names, n_top_words,
               'Topics in NMF model',num_topics,{})
    #Now return the vectorizer, the tfidf and the plot
    return tfidf_vectorizer, nmf, plot_image
#Predict the number of LDA topics
def predict_lda_topics(df):
    search_params = {'n_components': [5,6,7,8,9,10,11,12]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=1000,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(df)
    # Do the Grid Search
    model.fit(tf)
    # Best Model
    best_lda_model = model.best_estimator_
    
    num_topics = best_lda_model.n_components
    return int(num_topics)

#Train the LDA Model
def train_lda(df,num_topics):
    #Create a tf vectorizer
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=1000,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(df)
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tf)
    n_top_words = 10
    plot_image = plot_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words, 'Topics in LDA model',num_topics,{}) 
    return tf_vectorizer, lda, plot_image

#Train TF-IDF models
def train_tfidf(text,labels):
    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    #Start making changes for multi lables
    # Loop though each text entry
    out_text = []
    out_lable = []
    row_count = 0
    for t in text:
        if labels[row_count] == np.nan:
            row_count +=1
            continue
        if type(labels[row_count]) != str:
            row_count +=1
            continue
        else:
            l_list = labels[row_count].split(',')
        
        row_count += 1
        #Now copy the text and add the lables so we have duplicate texts with different lables
        for l in l_list:
            out_text.append(t)
            out_lable.append(l)


    #End making changes for multi lables
    X = vectorizer.fit_transform(out_text)
    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(
            X, out_lable, test_size=0.3, random_state=42
        )
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf',probability=True)
    svm.fit(X_train, y_train)
    score = svm.score(X_test,y_test)
    
    #Get the unique labels and loop
    u_lbls = set(out_lable)
    con_tbl = pd.DataFrame()
    con_tbl['Text'] = out_text
    con_tbl['Category'] = out_lable
    
    lbls = []
    top_words = []
    for x in u_lbls:
        #Now only select rows for this label
        c_text = con_tbl[con_tbl['Category'] == x]
        # Now combine all of the text for that category
        combine_text = ' '.join(c_text['Text'])
      
        response = vectorizer.transform([combine_text])
        feature_names = vectorizer.get_feature_names()
        words = []
        scores = []
        for col in response.nonzero()[1]:
            words.append(feature_names[col])
            scores.append(response[0, col])
        #Now convert into a dataframe and sort
        df = pd.DataFrame()
        df['Words'] = words
        df['Score'] = scores
        
        df = df.sort_values(by='Score',ascending=False)            
        df = df.head(20)
        lbls.append(x)
        top_words.append(df)
    
    plot_image = plot_tfidf(lbls,top_words)
    return vectorizer, svm, score, plot_image,lbls


