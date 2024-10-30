from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
import spacy
import pandas as pd
import string
import fasttext
import fasttext.util
import numpy as np
from .models import User
from transformers import AutoModel,AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

dataset = load_dataset("memray/inspec")




nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

stopWords = set(stopwords.words('english'))

def preprocessArticle(article):
    
    tokens = word_tokenize(article.lower())
    removedPunc = []

    for token in tokens:
        if (token not in string.punctuation):
            removedPunc.append(token)
    
    tokens = removedPunc

    removedStop = []

    for token in tokens:
        if(token not in stopWords):
            removedStop.append(token)

    tokens = removedStop
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    article = ' '.join(tokens)
    return article
    
def preprocessKeywords(keywords):
    newKeywords = []
    keywords = keywords.split(';')
 
    for key in keywords : 
       
        key = ''.join(key)
        
        key = word_tokenize(key.lower())
      
        removedKeyPunc = []
        for k in key:
            if(k not in string.punctuation):
                removedKeyPunc.append(k)    


        key =removedKeyPunc

        removedStopKey = []
        for k in key : 
            if(k not in stopWords): 
                removedStopKey.append(k)

        key = removedStopKey


        key = [lemmatizer.lemmatize(k) for k in key]
        key = ' '.join(key)
        newKeywords.append(key)

    keywords = newKeywords 
    return keywords


newArticles = []
newKeywords = []
articleIds = []
for keys in dataset.keys():
    for data in dataset[keys]:
        artID = data['name']
        articleText = ''.join(data['abstract'])
        newArticle = preprocessArticle(articleText)
        newArticles.append(newArticle)

        keywordsText = data['keywords']
        # print(keywordsText)
        newKey = preprocessKeywords(keywordsText)
        newKeywords.append(newKey)
        articleIds.append(artID)

preprocessedData = pd.DataFrame({
    'name' : articleIds,
    'preprocessed_articles' : newArticles,
    'preprocessed_keywords' : newKeywords,
})
# print(dataset['train']['id'][0])
preprocessedData.to_csv('preprocessedData.csv',index = False)

df = pd.read_csv('preprocessedData.csv')

#fasttext ile vektör oluşturma
fasttext.util.download_model('en',if_exists='ignore')
ftext = fasttext.load_model('cc.en.300.bin')

#scibert
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

def compute_fasttext_vector(texts):
    vectors = [ftext.get_sentence_vector(text) for text in texts]
    if vectors:
        avg_vector = np.mean(vectors, axis=0)
    else:
        avg_vector = np.zeros(300)  # Vektör boyutu 300
    return avg_vector.tolist()

article_vectors = []
sci_vectors = []

def calculateScibert(text):
    vectorInputs = tokenizer(text, return_tensors='pt',padding=True)

    vectorOutputs = model(**vectorInputs)
    
    return (vectorOutputs.last_hidden_state[:,0,:].squeeze().detach().numpy()).tolist()


for (_, row) in df.iterrows():
    preArticle = row['preprocessed_articles']

    sci_vector  = calculateScibert([preArticle])

    article_vector = compute_fasttext_vector([preArticle])

    article_vectors.append(article_vector)
    sci_vectors.append(sci_vector)
    

df['fasttext_vector'] = article_vectors

df['sci_vector'] = sci_vectors


df.to_csv('preprocessedData.csv', index=False)




def compute_interest_vectors(userId,ilgi_alanlari):
    # users = User.objects.all()
    user_vectors = []

    # for user in users:
    #interests = ilgi_alanlari.split(',')
    interest_vectors = []

    for interest in ilgi_alanlari:
        vector = ftext.get_sentence_vector(interest)
        interest_vectors.append(vector)

        # Kullanıcının ilgi alanlarının ortalama vektörünü hesaplama
    if interest_vectors:
        avg_vector = np.mean(interest_vectors, axis=0)
    else:
        avg_vector = [0] * 300  # Vektör boyutu 300

    user_vectors.append({
        'user_id': userId,
        'average_vector': avg_vector.tolist()
    })

    # CSV dosyasına yazma
    user_vector_df = pd.DataFrame(user_vectors)
    user_vector_df.to_csv('user_vectors.csv', index=False)

def compute_interest_vectors_forSci(userId,ilgi_alanlari):
    # users = User.objects.all()
    user_vectors = []

    # for user in users:
    #interests = ilgi_alanlari.split(',')
    interest_vectors = []

    for interest in ilgi_alanlari:
        vector = calculateScibert(interest)
        interest_vectors.append(vector)

        # Kullanıcının ilgi alanlarının ortalama vektörünü hesaplama
    if interest_vectors:
        avg_vector = np.mean(interest_vectors, axis=0)
    else:
        avg_vector = [0] * 768  # Vektör boyutu 300

    user_vectors.append({
        'user_id': userId,
        'sci_vector': avg_vector.tolist()
    })

    # CSV dosyasına yazma
    user_vector_df = pd.DataFrame(user_vectors)
    user_vector_df.to_csv('user_vectors_forSci.csv', index=False)

def safe_eval_vector(vector_str):
    try:
        return np.array(ast.literal_eval(vector_str))
    except (ValueError, SyntaxError):
        return np.zeros(300)

#cosine similarity
# Load article vector
def cosineHesapla():
    article_df = pd.read_csv('preprocessedData.csv')

    # Load user vectors
    user_vector_df = pd.read_csv('user_vectors.csv')
    user_sciVector_df = pd.read_csv('user_vectors_forSci.csv')


    article_vectors = np.stack(article_df['fasttext_vector'].apply(safe_eval_vector).values)

    sci_vectors = np.stack(article_df['sci_vector'].apply(safe_eval_vector).values)

    article_ids = article_df['name'].values

    user_vector = np.array(ast.literal_eval(user_vector_df['average_vector'][0]))
    userSci_vector = np.array(ast.literal_eval(user_sciVector_df['sci_vector'][0]))

    user_vector = user_vector.reshape(1, -1)
    userSci_vector = userSci_vector.reshape(1, -1)

    similarities = cosine_similarity(user_vector, article_vectors)[0]

    sci_similarities = cosine_similarity(userSci_vector,sci_vectors)[0]

    top_articles_df = pd.DataFrame({
        'article_name': article_ids,
        'similarity_score': similarities
    })
    top_sciArticles_df = pd.DataFrame({
        'article_name': article_ids,
        'sci_similarity': sci_similarities
    })

    top_articles_df = top_articles_df.sort_values(by='similarity_score', ascending=False).head(5)

    top_sciArticles_df = top_sciArticles_df.sort_values(by='sci_similarity', ascending=False).head(5)


    top_articles_df.to_csv('top_articles_recommendations.csv', index=False)
    top_sciArticles_df.to_csv('top_articles_scibert.csv', index=False)

def showRecommandations():
     topArticles_df = pd.read_csv('top_articles_recommendations.csv')

     article_ids = topArticles_df['article_name'].values

     return article_ids
def showSciRecommendation():
     topSciArticles_df = pd.read_csv('top_articles_scibert.csv')

     article_idsSci = topSciArticles_df['article_name'].values

     return article_idsSci