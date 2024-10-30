# ArticleRecommandationWithNLP


This project is made for a class homework. It is a article recommendation website using natural language processing(NLP).
We used Django Framework for website, Python for NLP operations, HTML for page templates and MySql for database. We used [Inspec](https://huggingface.co/datasets/midas/inspec/tree/main) dataset for articles. 

The website includes login, register, home and detail pages. When user register to the website, user should fill their name, surname, e-mail, password and their interests about articles. These informations are saved to MySql database. User interest data is used for creating vectors and calculating similarity between user interests and article vectors.Home page has search bar, recommended articles based on vectors' similarity and list of clickable article titles. In detail page, you can see clicked article's detailed version.

Firstly, we preprocess article in the dataset using NLTK library in Python. These are including removing stopwords, punctiations, and lemmatization:
* We tokenize each article then preprocess them and after that we save them to a csv file. You can see said .csv file in the repository.
* Other operation was to use FastText and SCIBERT models on preprocessed data. We use these models to create vectors for each article. Also for logged in user's interests, we again use these models to create vector embeddings.
* After calculating vectors for both articles and user interest, we used cosine similarity to calculate similarity and recommend 5 article based on that.
* In home page, we display these recommended top 5 articles both for SCIBERT and FastText. Also in searchbar there, we can filter articles based on typed keywords.

In repository, you can also see articles after preprocessing operations in **preprocessedData.csv**. Also in preprocessedData.csv file there are two additional columns which are FastText and SCIBERT vector embeddings of each article. User interest vectors are saved to **user_vectors.csv** for FastText and **user_vectors_forSci.csv** for SCIBERT. Like I said before these vectors are used to calculate similarity. After that calculation we filtered top 5 similar articles both for SCIBERT and FastText model and saved them to **top_articles_scibert.csv** and **top_articles_recommendations.csv** respectively.

All article preprocessing, creating vectors and calculating similarities are coded in preprocessing.py file in articleapp folder. If you are only interested in natural language processing(NLP) part, you can download Inspec dataset, try it out and see results in your own csv files.

There are some images from the app: 
<p float="left">
  <img src="https://github.com/cigdeemtok/ArticleRecommandationWithNLP/blob/main/images/register_page.jpg" width="75%" />
  <br>
  <img src="https://github.com/cigdeemtok/ArticleRecommandationWithNLP/blob/main/images/login_page.jpg" width="75%" />
  <br>
  <img src="https://github.com/cigdeemtok/ArticleRecommandationWithNLP/blob/main/images/home_page.jpg" width="75%" />
  <br>
  <img src="https://github.com/cigdeemtok/ArticleRecommandationWithNLP/blob/main/images/article_detail.jpg" width="75%" />
</p>
This is the database structure for users and articles:
<p float="left">
  <img src="https://github.com/cigdeemtok/ArticleRecommandationWithNLP/blob/main/images/db_articles.jpg" width="75%" />
  <br>
  <img src="https://github.com/cigdeemtok/ArticleRecommandationWithNLP/blob/main/images/db_user.jpg" width="75%" />
  <br>
</p>

You can clone this project and make sure you have installed everything you need like libraries, database, FastText and SCIBERT models etc. And after that you can run the project by running this command: 
```
python manage.py runserver
```

We load the dataset and save it to database after login. And also we use functions that we defined in preprocessing.py in home section of website. So all computation and more is going to be made there. Because recommendations can differ based on different user interests. Because of all of that, after you run the project it can take some time to load home page, you should be aware of that. 

Any feedback is appreciated. I hope you find this project helpful. 
