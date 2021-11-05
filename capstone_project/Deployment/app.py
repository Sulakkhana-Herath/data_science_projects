
import pandas as pd
import spacy
import regex as re
import joblib

from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer



class MachineLearningModel:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.rating_model = joblib.load(open('sentiment_model.joblib','rb'))
        self.recommendation_model = joblib.load(open('recommendation_model.joblib','rb'))

    def clean_text(self, text):
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)

        # remove double quotes
        text = re.sub(r'"', '', text)
        # remove special characters, numbers, punctuations
        text = re.sub(r'[^a-zA-Z#]', ' ', text)
        # remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ' , text)
        # reduce multiple spaces and newlines to only one
        text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)
        return text

    # remove stop words, puctuations and  perform lemmatization
    def convert_text(self, text):
        doc = self.nlp(text)
        t = [w for w in doc if not (w.is_stop | w.is_punct)]
        x = [w.lemma_.lower() for w in t]

        s= " ".join(x)
        return s

    def count_num(self, text):
        doc = self.nlp(text)
        count = 0
        for w in doc:
            if w.pos_ == "NOUN":
                count+=1
        return count

    def adj_count(self, text):
        doc = self.nlp(text)
        count = 0
        for w in doc:
            if w.pos_ == "ADJ":
                count+=1
        return count

    def adv_count(self, text):
        doc = self.nlp(text)
        count = 0
        for w in doc:
            if w.pos_ == "ADV":
                count+=1
        return count

    def propn_count(self, text):
        doc = self.nlp(text)
        count = 0
        for w in doc:
            if w.pos_ == "PRON":
                count+=1
        return count

    def num_count(self, text):
        doc = self.nlp(text)
        count = 0
        for w in doc:
            if w.pos_ == "NUM":
                count+=1
        return count

    def preprocess(self, x):
        x_dict = {"X": [x]}
        df = pd.DataFrame(x_dict)
        df['X'].apply(self.clean_text)
        df['X'].apply(self.convert_text)
        df['Polarity'] = df['X'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['char count'] = df['X'].apply(len)
        df['word count'] = df['X'].apply(lambda x: len(x.split()))
        df['avg word lenth'] = df['char count'] / (df['word count']+1)
        df['avg sentence lenght'] = df['word count'] / (df['X'].apply(lambda x: len(str(x).split("."))))
        df['noun_count'] = df['X'].apply(self.count_num)
        df['adj_count'] = df['X'].apply(self.adj_count)
        df['adv_count'] = df['X'].apply(self.adv_count)
        df['num_count'] = df['X'].apply(self.num_count)

        tfidf_vect = TfidfVectorizer(analyzer = 'word',
                                    token_pattern = r'\w{1,}',
                                    max_features = 500)
        tfidf=tfidf_vect.fit_transform(df['X']).toarray()
        text_vect = pd.DataFrame(tfidf)
        for c in range(len(text_vect.columns), 500, 1):
            text_vect[str(c)] = 0

        features = ['Polarity','char count','word count','avg word lenth','avg sentence lenght','noun_count','adj_count','adv_count','num_count']
        return pd.merge(text_vect, df[features], left_index=True, right_index=True)

    def predict_rating(self, x):
        return self.rating_model.predict(self.preprocess(x))

    def predict_recommendation(self, x):
        return self.recommendation_model.predict(self.preprocess(x))


app = Flask(__name__)
model = MachineLearningModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    input = request.form['text']
    predicted_rating = model.predict_rating(input)[0]
    predicted_recommendation = model.predict_recommendation(input)[0]

    if predicted_rating == -1:
        rating_output = 'negative'
    elif predicted_rating == 0:
        rating_output = 'neutral'
    else:
        rating_output = 'positive'

    if predicted_recommendation == 0:
        recommendation_output = 'not recommended'
    else:
        recommendation_output = 'recommended'

    return render_template('index.html', sentiment=f'Sentiment is {rating_output} and Product is {recommendation_output}.')


if __name__ == "__main__":
    app.run(debug=True)