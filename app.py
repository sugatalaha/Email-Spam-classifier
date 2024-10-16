import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email spam classifier')

input_message=st.text_area("Enter your email")
if st.button('Predict'):

    transformed_message=transform_text(input_message)

    vector_input=tfidf.transform([transformed_message])

    result=model.predict(vector_input)[0]

    if result==0:
        st.header('Not spam')
    else:
        st.header('Spam!')
