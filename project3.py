import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("chatbot.txt", "r", errors="ignore") as f:
    raw = f.read().lower()


sent_tokens = nltk.sent_tokenize(raw)


from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "hey", "greetings", "sup")
GREETING_RESPONSES = ["hi", "hey", "hello", "I’m glad you’re talking to me!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    user_response = user_response.lower()
    sent_tokens.append(user_response)  
    
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, token_pattern=None)
    tfidf = vectorizer.fit_transform(sent_tokens)
    
    vals = cosine_similarity(tfidf[-1], tfidf)  
    idx = vals.argsort()[0][-2]
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    sent_tokens.pop(-1) 
    
    if req_tfidf == 0:
        return "I’m sorry! I don’t understand you."
    else:
       
        if sent_tokens[idx].endswith("?") and idx + 1 < len(sent_tokens):
            return sent_tokens[idx + 1]
        else:
            return sent_tokens[idx]


print("Chatbot: I am your chatbot. Type 'bye' to exit.")

while True:
    user_response = input("You: ")
    if user_response.lower() == 'bye':
        print("Chatbot: Bye! Take care.")
        break
    elif user_response.lower() in ('thanks', 'thank you'):
        print("Chatbot: You’re welcome!")
    else:
        if greeting(user_response) is not None:
            print("Chatbot:", greeting(user_response))
        else:
            print("Chatbot:", response(user_response))
