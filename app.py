import pickle
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from langdetect import detect, LangDetectException

#LOAD THE Tokenizer
with open('sentiment_analysis_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# LOAD THE MODEL
model_parts = []
for i in range(11): 
    part_file = f'sentiment_analysis_model_part_{i}.pkl'
    with open(part_file, 'rb') as f:
        model_part = pickle.load(f)
        model_parts.append(model_part)

model = tf.keras.Sequential(model_parts)

# PRIDECTION FN.
def predict_sentiment(tweet):
    encoding = tokenizer(tweet, truncation=True, padding=True, return_tensors='tf')
    logits = model(encoding['input_ids'])  
    predicted_class = tf.argmax(logits, axis=-1).numpy()[0]
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment

test_tweets = [
    "I love the way this app works, it's absolutely fantastic!",
    "I can't believe how bad the service was, it ruined my day.",
    "This is the worst movie I've ever watched, such a waste of time!",
    "I'm feeling really happy today, everything is going great!",
    "The product is okay, but I think it could be better."
]

for tweet in test_tweets:
    print(f"Tweet: '{tweet}' => Sentiment: {predict_sentiment(tweet)}")
