# import libraries
from sklearn.base import BaseEstimator as estimator
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# set of english stopwords
stop_words = set(stopwords.words('english'))

# function for preprocessing messages
def preprocess_text(text:str) -> list[str]:
    '''
    Function to preprocess text
    Args:
        text (str): The text to preprocess
    Returns:
        list[str]: The preprocessed text
    '''
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Converting all text to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Removing empty strings
    tokens = [token for token in tokens if token != '']
    
    return tokens

def score(text:str, model:estimator, threshold:float = 0.5) -> tuple[bool,float]:
    '''
    Function to score a trained model on a given text
    Args:
        text (str): The text to score
        model (estimator): The model to score
        threshold (float): The threshold to use for the prediction (default: 0.5)
    Returns:
        tuple[bool,float]: The prediction and the propensity score
    '''
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    if not isinstance(model, estimator):
        raise ValueError("Model must be an instance of sklearn BaseEstimator")
    
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")
    
    # Preprocess the text
    tokens = preprocess_text(text)
    tokens = str(tokens)

    # Get the propensity score
    propensity = model.predict_proba([tokens])[0][1]

    # Make a prediction based on the threshold
    prediction = propensity > threshold

    return bool(prediction.item()), float(propensity)