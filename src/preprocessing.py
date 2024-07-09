from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from bs4 import BeautifulSoup


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(texts):
    preprocessed_texts = []
    for text in texts:
        if isinstance(text, str):
            # Tokenize
            words = nltk.word_tokenize(text)

            # Remove Numbers
            words = [word for word in words if not re.search(r'\d', word)]

            # Lemmatization
            lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in words]

            # Remove stopwords
            filtered_words = [word for word in lemmatized_words if word.lower() not in set(stopwords.words('english'))]

            # Lowercasing
            filtered_text = ' '.join(filtered_words).lower()

            # Special character removal
            filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', filtered_text)

            # Punctuation removal
            filtered_text = re.sub(r'[^\w\s]', '', filtered_text)
            
            # Remove HTML/CSS tags from text
            soup = BeautifulSoup(filtered_text, 'html.parser')
            filtered_text = soup.get_text(separator=' ', strip=True)
            # Remove HTML entities (e.g., &amp;, &lt;)
            filtered_text = re.sub(r'&\w+;', '', filtered_text)
            # Remove residual tags and styles
            filtered_text = re.sub(r'<.*?>', '', filtered_text)
            filtered_text = re.sub(r'\{.*?\}', '', filtered_text)  # Remove CSS styles within {}

            # Remove extra whitespaces
            filtered_text = ' '.join(filtered_text.split())

            preprocessed_texts.append(filtered_text)
    return preprocessed_texts
