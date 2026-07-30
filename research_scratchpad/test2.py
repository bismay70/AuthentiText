import spacy
import textstat
from textblob import TextBlob
import sklearn

print("🔄 Testing Week 2 Environments...")
try:
    # Test spaCy loading
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("This is a quick test for NIT Rourkela AI/ML pipeline.")
    print("✅ spaCy & Language Model: LOADED")
    print(f"   Sample Tokens: {[token.text for token in doc[:4]]}")
    
    # Test textstat
    readability = textstat.flesch_reading_ease("This is a simple sentence layout.")
    print(f"✅ Textstat Readability  : WORKING (Score: {readability})")
    
    # Test TextBlob
    sentiment = TextBlob("This project is going incredibly well!").sentiment.polarity
    print(f"✅ TextBlob Sentiment    : WORKING (Polarity: {sentiment})")
    
    print("\n🎉 All Week 2 libraries successfully configured and ready!")
except Exception as e:
    print(f"❌ Verification Failed: {e}")