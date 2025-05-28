import spacy
from textblob import TextBlob

nlp = spacy.load("pt_core_news_lg")

def preprocessar_texto(texto):
    doc = nlp(texto)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def analisar_sentimento(texto):
    texto_limpo = preprocessar_texto(texto)
    sentimento = TextBlob(texto_limpo).sentiment
    return sentimento.polarity, sentimento.subjectivity
