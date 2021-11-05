import spacy, nltk, os, en_core_web_md
nlp = en_core_web_md.load()

from collections import defaultdict
from math import e


def get_text(name: str, folder: str = 'data'):
    with open(os.path.join(folder, f'{name}.txt'), 'r') as fread:
        text = fread.read()
    
    return text


def preprocess(text: str, nlp=nlp):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def postprocess_lemma(keywords_scores: dict, nlp=nlp):
    lemma_dict = defaultdict(list)
    for word, score in keywords_scores.items():
        lemma = nlp(word)[0].lemma_
        lemma_dict[lemma].append(score)

    return {lemma : sum(scores) for lemma, scores in lemma_dict.items()}


def postprocess_stemm(keywords_with_scores: dict):
    keywords = list(keywords_with_scores.keys())
    stemmer = nltk.stem.PorterStemmer()
    stemmed_keywords = [stemmer.stem(key) for key in keywords]

    stem_dict = defaultdict(list)
    for i, stem in enumerate(stemmed_keywords):
        stem_dict[stem].append(keywords[i])
    
    result_keywords_with_scores = dict()

    for stem, keywords in stem_dict.items():
        score = sum(map(lambda k: keywords_with_scores[k], keywords))
        keyword = sorted(keywords)[0] if len(keywords) > 1 else keywords[0]
        result_keywords_with_scores[keyword] = score
    
    return result_keywords_with_scores


def postprocess_norm(keywords_scores: dict):
    all_scores = keywords_scores.values()

    return {key : softmax(score, all_scores) for key, score in keywords_scores.items()}


def postprocess(keywords_scores: dict, top_words: int = 5):
    lemmatization = postprocess_lemma(keywords_scores)
    stemming = postprocess_stemm(lemmatization)

    top_words = len(stemming) if len(stemming) < top_words else top_words
    keywords = dict(sorted(stemming.items(), key=lambda item: item[1], reverse=True)[:5])

    return postprocess_norm(keywords)


def softmax(score, scores, order='max'):
    score = score if order == 'max' else (1 - score)
    return e ** score / sum([e ** s for s in scores])


if __name__ == '__main__':
    pass
