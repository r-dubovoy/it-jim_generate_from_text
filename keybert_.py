from keybert import KeyBERT

def keybert(text: str, words: int = 5):
    keywords = KeyBERT().extract_keywords(text, top_n=words)
    return dict(keywords)

if __name__ == '__main__':
    pass