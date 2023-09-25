from typing import List, Any, Union, Optional
from decimal import Decimal
from string import punctuation
from enum import Enum, auto

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from nltk import FreqDist, tokenize, corpus, RSLPStemmer
import seaborn as sns
import unidecode

from util import get_logger_factory

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
logger = get_logger_factory()

class VectorizerOption(Enum):
    TFIDF_BRUTE_VECTORIZER = auto()
    TFIDF_VECTORIZER = auto()
    COUNT_VECTORIZER = auto()


def classify_text(text: pd.DataFrame, column_text: str = "removed_stopwords", column_classification: str = "classification", max_features: Optional[int] = 50, random_state: int = 42, vectorizer_option: VectorizerOption = VectorizerOption.COUNT_VECTORIZER) -> Decimal:
    """Classify text to determine sentiment.
    Vocabulary:
    * Train - teach ML model.
    * Test - use model to our purpose."""
    # [CountVectorizer] For text_pt: without set max_features the accuracy is 0.8814395471087748. max_features=50 was recomended by tutor. Accuracy is 0.6583097452486858 with max_features=50.

    vectorizer: Any[CountVectorizer, TfidfVectorizer] = None

    if vectorizer_option == VectorizerOption.COUNT_VECTORIZER:
        vectorizer = CountVectorizer(lowercase=False, ngram_range = (1,2), max_features=max_features) if max_features is not None else CountVectorizer(lowercase=False, ngram_range = (1,2))
    elif vectorizer_option == VectorizerOption.TFIDF_VECTORIZER:
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range = (1,2), max_features=max_features) if max_features is not None else TfidfVectorizer(lowercase=False, ngram_range = (1,2))
    else:
        raise Exception(f"Unused vectorizer_option: {vectorizer_option}")

    bag_of_words = vectorizer.fit_transform(text[column_text])
    train, train_test, train_class, test_class = train_test_split(bag_of_words, text[column_classification], random_state=random_state)
    logistic_regression: LogisticRegression = LogisticRegression() # default is: solver='lbfgs'
    logistic_regression.fit(train, train_class)
    accuracy: Decimal = logistic_regression.score(train_test, test_class)

    weights: pd.DataFrame = pd.DataFrame(logistic_regression.coef_[0].T, index=vectorizer.get_feature_names_out()) # get_feature_names doesn't exist anymore
    
    logger.debug(f"nlargest (10): {weights.nlargest(10, 0)}")
    logger.debug(f"nsmallest (10): {weights.nsmallest(10, 0)}")
    logger.debug(f"Accuracy found was: {accuracy} vectorizer_option: {vectorizer_option} max_features: {max_features}")

    return accuracy


def text_cloud_image(reviews: pd.DataFrame, column: str = 'text_pt', width: int = 1900, height: int = 1500, max_font_size: int = 110, collocations: bool = False) -> None:
    """Show a cloud of text in a image to get insights."""

    all_words: str = ' '.join([text for text in reviews[column]])
    word_cloud: WordCloud = WordCloud(width=width, height=height, max_font_size=max_font_size, collocations=collocations).generate(all_words)
    plt.figure(figsize=(20,15))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def pareto(phrase: str, figsize: tuple = (12, 8), color_frequence: str = 'grey', color_percentage: str = 'red', quantity: int = 10, wt: Union[tokenize.WhitespaceTokenizer, tokenize.WordPunctTokenizer] = tokenize.WhitespaceTokenizer()) -> None:
    """Show the N more used words and plot it with pareto
    distribution line."""

    freq_dist = FreqDist(wt.tokenize(phrase))
    df = pd.DataFrame({"Word": list(freq_dist.keys()), "Frequence": list(freq_dist.values())})

    df = df.nlargest(columns="Frequence", n=quantity) # focus on the most meaningful data
    total = df['Frequence'].sum()
    df['Percentage'] = df['Frequence'].cumsum() / total * 100 # cumulative sum to see evolution.

    logger.info("df>:")
    print(df)
    print("-----------------------")
    print("-----------------------")
    print("")
    logger.info(f"{total = }")
    print("-----------------------")
    print("-----------------------")
    print("")

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x="Word", y="Frequence", color=color_frequence)
    ax.set(ylabel="Counting")
    ax2 = ax.twinx()
    sns.lineplot(data=df, x="Word", y="Percentage", color=color_percentage, sort=False, ax=ax2)
    logger.info("Showing pareto data!")
    plt.show()


def remove_stopwords(opnions: pd.core.series.Series, meaningless_words: Union[List[str], None] = corpus.stopwords.words("portuguese"), wt: Union[tokenize.WhitespaceTokenizer, tokenize.WordPunctTokenizer] = tokenize.WhitespaceTokenizer(), lower: bool = True, stemmer: RSLPStemmer = RSLPStemmer()) -> List[str]:
    """Pre-process an opnion to be used in classification. By default
    use basic stopwords from nltk corpus and white space tokenize. Also,
    remove punctuation and accents and use all words in lower case."""

    # FIXME lower, remove accent (and all pre-process) is applied in opinion too or only in meaningless words?! Apply lower to corpus in meaningless_words
    meaningless_words = [punct for punct in punctuation] + list(set(corpus.stopwords.words("portuguese") + [unidecode.unidecode(stopwords) for stopwords in corpus.stopwords.words("portuguese")])) if meaningless_words is None else meaningless_words

    return [' '.join([word if stemmer is None else stemmer.stem(word) for word in wt.tokenize(unidecode.unidecode(opnion.lower()) if lower else unidecode.unidecode(opnion)) if word not in meaningless_words]) for opnion in opnions]
