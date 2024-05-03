from collections import Counter, defaultdict, OrderedDict
from queue import Queue, PriorityQueue, LifoQueue, SimpleQueue, SimpleStack
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import string


class Hash:
    def __init__(self, s=None, vec=None, Base=0):
        self.n = len(s) if s is not None else len(vec)
        self.p1, self.p2 = 31, 127
        self.m1, self.m2 = 10**9 + 7, 10**9 + 9
        self.pow1, self.pow2, self.h1, self.h2 = (
            [0] * (self.n + 5),
            [0] * (self.n + 5),
            [0] * (self.n + 5),
            [0] * (self.n + 5),
        )

        self.pow1[0] = self.pow2[0] = 1
        for i in range(1, self.n + 1):
            self.pow1[i] = (self.pow1[i - 1] * self.p1) % self.m1
            self.pow2[i] = (self.pow2[i - 1] * self.p2) % self.m2

        if s is not None:
            self.h1[0] = self.h2[0] = 1
            for i in range(1, self.n + 1):
                self.h1[i] = (
                    self.h1[i - 1] * self.p1 + ord(s[i - (not Base)])
                ) % self.m1
                self.h2[i] = (
                    self.h2[i - 1] * self.p2 + ord(s[i - (not Base)])
                ) % self.m2
        elif vec is not None:
            self.h1[0] = self.h2[0] = 1
            for i in range(1, self.n + 1):
                self.h1[i] = (self.h1[i - 1] * self.p1 + vec[i - (not Base)]) % self.m1
                self.h2[i] = (self.h2[i - 1] * self.p2 + vec[i - (not Base)]) % self.m2

    def sub(self, l, r):
        F = self.h1[r]
        F -= self.h1[l - 1] * self.pow1[r - l + 1]
        F = ((F % self.m1) + self.m1) % self.m1

        S = self.h2[r]
        S -= self.h2[l - 1] * self.pow2[r - l + 1]
        S = ((S % self.m2) + self.m2) % self.m2

        return F, S

    def merge_hash(self, l1, r1, l2, r2):
        a = self.sub(l1, r1)
        b = self.sub(l2, r2)
        F = ((a[0] * self.pow1[r2 - l2 + 1]) + b[0]) % self.m1
        S = ((a[1] * self.pow2[r2 - l2 + 1]) + b[1]) % self.m2
        return F, S

    def at(self, idx):
        return self.sub(idx, idx)

    def equal(self, l1, r1, l2, r2):
        return self.sub(l1, r1) == self.sub(l2, r2)

class KMPSearch:
    
    def prefix_function(a):
        n = len(a)
        prefix = [0] * n
        for i in range(1, n):
            j = prefix[i - 1]
            while j > 0 and a[i] != a[j]:
                j = prefix[j - 1]
            j += (a[i] == a[j])
            prefix[i] = j
        return prefix

    def occurrences_of_each_prefix(prefix, n):
        ret = [0] * (n + 1)
        for i in range(n):
            ret[prefix[i]] += 1
        for i in range(n - 1, 0, -1):
            ret[prefix[i - 1]] += ret[i]
        for i in range(n + 1):
            ret[i] += 1
        return ret


class text_processing(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lower=False,
        upper=False,
        remove_special_characters=False,
        remove_punctuation=False,
        remove_stop_words=False,
        stem_the_words=False,
        remove_digits=False,
        remove_whitespace=False,
        remove_html=False,
        remove_urls=False,
        remove_emails=False,
        remove_emojis=False,
        remove_newlines=False,
        remove_hashtags=False,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.remove_special_characters = remove_special_characters
        self.remove_punctuation = remove_punctuation
        self.remove_stop_words = remove_stop_words
        self.stem_the_words = stem_the_words
        self.remove_digits = remove_digits
        self.remove_whitespace = remove_whitespace
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_emojis = remove_emojis
        self.remove_newlines = remove_newlines
        self.remove_hashtags = remove_hashtags
        super().__init__()

    def fit(self, X, y=None):
        return self

    # for converting the text to lower case
    def __lower_text(self, X):
        X = X.apply(lambda x: x.lower())
        return X

    # for converting the text to upper case
    def __upper_text(self, X):
        X = X.apply(lambda x: x.upper())
        return X

    # for converting the text to words
    def __TextToWord(self, X):
        words = word_tokenize(X)
        return words

    # for removing Special Characters
    """    
        First we need to convert the text to words
        Then we will remove the special characters
        We will use the isalnum() method to check if the word is alphanumeric
        At the end we will join the words to form a text
    """

    def __remove_spec(self, X):
        words = self.__TextToWord(X)
        fixed_words = []
        for w in words:
            if w.isalnum():
                fixed_words.append(w)
        return " ".join(fixed_words)

    def __Remove_Special_Characters(self, X):
        X = X.apply(self.__remove_spec)
        return X

    # for removing Punctuation
    """
        First we need to convert the text to words
        Then we will remove the punctuation
        We will use the string.punctuation to get all the punctuation
        At the end we will join the words to form a text
    """

    def __remove_puncs(self, X):
        words = self.__TextToWord(X)
        punctuation = set(string.punctuation)
        fixed_words = []
        for w in words:
            if not punctuation.__contains__(w):
                fixed_words.append(w)
        return " ".join(fixed_words)

    def __Remove_Punctuation(self, X):
        X = X.apply(self.__remove_puncs)
        return X

    # for removing Stop Words
    """
        First we need to convert the text to words
        Then we will remove the stop words
        We will use the nltk stopwords to get all the stop words
        At the end we will join the words to form a text
    """

    def __remove_stop(self, X):
        stop_words = set(stopwords.words("english"))
        words = self.__TextToWord(X)
        fixed_words = []
        for w in words:
            if not stop_words.__contains__(w):
                fixed_words.append(w)
        return " ".join(fixed_words)

    def __Remove_stop_words(self, X):
        X = X.apply(self.__remove_stop)
        return X

    # for Stemming the words
    """
        First we need to convert the text to words
        Then we will stem the words
        We will use the PorterStemmer to stem the words
        At the end we will join the words to form a text    
    """

    def __stem_text(self, text):
        stemmer = PorterStemmer()
        words = self.__TextToWord(text)
        stemmed_words = []
        for w in words:
            stemmed_words.append(stemmer.stem(w))
        return " ".join(stemmed_words)

    def __Stemming(self, X):
        X = X.apply(self.__stem_text)
        return X

    # for removing the digits from text
    """
        I will do that by using the regular expression
        which will remove the digits from the text
    """

    def __remove_digits(self, text):
        text = re.sub(r"\d+", "", text)
        return text

    def __Remove_digits(self, X):
        X = X.apply(self.__remove_digits)

    # for removing the whitespaces
    """
        I will do that by using the strip() method
        which will remove the whitespaces from the text
    """

    def __strip_whitespace(self, text):
        text = text.strip()
        return text

    def __Remove_whitespace(self, X):
        X = X.apply(self.__strip_whitespace)
        return X

    # for removing the html tags
    """
        I will do that by using the BeautifulSoup library
        which will remove the html tags from the text
    """

    def __html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        return text

    def __Remove_html(self, X):
        X = X.apply(self.__html)
        return X

    # for removing the urls
    """
        I will do that by using the regular expression
        which will remove the urls from the text
    """

    def __remove_url(self, text):
        text = re.sub(r"http\S+", "", text)
        return text

    def __Remove_url(self, X):
        X = X.apply(self.__remove_url)
        return X

    # for removing the emails
    """
        I will do that by using the regular expression
        which will remove the emails from the text
    """

    def __remove_email(self, text):
        text = re.sub(r"\S+@\S+", "", text)
        return text

    def __Remove_emails(self, X):
        X = X.apply(self.__remove_email)
        return X

    # for removing the emojis
    """
        I will do that by using the encode() method
        which will remove the emojis from the text
    """

    def __remove_emoji(self, text):
        text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def __Remove_emojis(self, X):
        X = X.apply(self.__remove_emoji)
        return X

    # for removing the newlines
    """
        I will do that by using the replace() method
    """

    def __remove_newline(self, text):
        text = text.replace("\n", " ")
        return text

    def __Remove_newline(self, X):
        X = X.apply(self.__remove_newline)
        return X

    # for removing the hashtags
    """
        I will do that by using the regular expression
        which will remove the hashtags from the text
    """

    def __Remove_hash(self, text):
        text = re.sub(r"#\S+", "", text)
        return text

    def __Remove_hashtags(self, X):
        X = X.apply(self.__Remove_hash)
        return X

    # we will apply the text processing based on the parameters
    def process_text(self, X):
        data = X.copy()
        # will lowercase the text
        if self.lower:
            data = self.__lower_text(data)
        # will uppercase the text
        if self.upper:
            data = self.__upper_text(data)
        # will remove the special characters
        if self.remove_special_characters:
            data = self.__Remove_Special_Characters(data)
        # will remove the punctuation
        if self.remove_punctuation:
            data = self.__Remove_Punctuation(data)
        # will remove the stop words
        if self.remove_stop_words:
            data = self.__Remove_stop_words(data)
        # will stem the words
        if self.stem_the_words:
            data = self.__Stemming(data)
        # will remove the digits
        if self.remove_digits:
            data = self.__Remove_digits(data)
        # will remove the whitespaces
        if self.remove_whitespace:
            data = self.__Remove_whitespace(data)
        if self.remove_html:
            data = self.__Remove_html(data)
        if self.remove_urls:
            data = self.__Remove_url(data)
        if self.remove_emails:
            data = self.__Remove_emails(data)
        if self.remove_emojis:
            data = self.__Remove_emojis(data)
        if self.remove_newlines:
            data = self.__Remove_newline(data)
        if self.remove_hashtags:
            data = self.__Remove_hashtags(data)
        return data

    def transform(self, X, y=None):
        return self.process_text(X)
