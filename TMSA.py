"""
Daniel Palacios Programming class project. Given a research article in pdf format, it extracts the text data
and performs basic Natural Language Processing (NLP) technquies for text cleaning. Then it computes TF-IDF analysis
which calculates the relevance of each word. Also two unsupervised tool for topic modeling:
Latent Dirichlet Allocation (LDA)  and Negative Matrix Factorization (NMF) are used to extract topics from the
pages of the given article.
"""
# Importing dependencies
import PyPDF2
import string as st
import nltk
import os
from nltk import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string
import en_core_web_sm
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
from spacy.matcher import Matcher
from tqdm import tqdm
import networkx as nx

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.io import output_file, show, save
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool)

from bokeh.plotting import from_networkx
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool)


# pip3 install python-docx
nlp = en_core_web_sm.load()

def display_topics(model_name, model, feature_names, no_top_words):
    """
    Function to display topics found by LDA or NMF.
    :param model: Latent Dirichlet Allocation (LDA) or Non-Negative Matrix Factorization (NMF)
    :param feature_names: Defined by CountVectorizer or TFIDFVectorizer, features are usually words or phrases (e.g.
    bigrams)
    :param no_top_words: number of desire topics to display (displays the top scoring topics)
    :return:
    """
    Topics_num = []
    Topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        Topics_num.append(topic_idx)
        Topic_words.append(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

    df_temp = pd.DataFrame()
    df_temp['Topic_number'] = Topics_num
    df_temp['Topic_words'] = Topic_words

    title = 'Results/' + str(model_name) + '_Top_Topics.csv'
    df_temp.to_csv(title, sep = '\t', index = None)

# Start for preprocessing pipeline
def remove_punct(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

def stemming(text):
    ps = SnowballStemmer('english')
    return [ps.stem(word) for word in text]

def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

def return_sentences(tokens):
    return " ".join([word for word in tokens])

def remove_cstopwords(text):
    cwords = ['of', 'and', 'the', 'in', 'to', 'usa', '2022', '2021', '2020',
              'user', 'figur', 'figure', 'table', 'tabl', 'database', 'databas',
              'boxplot', 'figure', 'visual', 'picture', 'image', 'august', 'publish',
              'issue', 'research', 'online', 'paper', 'article', 'sorted', 'size',
              'color', 'code', 'published', 'email', 'also']

    return [word for word in text if word not in cwords]

def main_preprocess(text):
    """
    Preprocessing pipeline, takes a piece of text and performs:
    1.- Remove unecessary punctuation characters
    2.- Tokenize text
    3.- Removes short words
    4.- Remove common english words
    5.- Remove customized stop words
    6.- Stemming
    7.- Lemmatization
    8.- Join back sentences
    :param text: Sentence or paragraphs in a string
    :return: clean sentence or paragraph.
    """
    text = remove_punct(text)
    text = tokenize(text)
    text = remove_small_words(text)
    text = remove_stopwords(text)
    #text = stemming(text) # stemming ommited for results clarity
    text = lemmatize(text)
    text = remove_cstopwords(text)
    text = return_sentences(text)
    return text

def string_processing(string_list):
    """
    Additional string preprocessing.
    :param string_list: text list
    :return: deletes additional punctuation
    """
    ret = []
    for word in string_list:
        ret.append(''.join(x for x in word if x not in string.punctuation))
    return ' '.join(ret)

def Text_extract(pdf_name, type):
    """
    Extracts text from pdf file with pyPDF2 and cleans the data with a combination of text preprocessing
    NLP functions.
    :param pdf_name: name of the file if it is in the same directory, or path to the file.
    :return: dataframe with a column with text per page found in the article.
    """
    # Open given pdf file
    Text = []
    if type == 'pdf':
        pdf_name = str(pdf_name)
        pdfFileObj = open(pdf_name, 'rb')

        # Read pdf
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)


        # Extract text from pdf object
        for i in range(pdfReader.numPages):

            # creating a page object
            pageObj = pdfReader.getPage(i)

            # extracting text from page
            pageObj.extractText()

            text_to_process = pageObj.extractText()

            preprocessed_text = text_to_process

            # Append clean text
            Text.append(preprocessed_text)

    # Close pdf
        pdfFileObj.close()

    # If word document is given instead.
    if type == 'docx':
        document = Document(str(pdf_name))
        doc = document
        for i in doc.paragraphs:
            text_to_process = i.text

            try:
                preprocessed_text = text_to_process

                Text.append(preprocessed_text)
            except:
                pass

    # Create dataframe to store and manage data
    df = pd.DataFrame()

    # Store Text in a dataframe, each row corresponds to each page of the article.
    df['Text'] = Text

    # Call main function for data text cleaning
    df['Clean_Text'] = df['Text'].apply(lambda x : main_preprocess(x))

    str_list = df['Clean_Text'].to_numpy()
    str_list = list(filter(None, str_list))

    dk = pd.DataFrame()

    dk['Clean_Text'] = str_list

    dk= dk.dropna().reset_index(drop=True)

    return dk


def Topic_extractor(df):
    """
    Given a dataframe with a column of clean text, a TF-IDF, NMF and LDA tools are used to extract key phrases and
    topics from the given text.
    :param df: dataframe with column named 'Clean_Text' directly output from Text_Extract() function
    :return: prints top topics found from NMF and LDA, and top 10 TF-IDF words with scores.
    """
    # TF IDF scores
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.95, min_df=2, max_features=300,
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df['Clean_Text'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    #print(tfidf_feature_names[:10])
    df = pd.DataFrame(tfidf[0].T.todense(), index=tfidf_feature_names, columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    df = df.head(10)

    a = df.index
    b = df['TF-IDF']
    d = {}
    for A, B in zip(a,b):
        d[A] = B

    c = list(d.keys())
    v = list(d.values())

    plt.figure(1)

    plt.bar(c,v, color = 'maroon', width = 0.3)
    plt.xlabel('Top relevant words')
    plt.ylabel('TF-IDF Scoring')
    root_path = 'Results'
    # Create directory to store results:
    try:
        os.mkdir(root_path)
    except:
        pass # Directory already exist

    plt.savefig(root_path + '/' + 'TF_IDF_Histogram.png')

    no_topics = 10

    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, l1_ratio=.5).fit(tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tfidf)

    no_top_words = 5

    display_topics('NMF', nmf, tfidf_feature_names, no_top_words)
    display_topics('LDA', lda, tfidf_feature_names, no_top_words)


def get_entities(sent):
    """
    Entity extraction. Directly taken from:
    https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
    """

    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    for tok in nlp(sent):

        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    """
    Extracts grammatical dependency relations using spacy default matcher.
    Directly taken from:
    https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
    """
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    #define the pattern
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

    # Corrected update does not take None anymore <<<<
    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return(span.text)


def KnowledgeGraph(df):
    """
    Creates knowledge Graphs which displays found relationships among entities found from Spacy using networkx packages.
    :param df: dataframe with column named 'Clean_Text' directly output from Text_Extract() function
    :return: image of Knowledge graphs.
    """
    entity_pairs = []

    for i in tqdm(df['Clean_Text']):
        entity_pairs.append(get_entities(i))

    relations = [get_relation(i) for i in tqdm(df['Clean_Text'])]

    # extract subject
    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]

    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

    G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure(2, figsize=(12, 12))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)

    root_path = 'Results'
    plt.savefig(root_path + '/' + 'Static_Knowledge_Graph.png')

def Interactive_KnowledgeGraph(df):
    """
    Creates an interactive knowledge graph in html format.
    Reference: https://docs.bokeh.org/en/latest/docs/user_guide/graph.html
    :param df: dataframe with column named 'Clean_Text' directly output from Text_Extract() function
    :return: Stores html file with graph in Results directory.
    """
    entity_pairs = []

    for i in tqdm(df['Clean_Text']):
        entity_pairs.append(get_entities(i))

    relations = [get_relation(i) for i in tqdm(df['Clean_Text'])]

    # extract subject
    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]

    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

    G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                edge_attr=True, create_using=nx.MultiDiGraph())

    # Show with Bokeh
    plot = Plot(width=800, height=600,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

    plot.title.text = "Interactive Knowledge Graph"

    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))

    # Hover options and selection box colors
    node_hover_tool = HoverTool(tooltips=[("word: ", "@word")])

    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool(), BoxSelectTool(), TapTool())


    color1 = "#455054"
    color2 = "#f2953c"
    color3 = "#308695"
    color4 = "#d45769"

    # Selection mechanisms
    graph_renderer.node_renderer.glyph = Circle(size=10, fill_color=color1)
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=color2)
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=color3)

    graph_renderer.edge_renderer.glyph = MultiLine(line_color=color4, line_alpha=0.8, line_width=3)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=color2, line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=color3, line_width=5)

    # Important to correct this step, if a different inspection policy is chosen, data from the nodes might be
    # ignored and won't display words.

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    # Load word name nodes from G network graph.
    graph_renderer.node_renderer.data_source.data['word'] = list(G.nodes)

    plot.renderers.append(graph_renderer)


    output_file("Results\interactive_graphs.html")
    save(plot)

def Word_cloud(df):
    """
    Generates a word cloud figure, words is plotted as increasing size proportional to its occurancce in the given
    article.

    Reference: https://www.geeksforgeeks.org/generating-word-cloud-python/
    :param df: dataframe with column named 'Clean_Text' directly output from Text_Extract() function
    :return: word cloud image.
    """
    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in df['Clean_Text']:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens) + " "

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(3, figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    root_path = 'Results'
    plt.savefig(root_path + '/' + 'Word_cloud.png')


# Ask input from user: Input is file name. For demonstration we assume file is in same directory as main code.

paper_name = input("Enter name of research article for text mining: ")

# For this example paper_name = "aQTL_atlas.docx"

print("Extracting text from given file ...")

# pdf also works and it was tested.
# df = Text_extract('aQTLpaper.pdf', 'pdf')

paper_type = paper_name.split(".")[-1]

# For this example we use word:
df = Text_extract(str(paper_name), str(paper_type))

print("Computing TF-IDF, NMF, and LDA analysis ...")
# Run TF-IDF,
Topic_extractor(df)

print("Creating Knowledge Graphs ...")
# Run Spacy Knowledge graph from dependency relations
KnowledgeGraph(df)

Interactive_KnowledgeGraph(df)

print("Creating Word Cloud Figure ...")
Word_cloud(df)


# Resources : https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/
# https://pypi.org/project/text-preprocessing/
# https://stackoverflow.com/questions/66348359/adding-a-space-between-string-words
# https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/
# https://www.kdnuggets.com/2018/11/text-preprocessing-python.html
# https://www.analyticsvidhya.com/blog/2021/06/part-3-topic-modeling-and-latent-dirichlet-allocation-lda-using-gensim-and-sklearn/
# Convert to word https://stackoverflow.com/questions/26358281/convert-pdf-to-doc-python-bash
# Using Gensim https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
# https://www.kaggle.com/code/balatmak/text-preprocessing-steps-and-universal-pipeline/notebook
# >> https://github.com/Shubha23/Text-processing-NLP/blob/master/NLP%20-%20Text%20processing%20pipeline.ipynb
# https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
# Graph knowledge networks https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
# https://github.com/nmolivo/tesu_scraper/blob/master/Python_Blogs/01_extract_from_MSWord.ipynb text from word
# https://blog.aspose.com/es/2021/11/25/extract-text-from-word-docx-in-python/
# https://www.geeksforgeeks.org/generating-word-cloud-python/
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html