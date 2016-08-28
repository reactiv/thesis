from gensim.models import Word2Vec
import nltk
from scipy.stats import mode
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from cleaning import clean_sentence

__author__ = 'jamesgin'
import dbconn
from nltk.corpus import stopwords
from model import *
from sklearn.feature_extraction.text import TfidfVectorizer
from tempfile import mkdtemp
from joblib import Memory
import numpy as np
session = dbconn.session
eng_stop = stopwords.words('english')
cachedir = mkdtemp()
memory = Memory('temp', verbose=1)

tokenise = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b').build_analyzer()

@memory.cache
def generate_section_set(max_df):
    print('Generating Section Set')
    all_sections = session.query(Section).filter(Section.source_id.isnot(None))
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop)
    docs = []
    y = []
    for s in all_sections:
        doc = s.name
        for c in s.clauses:
            if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
                doc += c.header + '. ' + c.cleaned + '. '
        docs.append(doc)
        y.append(s.id)
    pass

    tfidf = cv.fit_transform(docs)
    print(tfidf)
    ys = np.array(y)
    randidx = np.random.permutation(len(ys))
    tfidf = tfidf[randidx,:]
    ys = ys[randidx]
    print('{} Generated'.format(len(ys)))
    return tfidf, ys, cv

# @memory.cache
def generate_clause_set(yfunc, max_df):
    print('Generating Clause Set')
    all_sections = session.query(Section).filter(Section.source_id.isnot(None))
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop)
    docs = []
    y = []
    for s in all_sections:
        for c in s.clauses:
            if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
                check = yfunc(s, c)
                docs.append(c.header + '. ' + c.cleaned)
                y.append(check)

    tfidf = cv.fit_transform(docs)
    ys = np.array(y)
    randidx = np.random.permutation(len(ys))
    tfidf = tfidf[randidx,:]
    ys = ys[randidx]
    print('Generated')
    return tfidf, ys, cv

def generate_statement_set():
    print('Generating Clause Set')
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop)
    docs = []
    statements = session.query(Statement)
    for s in statements:
        if s.text() is not None:
            docs.append(s.text())
    print('{} Docs'.format(len(docs)))
    tfidf = cv.fit_transform(docs)

    print('Generated')
    return tfidf, docs, cv

@memory.cache
def get_all_statements():
    docs = []
    parts = session.query(StatementPart, RawClause, Section).join(RawClause).join(Section).filter((StatementPart.parent_id.is_(None))).order_by(StatementPart.id)
    tot = float(parts.count())
    i = 0
    for statement, clause, section in parts:
        for shortest in statement.all_sentences():
            sent = fix_multipart_sentence(shortest)
            sent = sent.replace(u'\u2019', "'")
            docs.append(sent)
            i+=1
            if i%1000 == 0:
                print(i/tot)
    return docs

@memory.cache
def get_dataset(level, extractor):
    def null_to_str(header, cleaned):
        str = ''
        if header:
            str += header + '. '
        if cleaned:
            str += cleaned
        return str

    docs = []
    if level == 'clause':
        # docs = list(session.query(RawClause.header, RawClause.cleaned, Section).join(Section).filter(RawClause.content_html.notilike('%<table%')
        #                                                                                              & RawClause.content_html.notilike('deleted')
        #                                                                                              & Section.source_id.isnot(None)))
        # # docs = list(session.query(RawClause.cleaned).distinct()) + list(session.query(RawClause.header).distinct())
        # docs = [null_to_str(d[0], d[1]) for d in docs if d]
        all_sections = session.query(Section).filter(Section.source_id.isnot(None))
        docs = []

        for s in all_sections:
            for c in s.clauses:
                if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
                    docs.append(c.header)
                    docs.append(c.cleaned)
        # return docs

    else:
        docs = get_all_statements()

    docs = [clean_sentence(s, extractor) for s in docs]
    return docs


@memory.cache
def generate_statement_from_parts_set():
    print('Generating Statement Set')
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop)
    docs = []
    statements = session.query(StatementPart, RawClause, Section).join(RawClause).join(Section).filter((StatementPart.parent_id.is_(None))).order_by(StatementPart.id)
    tot = float(statements.count())
    i = 0
    for statement, clause, section in statements:
        for shortest in statement.all_sentences():
            sent = fix_multipart_sentence(shortest)
            sent = sent.replace(u'\u2019', "'")
            docs.append(sent)
        i += 1
        if i % 100 == 0:
            print i / tot
    print('{} Docs'.format(len(docs)))
    tfidf = cv.fit_transform(docs)

    print('Generated')
    return tfidf, docs, cv

@memory.cache
def generate_w2v_corpus(w2v):
    print('Generating Clause Set')
    docs = []
    # all_sections = session.query(Section).filter(Section.source_id.isnot(None))
    # for s in all_sections:
    #     for c in s.clauses:
    #         if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
    #             docs.append(c.header + '. ' + c.cleaned)
    statements = session.query(Statement)
    for s in statements:
        if s.text() is not None:
            docs.append(s.text())


    d2v = [get_doc_vec(w2v, doc) for doc in docs]
    d2v = np.array(d2v)
    return d2v

@memory.cache
def generate_sentence_set(max_df):
    print('Generating Clause Set')
    all_sections = session.query(Section).filter(Section.source_id.isnot(None))
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop, max_df=max_df)
    docs = []

    for s in all_sections:
        for c in s.clauses:
            if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
                docs.append(c.header)

                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', c.cleaned)
                docs.extend(sentences)

    tfidf = cv.fit_transform(docs)
    print('{} Generated'.format(len(docs)))
    return tfidf, cv

@memory.cache
def generate_sentences():
    print('Generating Clause Set')
    tf = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', max_df=1)
    analyser = tf.build_analyzer()
    all_sections = session.query(Section).filter(Section.source_id.isnot(None))
    docs = []

    for s in all_sections:
        for c in s.clauses:
            if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
                docs.append(analyser(c.header))

                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', c.cleaned)

                docs.extend([analyser(sent) for sent in sentences])

    return docs

@memory.cache
def generate_w2v_clause_vecs(w2v):
    print('Generating Clause Set')
    all_sections = session.query(Section).filter(Section.source_id.isnot(None))
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop)
    vecs = []
    for s in all_sections:
        for c in s.clauses:
            if c.cleaned is not None and 'deleted' not in c.cleaned.lower():
                text = c.header + '. ' + c.cleaned
                vec = get_doc_vec(w2v, text)
                if vec is not None:
                    vecs.append(vec)
    return vecs

def get_doc_vec(w2v, sentence):
    tokens = tokenise(sentence)
    vecs = np.array([w2v[t] for t in tokens if t in w2v])
    if len(vecs) > 0:
        vec = vecs.sum(axis=0)
        vec /= np.linalg.norm(vec)
        return vec
    else:
        return np.zeros(w2v.vector_size)

def get_clause_id(section, clause):
    return clause.id

def get_section_id(section, clause):
    return section.id

def get_section_name(section, clause):
    if '/' in section.docpath:
        return section.docpath[:section.docpath.find('/')]
    else:
        return section.docpath


def generate_qa_set(tfidf, yfunc):
    questions = session.query(Question, RawClause, Section)\
        .join(RawClause).filter(RawClause != None).filter(RawClause.id != 54488).join(Section)
    docs = []
    ys = []
    for q, c, s in questions:
        docs.append(q.text())
        ys.append(yfunc(s, c))

    X = tfidf.transform(docs)
    y = np.array(ys)
    return X, y

def clip_end(part):
    for i in [';', ' ;', '; and', '; or']:
        if part.endswith(i):
            part = part[:-len(i)]

    return part

def fix_multipart_sentence(sentence):
    sentence = [clip_end(p).strip() for p in sentence]
    sent_string = ' '.join(sentence)
    if not sent_string.endswith('.'):
        sent_string += '.'
    return sent_string

@memory.cache
def get_tfidf(level, extractor):
    docs = get_dataset(level, extractor)
    cv = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', stop_words=eng_stop)
    vecs = cv.fit_transform(docs)
    return cv, vecs

@memory.cache
def get_w2v(level, extractor, **kwargs):
    docs = get_dataset(level, extractor)
    sents = []
    for d in docs:
        sents.extend(re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', d))
    tok_sents = [tokenise(s) for s in sents]
    w2v = Word2Vec(tok_sents, **kwargs)
    return w2v, [get_doc_vec(w2v, s) for s in sents]


if __name__ == '__main__':
    graphs = []
    for maxdf in [1]:
        for y_func in [get_clause_id]:
            X, y, tfidf = generate_clause_set(y_func, maxdf)
            X_test, y_test = generate_qa_set(tfidf, y_func)
            modal = []
            contained = []
            for i in range(1,2):
                et = KNeighborsClassifier(i, metric='cosine', algorithm='brute')
                neigh = NearestNeighbors(i, metric='cosine', algorithm='brute', n_jobs=4)
                neigh.fit(X)
                blp = neigh.kneighbors(X_test, return_distance=False)
                all_y_pred = y[blp]
                y_mat = np.concatenate([y_test.reshape(-1,1)]*i,axis=1)

                et.fit(X, y)
                y_pred = et.predict(X_test)
                print(y_pred)
                print(y_test)
                print((y_pred == y_test).mean())
                modal.append((y_pred == y_test).mean())
                contained.append((y_mat==all_y_pred).any(axis=1).mean())
            print(np.max(modal))

            graphs.append((modal, contained))
    # f, axarr = plt.subplots(3)
    # axarr[0].plot(graphs[0][0])
    # axarr[0].plot(graphs[0][1])
    # axarr[0].set_title('Correct Sourcebook')
    # axarr[1].plot(graphs[1][0])
    # axarr[1].plot(graphs[1][1])
    # axarr[1].set_title('Correct Section')
    # axarr[2].plot(graphs[2][0])
    # axarr[2].plot(graphs[2][1])
    # axarr[2].set_title('Correct Clause')
    # f.show()

    # pred = cross_val_predict(ExtraTreesClassifier(n_estimators=2000, n_jobs=-1), X, y)
    # print(confusion_matrix(y, pred))
    # print(y)
    # print(pred)
    # print((y == pred).mean())