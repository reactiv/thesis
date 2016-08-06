from nltk import AlignedSent, IBMModel2
from sklearn.feature_extraction.text import CountVectorizer

__author__ = 'jamesgin'

from features import *
from nltk.translate.ibm3 import IBMModel3

def semi_sup():
    pairs = session.query(Question, RawClause).filter((Question.related_clause.isnot(None)) &
                                                      (Question.related_clause != 54488))\
        .join(RawClause)
    for q, r in pairs:
        print('')
        print(q.body)
        print(q.get_correct())
        print(r.cleaned)

@memory.cache
def get_ibm_model3():
    X, y, tfidf = generate_clause_set(get_clause_id, 1)
    nn = NearestNeighbors(1, algorithm='brute', metric='cosine')
    nn.fit(X)
    question_bodies = session.query(Question.body).all()
    question_bodies = [q[0] for q in question_bodies]
    q_vec = tfidf.transform(question_bodies)
    neighs = nn.kneighbors(q_vec, return_distance=False)
    ids = y[neighs].ravel()

    bodies = []
    for i in ids:
        body = session.query(RawClause.cleaned).filter(RawClause.id == i).first()
        bodies.append(tokenise(body[0]))

    question_toks = [tokenise(q) for q in question_bodies]

    matched = zip(question_toks, bodies)
    aligned = [AlignedSent(*m) for m in matched]
    ibm = IBMModel2(aligned, 1)
    return ibm

def question_cluster():
    questions = session.query(Question.body).all()
    cv = CountVectorizer()
    freq_mat = cv.fit_transform([q[0] for q in questions])
    freqs = [(word, freq_mat.getcol(idx).sum()) for word, idx in cv.vocabulary_.items()]
    print sorted(freqs, key = lambda x: -x[1])
    fq = freq_mat.toarray()
    fq = fq / np.linalg.norm(fq, axis=1)[:,None]
    pass

if __name__ == '__main__':
    question_cluster()
    # ibm = get_ibm_model3()
    # semi_sup()