from scipy.sparse import csr_matrix, hstack, vstack, coo_matrix
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

__author__ = 'jamesgin'


from features import *
from model import *

def nn_test(n_neighbours, tfidf, X, questions):
    nn = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute', metric='cosine')
    nn.fit(X)
    correct = 0
    total = float(questions.count())
    for q in questions:
        searches = [q.body + '. ' + a for a in q.all_answers()]
        vecs = tfidf.transform(searches)
        neigh = nn.kneighbors(vecs, return_distance=True)
        dist = neigh[0].mean(axis=1)
        chosen = dist.argmin()
        if chosen == q.correct:
            correct += 1
    print(correct, total, correct/total)

def nn_investigate(n_neighbours, tfidf, X, y, questions):
    nn = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute', metric='cosine')
    nn.fit(X)
    correct = 0
    total = float(questions.count())
    for q in questions:
        searches = [q.body + '. ' + a for a in q.all_answers()]
        vecs = tfidf.transform(searches)
        neigh = nn.kneighbors(vecs, return_distance=True)
        dist = neigh[0].mean(axis=1)
        chosen = dist.argmin()
        # clause_ids = y[neigh[1]].ravel()
        # nearest_correct = session.query(RawClause.cleaned).filter(RawClause.id==clause_ids[q.correct]).first()
        # nearest_chosen = session.query(RawClause.cleaned).filter(RawClause.id==clause_ids[chosen]).first()
        #
        # print q.body
        # print q.all_answers()[q.correct]
        # print nearest_correct
        # print q.all_answers()[chosen]
        # print nearest_chosen

        if chosen == q.correct:
            correct += 1
    print(correct, total, correct/total)

def nn_test_w2v(n_neighbours, w2v, docs, questions):
    nn = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute', metric='cosine')

    nn.fit(docs)
    correct = 0
    total = float(questions.count())
    for q in questions:
        searches = [q.body + '. ' + a for a in q.all_answers()]
        vecs = [get_doc_vec(w2v, s) for s in searches]
        neigh = nn.kneighbors(vecs, return_distance=True)
        dist = neigh[0].mean(axis=1)
        chosen = dist.argmin()
        if chosen == q.correct:
            correct += 1
    print(correct, total, correct/total)

def w2v_test(n_neighbours, w2v, X, questions):
    nn = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute', metric='cosine')
    nn.fit(X)
    correct = 0
    total = float(questions.count())
    for q in questions:
        searches = [q.body + '. ' + a for a in q.all_answers()]
        vecs = [get_doc_vec(w2v, s) for s in searches]
        neigh = nn.kneighbors(vecs, return_distance=True)
        dist = neigh[0].mean(axis=1)
        chosen = dist.argmin()
        if chosen == q.correct:
            correct += 1
    print(correct, total, correct/total)

def w2v_discrim_test(w2v, questions):
    correct = 0
    total = float(questions.count())
    X = []
    y = []

    for q in questions:
        q_vec = get_doc_vec(w2v, q.body)
        a_vecs = [get_doc_vec(w2v, a) for a in q.all_answers()]
        vecs = [np.hstack((q_vec, a)) for a in a_vecs]
        vecs = [v / np.linalg.norm(v) for v in vecs]
        X.extend(vecs)
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)

    X = np.array(X)
    y = np.concatenate(y)
    segs = cross_val_score(LogisticRegressionCV(scoring='roc_auc'), X, y, scoring='roc_auc', cv=8)
    print(segs)
    print(np.mean(segs))
    # print(correct, total, correct/total)

def tfidf_discrim_test(tfidf, knowledge, questions):
    correct = 0
    total = float(questions.count())
    X = []
    y = []

    for q in questions:
        # q_vec = tfidf.transform([q.body])
        # a_vecs = [tfidf.transform([a]) for a in q.all_answers()]
        # vecs = [np.hstack((q_vec.toarray().ravel(), a.toarray().ravel())) for a in a_vecs]
        docs = [q.body + '. ' + a for a in q.all_answers()]
        vecs = tfidf.transform(docs)
        # vecs = [v / np.linalg.norm(v) for v in vecs]
        X.extend(vecs)
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)


    X = vstack(X)
    true_X = knowledge
    y = np.concatenate(y)
    all_y = np.ones(knowledge.shape[0])
    segs = []

    pred = cross_val_predict(LogisticRegressionCV(class_weight='balanced'), X, y, cv=8)

    for train, test in KFold(X.shape[0], n_folds=8, shuffle=True):
        qpart = X[train, :]
        train_X = qpart
        train_y = np.concatenate([y[train]], axis=0)
        l = LogisticRegressionCV(class_weight='balanced')
        l.fit(train_X, train_y)
        segs.append(l.score(X[test, :], y[test]))
    print(np.mean(segs))
    segs = cross_val_score(LogisticRegressionCV(class_weight='balanced'), X, y, cv=8)
    print(segs)
    print(np.mean(segs))


def w2v_sim_test(w2v):
    correct = 0
    err = 0
    total = float(questions.count())
    for q in questions:
        try:
            searches = [a for a in q.all_answers()]
            vecs = [get_doc_vec(w2v, s) for s in searches]
            q_vec = get_doc_vec(w2v, q.body)
            chosen = np.array(vecs).dot(q_vec).argmax()
            if chosen == q.correct:
                correct += 1
        except:
            err += 1

    print(correct, total, err, correct/total)

def get_search_set(questions, tfidf, w2v, k):
    X = []
    y = []

    for q in questions:
        dense_vec = get_doc_vec(w2v, q.body)
        q_vec_tfidf = tfidf.transform([q.body])
        q_vec = np.hstack((q_vec_tfidf.toarray().ravel() * k, dense_vec * (1-k)))
        a_vecs = [np.hstack((tfidf.transform([a]).toarray().ravel() * k, get_doc_vec(w2v, a) * (1-k))) for a in q.all_answers()]

        vecs = [np.hstack((q_vec, a)) for a in a_vecs]
        for v in vecs:
            norm = np.linalg.norm(v)
            v /= norm


        X.extend(vecs)
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)
    X = csr_matrix(X)
    y = np.concatenate(y)
    return X, y

def cross_val_questions(questions, clf, tfidf, w2v, ks, cv=3):
    kf = KFold(questions.count(), cv, shuffle=True)
    vals = []
    questions = questions.all()
    tfidf_vectors = get_tfidf_vectors(questions, tfidf)
    w2v_vectors = get_w2v_vectors(questions, w2v)
    y = get_y_vector(questions)
    for k in ks:
        X = hstack((tfidf_vectors * k, w2v_vectors * (1-k))).tocsr()

        for train, test in kf:
            train_all = get_fourfold_index(train)
            test_all = get_fourfold_index(test)
            train_X = X[train_all, :]
            test_X = X[test_all, :]
            train_y = y[train_all]

            test_qs = [questions[i] for i in test]

            clf.fit(train_X, train_y)
            y_prob = clf.predict_proba(test_X)
            y_pred = y_prob[:,1].reshape(len(test), 4).argmax(axis=1)
            y_true = np.array([q.correct for q in test_qs])
            val = (y_true == y_pred).mean()
            vals.append(val)
            # print(val)

        print(k, np.mean(vals))

def get_fourfold_index(index):
    i4 = index*4
    return np.concatenate([i4, i4+1, i4+2, i4+3])

def get_w2v_vectors(questions, w2v):
    X = []
    for q in questions:
        q_vec = get_doc_vec(w2v, q.body)
        a_vecs = [get_doc_vec(w2v, a) for a in q.all_answers()]

        vecs = [np.hstack((q_vec, a)) for a in a_vecs]
        X.extend(vecs)
    X = coo_matrix(X)
    X /= np.sqrt(2)
    return X

def get_tfidf_vectors(questions, tfidf):
    X = []
    for q in questions:
        q_vec = tfidf.transform([q.body])
        a_vecs = tfidf.transform(q.all_answers())

        vecs = [hstack((q_vec, a)) for a in a_vecs]
        X.extend(vecs)

    X = vstack(X)
    X /= np.sqrt(2)
    return X

def get_y_vector(questions):
    y = []
    for q in questions:
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)
    y = np.concatenate(y)
    return y


@memory.cache
def get_w2v(size, a, iters):
    sents = generate_sentences()
    w2v = Word2Vec(sents, size, alpha=a, sg=0, iter=iters, workers=8, sample=0)
    return w2v

if __name__ == '__main__':
    questions = session.query(Question).filter((Question.type == None))
                                               # (Question.related_clause.isnot(None))&
                                               # (Question.related_clause != 54488))
        # .order_by(Question.id.desc())
    # X, y, tfidf = generate_section_set(1)
    # X, y, tfidf = generate_clause_set(get_clause_id, 1)
    X, docs, tfidf = generate_statement_set()
    for i in range(1,30,2):
        nn_investigate(i, tfidf, X, docs, questions)
    # nn_test(1, tfidf, X, questions)
    # tfidf_discrim_test(tfidf, X, questions)
    # for i in range(1,7):
    #     nn_test(i, tfidf, X, questions)
    # sents = generate_sentences()
    # for i in [100]:
    #     for a in [0.025]:
    #         for iters in [1]:
    #             print(i, a)
    #             w2v = get_w2v(i, a, 80)
    #             docs = generate_w2v_corpus(w2v)
    #             w2v_sim_test(w2v)
                #
                # print(w2v.most_similar('client'))

                # cross_val_questions(questions, ExtraTreesClassifier(100), tfidf, w2v, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 1], 10)
                # cross_val_questions(questions, ExtraTreesClassifier(2000), 8)
                # w2v_discrim_test(w2v, questions)
                # for j in [1,3,5,7,9]:
                #     nn_test_w2v(j,w2v,docs,questions)
                # vecs = generate_w2v_clause_vecs(w2v)
                # for j in range(1, 10):
                #     w2v_test(j, w2v, vecs, questions)
    # # # pass
    # # # for i in [0.025, 0.05]:
    # # #     X, y, tfidf = generate_section_set(i)
    # # #     print(i)
    # # #     for j in range(1,10):
    # # #         nn_test(j, tfidf, X, questions)