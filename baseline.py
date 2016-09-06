from functools import partial
from itertools import chain, product
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from cleaning import clean_sentence, extract_entities, edit_sq_group, reference, entity_and_reference, get_entities
from inference import get_w2v as get_w2v2
from prepare_nn import encode_questions, encode_generated_questions, encode_questions_with_gen
import pandas as pd
import pickle

__author__ = 'jamesgin'
from features import *
from model import *
import scipy.sparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('log/baseline.log'))

def ir_method(transformer, docs, max_neighbours, questions):
    correct = [0] * max_neighbours
    nn = NearestNeighbors(max_neighbours, algorithm='brute', metric='cosine')
    nn.fit(docs)
    for q in questions:
        query_vectors = transformer(q)
        neigh = nn.kneighbors(query_vectors)
        for i in range(1, max_neighbours+1):
            dist = neigh[0][:, :i+1].mean(axis=1)
            if q.type == 0:
                chosen = dist.argmin()
            else:
                chosen = dist.argmin()
            if chosen == q.correct:
                correct[i-1] += 1

    return np.array(correct) / float(len(questions))

def similarity_method(transformer, questions):
    correct = 0
    for q in questions:
        query_vectors = scipy.sparse.csr_matrix(transformer(q))
        q_part = query_vectors[:,:query_vectors.shape[1]/2]
        a_part = query_vectors[:,query_vectors.shape[1]/2:]
        if q.type == 0:
            chosen = np.einsum('ij,ij->i', q_part.toarray(), a_part.toarray()).argmax()
        else:
            chosen = np.einsum('ij,ij->i', q_part.toarray(), a_part.toarray()).argmax()
        if chosen == q.correct:
            correct += 1


    return correct / float(len(questions))

def get_discrim_train_set(train_qs, transformer):
    X = []
    y = []
    for q in train_qs:
        vecs = transformer(q)
        X.append(vecs)
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)
    try:
        X = scipy.sparse.vstack(X)
    except:
        X = np.vstack(X)

    return X, np.concatenate(y)

def get_discrim_test_set(test_qs, transformer):
    test_X = []
    test_y = []
    for q in test_qs:
        vecs = transformer(q)
        test_X.append(vecs)
        test_y.append(q.correct)
    try:
        test_X = scipy.sparse.vstack(test_X)
    except:
        test_X = np.vstack(test_X)

    return test_X, np.array(test_y)

def discrim_method(transformer, questions, test_questions, classifier, folds):

    total = float(len(questions))

    q_list = questions

    scores = []
    for train, test in KFold(int(total), folds, shuffle=True):
        train_qs = [q_list[i] for i in train]
        test_qs = [q_list[i] for i in test]
        X, y = get_discrim_train_set(train_qs, transformer)
        test_X, test_y = get_discrim_test_set(test_qs, transformer)

        classifier.fit(X, y)
        y_prob = classifier.predict_proba(test_X)
        y_pred = y_prob[:,1].reshape(len(test), 4).argmax(axis=1)
        scores.append((test_y == y_pred).mean())

    cv_score = np.mean(scores)
    X, y = get_discrim_train_set(questions, transformer)
    test_X, test_y = get_discrim_test_set(test_questions, transformer)
    classifier.fit(X, y)
    y_prob = classifier.predict_proba(test_X)
    y_pred = y_prob[:,1].reshape(len(test_y), 4).argmax(axis=1)
    test_score = (test_y == y_pred).mean()

    return cv_score, test_score

def tfidf_transformer(tfidf, concatenate, question):
    if concatenate:
        q_vecs = tfidf.transform([question.body]*4)
        a_vecs = tfidf.transform(question.all_answers())
        return scipy.sparse.hstack([q_vecs, a_vecs])
    else:
        q_a_vecs = tfidf.transform([question.body + '. ' + a for a in question.all_answers()])
        return q_a_vecs

def word2vec_transformer(w2v, concatenate, question):
    if concatenate:
        q_vecs = np.vstack([get_doc_vec(w2v, question.body)] * 4)
        a_vecs = np.vstack([get_doc_vec(w2v, a) for a in question.all_answers()])
        return np.hstack([q_vecs, a_vecs])
    else:
        q_a_vecs = [get_doc_vec(w2v, question.body + '. ' + a) for a in question.all_answers()]
        return np.vstack(q_a_vecs)

def blended_transformer(w2v, tfidf, concatenate, alpha, question):
    w2v_vecs = word2vec_transformer(w2v, concatenate, question)
    tfidf_vecs = tfidf_transformer(tfidf, concatenate, question)
    w2v_vecs *= alpha
    tfidf_vecs *= (1-alpha)
    return scipy.sparse.hstack([w2v_vecs, tfidf_vecs])

def w2v_similarity_test(questions, level, extractor):
    logger.info('Running w2v similarity test with {} as documents and {} as extractor.'.format(level, str(extractor)))
    w2v, X = get_w2v(level, extractor, size=100, sg=0, iter=80, alpha=0.025)
    transformer = partial(word2vec_transformer, w2v, True)
    blop = similarity_method(transformer, questions)
    print(blop)
    # print((size, sg, iter, blop))

def tfidf_similarity_test(questions, level, extractor):
    logger.info('Running TF-IDF similarity test with {} as documents and {} as extractor.'.format(level, str(extractor)))
    tfidf, _ = get_tfidf(level, extractor)
    transformer = partial(tfidf_transformer, tfidf, True)
    blop = similarity_method(transformer, questions)
    print(blop)

def ir_tfidf_test_clauses(questions, level, extractor):
    logger.info('Running IR TF-IDF test with {} as documents and {} as extractor.'.format(level, str(extractor)))
    # X, y, tfidf = generate_clause_set(get_clause_id, 1)
    tfidf, X = get_tfidf(level, extractor)
    results = []
    transformer = partial(tfidf_transformer, tfidf, False)
    results.append(ir_method(transformer, X, 6, questions))

    logger.info(results)

def ir_w2v_test_old(questions):
    logger.info('Running IR W2V test with clauses as documents.')
    w2v = get_w2v2(100, 0.025, 80)
    docs = generate_w2v_corpus(w2v)
    print w2v.most_similar('client')
    transformer = partial(word2vec_transformer, w2v, False)
    results = []
    for i in [5]:
        results.append(ir_method(transformer, docs, i, questions))

    logger.info(results)

def ir_w2v_test(questions, level, extractor):
    logger.info('Running w2v TF-IDF test with {} as documents and {} as extractor.'.format(level, str(extractor)))
    w2v, X = get_w2v(level, extractor, size=150, alpha=0.025, sg=1, iter=60)
    print w2v.most_similar('client')
    transformer = partial(word2vec_transformer, w2v, False)
    results = []
    results.append(ir_method(transformer, X, 5, questions))

    logger.info(results)

def discrim_tfidf_test(questions, test_questions, level, extractor):
    logger.info('Running discriminative test using TF-IDF')
    # _, _, tfidf = generate_clause_set(get_clause_id, 1)
    tfidf, _ = get_tfidf(level, extractor)
    transformer = partial(tfidf_transformer, tfidf, True)
    cls = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # cls = LogisticRegressionCV(scoring='roc_auc')
    result = discrim_method(transformer, questions, test_questions, cls, 10)
    logger.info(result)

def discrim_w2v_test(train_questions, test_questions, level, extractor):
    logger.info('Running discriminative test using w2v')
    # w2v = get_w2v(100, 0.025, 80)
    w2v, _ = get_w2v(level, extractor, size=100, alpha=0.025, sg=0, iter=80, workers=8, sample=0)
    transformer = partial(word2vec_transformer, w2v, True)
    cls = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # cls = LogisticRegressionCV(scoring='roc_auc')
    result = discrim_method(transformer, train_questions, test_questions, cls, 10)
    logger.info(result)

def discrim_blended_test(train_questions, test_questions, level, extractor):
    logger.info('Running discriminative test using a blended method')
    # _, _, tfidf = generate_clause_set(get_clause_id, 1)
    tfidf, _ = get_tfidf(level, extractor)
    # w2v = get_w2v(100, 0.025, 80)
    w2v, _ = get_w2v(level, extractor, size=100, alpha=0.025, sg=0, iter=80, workers=8, sample=0)
    for alpha in [0, 0.5, 1]:
        transformer = partial(blended_transformer, w2v, tfidf, True, alpha)
        # param_grid = {'n_estimators': [100, 500, 1000, 2000],
        #                 'max_depth': [2, 4, 8]}
        # cls = GridSearchCV(RandomForestClassifier(), param_grid, verbose=3, n_jobs=-1)
        cls = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        # cls = LogisticRegressionCV(scoring='roc_auc')
        result = discrim_method(transformer, train_questions, test_questions, cls, 2)
        logger.info((alpha, result))

def encode_for_keras(train_questions, test_questions, level, extractor):
    logger.info('Encoding for keras')
    w2v, _ = get_w2v(level, extractor, size=100, sg=1, iter=40)
    q_bodies = [q.body for q in questions]
    d_split = int(len(q_bodies) * 0.8)
    rnd = np.random.permutation(len(q_bodies))
    q_gen = pd.read_csv('generated_everything_n_lim.csv')
    qa = zip(q_gen['0'].astype(str), q_gen['1'].astype(str))
    # encode_questions_with_gen(w2v, questions, rnd[:d_split], qa)
    encode_questions(w2v, train_questions, test_questions)

def encode_generated_for_keras(filename, level, extractor):
    questions = pd.read_csv(filename)
    q2 = pd.read_csv('generated_everything_n_lim.csv')
    questions = pd.concat([q2, questions])
    w2v, _ = get_w2v(level, extractor, size=100, sg=1, iter=40)
    # w2v, _ = get_w2v(level, extractor, size=100, alpha=0.025, sg=0, iter=80, workers=8, sample=0)
    # true_qs = questions
    # blp = pd.read_csv('generated_qs.csv')
    # blp = blp[blp['3']]
    # qs = 52.209.168.116pd.concat([blp['1'], true_qs['1'].astype(str)])
    # a_s = pd.concat([blp['2'], true_qs['2'].astype(str)])
    encode_generated_questions(w2v, questions['0'].astype(str), questions['1'].astype(str))

def run_similarity_methods(source, questions):
    tfidf_similarity_test(questions, source, edit_sq_group)
    tfidf_similarity_test(questions, source, extract_entities)
    tfidf_similarity_test(questions, source, reference)
    tfidf_similarity_test(questions, source, entity_and_reference)
    w2v_similarity_test(questions, source, edit_sq_group)
    w2v_similarity_test(questions, source, extract_entities)
    w2v_similarity_test(questions, source, reference)
    w2v_similarity_test(questions, source, entity_and_reference)

def run_nn_methods(source, questions):
    ir_tfidf_test_clauses(questions, source, edit_sq_group)
    ir_tfidf_test_clauses(questions, source, extract_entities)
    ir_tfidf_test_clauses(questions, source, reference)
    ir_tfidf_test_clauses(questions, source, entity_and_reference)
    ir_w2v_test(questions, source, edit_sq_group)
    ir_w2v_test(questions, source, extract_entities)
    ir_w2v_test(questions, source, reference)
    ir_w2v_test(questions, source, entity_and_reference)

def run_discrim_methods(source, train_questions, test_questions):
    discrim_tfidf_test(train_questions, test_questions, source, edit_sq_group)
    discrim_tfidf_test(train_questions, test_questions, source, extract_entities)
    discrim_tfidf_test(train_questions, test_questions, source, reference)
    discrim_tfidf_test(train_questions, test_questions, source, entity_and_reference)
    discrim_w2v_test(train_questions, test_questions, source, edit_sq_group)
    discrim_w2v_test(train_questions, test_questions, source, extract_entities)
    discrim_w2v_test(train_questions, test_questions, source, reference)
    discrim_w2v_test(train_questions, test_questions, source, entity_and_reference)

# @memory.cache
def get_pairwise_dataset(questions, docs, w2v, transformer):
    nn = NearestNeighbors(1, algorithm='brute', metric='cosine')
    nn.fit(docs)
    sets = []
    sanity = 0
    tot = 0
    for q in questions:
        query_vectors = transformer(q)
        neigh = nn.kneighbors(query_vectors)
        dist = neigh[0].reshape(-1,1)

        sim_trans = word2vec_transformer(w2v, True, q)
        query_vectors = scipy.sparse.csr_matrix(sim_trans)
        q_part = query_vectors[:,:query_vectors.shape[1]/2]
        a_part = query_vectors[:,query_vectors.shape[1]/2:]

        sim = np.einsum('ij,ij->i', q_part.toarray(), a_part.toarray()).reshape(-1,1)
        if sim.argmax() == q.correct:
            sanity += 1
        colon = 0
        if ':' in q.body:
            colon = 1

        colon = (np.ones(4)*colon).reshape(-1,1)
        length = (np.ones(4)*len(q.body)).reshape(-1,1)
        a_type = (np.ones(4)*(q.a_type=='v')).reshape(-1,1)
        a_len = np.array([len(a) for a in q.all_answers()]).reshape(-1,1)
        wh_blocks = []
        for w in ['which', 'what', 'who', 'how']:
            wh_blocks.append((np.ones(4)*(w in q.body.lower())).reshape(-1,1))

        sim = np.einsum('ij,ij->i', q_part.toarray(), a_part.toarray()).reshape(-1,1)
        if sim.argmax() == q.correct:
            sanity += 1
        q_type = (np.ones(4) * q.type).reshape(-1,1)
        x = np.hstack(wh_blocks + [q_part.toarray(), dist, sim, colon, length, a_type, a_len,q_type])
        labs = []
        x_vecs = []
        ys = []
        for a, b in product([0,1,2,3],[0,1,2,3]):
            if a != b:
                # if a == q.correct or b == q.correct:
                x_vec = np.hstack((x[a,:], x[b,:]))
                if q.correct == a:
                    y = 0
                elif q.correct == b:
                    y = 1
                else:
                    y = 0

                x_vecs.append(x_vec)
                ys.append(y)
                labs.append((a, b))
        sets.append((q, labs, np.vstack(x_vecs), np.array(ys)))
    print sanity / float(len(questions))
    return sets

def get_ensemble_dataset(questions, docs, w2v, transformer):
    nn = NearestNeighbors(1, algorithm='brute', metric='cosine')
    nn.fit(docs)
    sets = []
    sanity = 0
    tot = 0
    vals = []
    ys = []
    correct = []
    for q in questions:
        query_vectors = transformer(q)
        neigh = nn.kneighbors(query_vectors)
        dist = neigh[0].reshape(-1,1)

        sim_trans = word2vec_transformer(w2v, True, q)
        query_vectors = scipy.sparse.csr_matrix(sim_trans)
        q_part = query_vectors[:,:query_vectors.shape[1]/2]
        a_part = query_vectors[:,query_vectors.shape[1]/2:]
        colon = 0
        if ':' in q.body:
            colon = 1

        colon = (np.ones(4)*colon).reshape(-1,1)
        length = (np.ones(4)*len(q.body)).reshape(-1,1)
        a_type = (np.ones(4)*(q.a_type=='v')).reshape(-1,1)
        a_len = np.array([len(a) for a in q.all_answers()]).reshape(-1,1)
        wh_blocks = []
        for w in ['which', 'what', 'who', 'how']:
            wh_blocks.append((np.ones(4)*(w in q.body.lower())).reshape(-1,1))

        sim = np.einsum('ij,ij->i', q_part.toarray(), a_part.toarray()).reshape(-1,1)
        if sim.argmax() == q.correct:
            sanity += 1
        q_type = (np.ones(4) * q.type).reshape(-1,1)

        vals.append(np.concatenate(wh_blocks + [q_part.toarray(), dist, q_type, a_len, a_type, sim, colon, length], axis=1))
        y = np.zeros(4)
        y[q.correct] = 1
        ys.append(y)
        correct.append(q.correct)

    all_y = np.concatenate(ys)
    X = np.vstack(vals)
    # print(np.mean(cross_val_score(LogisticRegressionCV(scoring='roc_auc'), X, all_y, scoring='roc_auc')))
    # print(np.mean(cross_val_score(RandomForestClassifier(500), X, all_y, scoring='roc_auc')))
    # print(np.mean(cross_val_score(RandomForestClassifier(2000), X, all_y, scoring='roc_auc', cv=8)))
    return X, all_y, correct

    # pass

# @memory.cache
def get_pairwise_ds(train_questions, test_questions, level, extractor):
    tfidf, X = get_tfidf(level, extractor)
    w2v, _ = get_w2v('clause', extractor, size=100, sg=0, iter=80, alpha=0.025)
    transformer = partial(tfidf_transformer, tfidf, False)
    train_set = get_pairwise_dataset(train_questions, X, w2v, transformer)
    test_set = get_pairwise_dataset(test_questions, X, w2v, transformer)
    return train_set, test_set

def discriminative_pairwise(train_set, test_set):
    # lr = LogisticRegressionCV(scoring='roc_auc', cv=8)
    lr = ExtraTreesClassifier(500, n_jobs=-1)
    X_train = np.vstack([t[2] for t in train_set])
    y_train = np.concatenate([t[3] for t in train_set])
    scaler = StandardScaler()
    # x_scaled = X_train
    lr.fit(X_train, y_train)
    # print lr.coef_
    corr = 0
    tot = 0
    mrr = []
    for q, labs, x_vecs, ys in test_set:
        # x_vecs_scaled = scaler.transform(x_vecs)
        x_vecs_scaled = x_vecs
        ranks = [[],[],[],[]]
        ranks = np.zeros(4)
        x_pred = lr.predict_proba(x_vecs)
        for i in range(len(ys)):
            a, b = labs[i]
            ranks[a] += x_pred[i][0]
            ranks[b] += x_pred[i][1]
            # ranks[a] -= x_pred[i][2]
            # ranks[b] -= x_pred[i][2]
            # ranks[a] = x_pred[i][0])
            # ranks[b] *= np.log(x_pred[i][1])
            # ranks[a] -= np.log(x_pred[i][2])
            # ranks[b] -= np.log(x_pred[i][2])

        if np.argmax(ranks) == q.correct:
            corr += 1
        rnk = np.argsort(ranks)
        mrr.append(1.0 / (rnk[q.correct]+1))
        tot += 1
    print('mrr {}'.format(np.mean(mrr)))
    print corr, tot, corr / float(tot)

def discriminative_straight(train_questions, test_questions):
    tfidf, docs = get_tfidf('statement', reference)
    w2v, _ = get_w2v('clause', reference, size=100, sg=0, iter=80, alpha=0.025)
    transformer = partial(tfidf_transformer, tfidf, False)
    X, y, _ = get_ensemble_dataset(train_questions, docs, w2v, transformer)
    test_X, test_y, corr = get_ensemble_dataset(test_questions, docs, w2v, transformer)

    # poly = PolynomialFeatures(2, True)
    # X = poly.fit_transform(X)
    #0.537878787879
    #0.321678321678
    GradientBoostingClassifier()
    param_grid = {'n_estimators': [500, 1000, 2000],
                  'learning_rate': [0.1, 0.01, 0.001],
                  'max_depth': [2,3,6],
                  'subsample': np.linspace(0.01,0.5,20),
                  }
    rf = RandomizedSearchCV(GradientBoostingClassifier(), param_grid, n_iter=100, scoring='roc_auc', cv=8, verbose=3, n_jobs=-1)

    # rf = LogisticRegressionCV(scoring='roc_auc', cv=8)
    rf = ExtraTreesClassifier(500, n_jobs=-1)
    rf.fit(X, y)
    y_prob = rf.predict_proba(test_X)
    # print(y_prob)
    # print(y_prob.shape)
    y_pred = y_prob[:,1].reshape(len(test_y)/4, 4).argmax(axis=1)
    rnk = y_prob[:,1].reshape(len(test_y)/4, 4).argsort(axis=1)
    rr = []
    for i in range(len(corr)):
        rr.append(1.0/(rnk[i,corr[i]]+1))
    print np.mean(rr)
    print (y_pred == np.array(corr)).mean()


def get_tsne():
    w2v, _ = get_w2v('statement', reference, size=150, sg=1, iter=60, alpha=0.025)
    t = TSNE(verbose=3)
    X = t.fit_transform(w2v.syn0)
    vocab = {}
    for word in w2v.vocab:
        vocab[w2v.vocab[word].index+1] = word

    pickle.dump((X, vocab), open('tsne.p', 'wb'))

if __name__ == '__main__':

    # questions = list(session.query(EntityQuestion).filter(EntityQuestion.type.in_([0,1])))
    questions = list(session.query(Question).filter(Question.type.in_([0,1])))
    np.random.seed(999)
    rand_ind = np.random.permutation(len(questions))
    test_ind = rand_ind[:len(questions)*0.2]
    train_ind = rand_ind[len(questions)*0.2:]
    test_questions = [questions[i] for i in test_ind]
    train_questions = [questions[i] for i in train_ind]
    # get_tsne()
    # w2v_similarity_test(train_questions, 'statement', reference)
    data = get_pairwise_ds(train_questions, test_questions, 'statement', reference)
    discriminative_pairwise(*data)
    discriminative_straight(train_questions, test_questions)

    run_nn_methods('clause', test_questions)
    run_nn_methods('statement', test_questions)
    run_similarity_methods('clause', test_questions)
    run_similarity_methods('statement', test_questions)
    
    encode_for_keras(train_questions, test_questions, 'statement', reference)
    encode_generated_for_keras('generated_everything.csv', 'statement', reference)