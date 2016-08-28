from functools import partial
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from cleaning import clean_sentence, extract_entities, edit_sq_group, reference, entity_and_reference, get_entities
from inference import get_w2v as get_w2v2

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

            chosen = dist.argmin()
            if chosen == q.correct:
                correct[i-1] += 1

    return np.array(correct) / float(questions.count())

def discrim_method(transformer, questions, classifier, folds):

    total = float(questions.count())

    q_list = list(questions)

    scores = []
    for train, test in KFold(int(total), folds, shuffle=True):
        X = []
        test_X = []
        y = []
        test_y = []
        train_qs = [q_list[i] for i in train]
        test_qs = [q_list[i] for i in test]
        for q in train_qs:
            vecs = transformer(q)
            X.append(vecs)
            ys = np.zeros(4)
            ys[q.correct] = 1
            y.append(ys)

        for q in test_qs:
            vecs = transformer(q)
            test_X.append(vecs)
            test_y.append(q.correct)
        try:
            X = scipy.sparse.vstack(X)
            test_X = scipy.sparse.vstack(test_X)
        except:
            X = np.vstack(X)
            test_X = np.vstack(test_X)

        y = np.concatenate(y)

        test_y = np.array(test_y)
        classifier.fit(X, y)
        y_prob = classifier.predict_proba(test_X)
        y_pred = y_prob[:,1].reshape(len(test), 4).argmax(axis=1)
        scores.append((test_y == y_pred).mean())
    return np.mean(scores)



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
    w2v, X = get_w2v(level, extractor, size=100, alpha=0.025, sg=0, iter=80, workers=8, sample=0)
    print w2v.most_similar('client')
    transformer = partial(word2vec_transformer, w2v, False)
    results = []
    results.append(ir_method(transformer, X, 5, questions))

    logger.info(results)

def discrim_tfidf_test(questions, level, extractor):
    logger.info('Running discriminative test using TF-IDF')
    # _, _, tfidf = generate_clause_set(get_clause_id, 1)
    tfidf, _ = get_tfidf(level, extractor)
    transformer = partial(tfidf_transformer, tfidf, True)
    cls = RandomForestClassifier(n_estimators=1000, verbose=1, n_jobs=-1)
    cls = LogisticRegressionCV(scoring='roc_auc')
    result = discrim_method(transformer, questions, cls, 10)
    logger.info(result)

def discrim_w2v_test(questions, level, extractor):
    logger.info('Running discriminative test using w2v')
    # w2v = get_w2v(100, 0.025, 80)
    w2v, _ = get_w2v(level, extractor, size=100, alpha=0.025, sg=0, iter=80, workers=8, sample=0)
    transformer = partial(word2vec_transformer, w2v, True)
    cls = RandomForestClassifier(n_estimators=1000, verbose=1, n_jobs=-1)
    cls = LogisticRegressionCV(scoring='roc_auc')
    result = discrim_method(transformer, questions, cls, 20)
    logger.info(result)

def discrim_blended_test(questions, level, extractor):
    logger.info('Running discriminative test using a blended method')
    # _, _, tfidf = generate_clause_set(get_clause_id, 1)
    tfidf, _ = get_tfidf(level, extractor)
    # w2v = get_w2v(100, 0.025, 80)
    w2v, _ = get_w2v(level, extractor, size=100, alpha=0.025, sg=0, iter=80, workers=8, sample=0)
    for alpha in np.linspace(0,1,6):
        transformer = partial(blended_transformer, w2v, tfidf, True, alpha)
        # cls = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
        cls = LogisticRegressionCV(scoring='roc_auc')
        result = discrim_method(transformer, questions, cls, 10)
        logger.info((alpha, result))







if __name__ == '__main__':
    questions = session.query(Question).filter(Question.type.is_(None))
    entity_questions = session.query(EntityQuestion).filter(EntityQuestion.type.is_(None))
    # ir_tfidf_test_statements(questions)
    # ir_tfidf_test_clauses(questions, 'statement', edit_sq_group)
    # ir_tfidf_test_clauses(questions, 'statement', extract_entities)
    # ir_tfidf_test_clauses(entity_questions, 'statement', extract_entities)
    # ir_tfidf_test_clauses(questions, 'statement', reference)
    # ir_tfidf_test_clauses(questions, 'statement', entity_and_reference)
    # ir_tfidf_test_clauses(entity_questions, 'statement', entity_and_reference)
    # ir_tfidf_test_clauses(questions, 'clause', edit_sq_group)
    # ir_tfidf_test_clauses(questions, 'clause', extract_entities)
    # ir_tfidf_test_clauses(entity_questions, 'clause', extract_entities)
    # ir_tfidf_test_clauses(questions, 'clause', reference)
    # ir_tfidf_test_clauses(questions, 'clause', entity_and_reference)
    # ir_w2v_test(questions, 'statement', edit_sq_group)
    # ir_w2v_test(questions, 'statement', extract_entities)
    # ir_w2v_test(questions, 'statement', reference)
    # ir_w2v_test(questions, 'statement', entity_and_reference)
    # ir_w2v_test(questions, 'clause', edit_sq_group)
    # ir_w2v_test(questions, 'clause', extract_entities)
    # ir_w2v_test(questions, 'clause', reference)
    # ir_w2v_test(questions, 'clause', entity_and_reference)
    # ir_w2v_test_old(questions)
    # discrim_tfidf_test(questions)
    # discrim_w2v_test(questions)
    discrim_blended_test(questions, 'statement', entity_and_reference)
    discrim_blended_test(entity_questions, 'statement', entity_and_reference)
    # ir_tfidf_test_statements(questions)