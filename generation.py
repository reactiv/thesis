import pickle
from random import seed
from nltk import AlignedSent, IBMModel2
import pandas as pd
from scipy.sparse import vstack, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_auc_score
from cleaning import entity_and_reference, condense_ref
from inference import get_w2v

__author__ = 'jamesgin'

from features import *
import spacy

# nlp = spacy.load('en')

from nltk.translate.ibm3 import IBMModel3

censored = ['it meets the following conditions']

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
    fq = fq / np.linalg.norm(fq, axis=1)[:, None]
    pass

def questions_checker(output_file_name, for_parser):
    questions = session.query(Question).order_by(Question.id)
    with open(output_file_name, 'wb') as file_with_reference:
        with open(for_parser, 'wb') as file_for_parser:
            lines = []
            parser_lines = []
            for q in questions:
                lines.append('{},{}\n'.format(q.id, q.body.encode('ascii', 'ignore')))
                parser_lines.append(q.body + '\n')
                # print(clause.header)
                # print(section.name)
            file_with_reference.writelines(lines)
            file_for_parser.writelines(parser_lines)

def statement_checker(output_file_name, for_parser, extractor):
    nlp = spacy.load('en')
    statements = session.query(StatementPart, RawClause, Section).join(RawClause).join(Section).filter((StatementPart.parent_id.is_(None)))
                                                                         # & (Section.docpath.like('COBS%'))
                                                                         # & (Section.id == 2926) # .in_(range(2961, 2969))) #
                                                                         # & (RawClause.content_html.notilike('%<table%'))).order_by(StatementPart.id)
    with open(output_file_name, 'wb') as file_with_reference:
        with open(for_parser, 'wb') as file_for_parser:
            lines = []
            parser_lines = []
            for statement, clause, section in statements:

                # print(clause.name)
                # shortest = statement.shortest_sentence()
                for shortest in statement.all_sentences():
                    sent = fix_multipart_sentence(shortest)
                    sent = sent.replace(u'\u2019', "'")
                    if 'should present the information referred to in' in sent:
                        pass
                    sent = clean_sentence(sent, extractor)
                    doc = nlp(sent)

                    for s in doc.sents:
                        cleaned = str(s).replace(':', '')
                        if 'deleted' not in cleaned.lower() and 'Note ]' not in cleaned:
                            # if cleaned.startswith("If  a communication relates to a firm's business that is not"):
                            #     cleaned = clean_sentence(str(s), extractor)
                            print(cleaned)
                            lines.append('{},{}\n'.format(statement.id, cleaned))
                            parser_lines.append(cleaned + '\n')
                # print(clause.header)
                # print(section.name)
            file_with_reference.writelines(lines)
            file_for_parser.writelines(parser_lines)

def generate_clause_sentences(statements):
    """
    For multi sentence clauses, ascertain contextual information and then obligation / guidance
    For list based sentences, convert these into a single grammatical sentence.
    :param statements:
    :return:
    """
    pass

def clip_end(part):
    for i in [';', ' ;', '; and', '; or']:
        if part.endswith(i):
            part = part[:-len(i)]

    return part

def generate_distractors():
    pass

def load_additional_context():
    context = session.query(StatementPart.id, RawClause.header, Section.name).join(RawClause).join(Section)\
        .filter(Section.name.like('%COBS%')).all()
    return pd.DataFrame(context).set_index('id')


def remove_multiple_statements(part):
    for c in censored:
        part = part.replace(c, '')

    return part

def fix_multipart_sentence(sentence):
    sentence = [clip_end(p).strip() for p in sentence]
    sent_string = ' '.join(sentence)
    if not sent_string.endswith('.'):
        sent_string += '.'
    return sent_string

def get_test_vecs(tfidf, questions, concatenated):
    X = []
    y = []

    true_ans = []

    for q in questions:
        # print(q.body)
        if concatenated:
            q_vec = tfidf.transform([q.body])
            a_vecs = [tfidf.transform([a]) for a in q.all_answers()]
            vecs = [hstack([q_vec, a]) for a in a_vecs]
        else:
            docs = [q.body + '. ' + a for a in q.all_answers()]
            vecs = tfidf.transform(docs)

        X.extend(vecs)
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)
        true_ans.append(q.correct)

    X = vstack(X)
    y = np.concatenate(y)
    return X, y

# @memory.cache
def tfidf_test(tfidf, train_x, train_y, questions, concatenated):
    correct = 0
    # total = float(questions.count())
    X = []
    y = []

    true_ans = []

    for q in questions:
        # print(q.body)
        if concatenated:
            q_vec = tfidf.transform([q.body])
            a_vecs = [tfidf.transform([a]) for a in q.all_answers()]
            vecs = [hstack([q_vec, a]) for a in a_vecs]
        else:
            docs = [q.body + '. ' + a for a in q.all_answers()]
            vecs = tfidf.transform(docs)

        X.extend(vecs)
        ys = np.zeros(4)
        ys[q.correct] = 1
        y.append(ys)
        true_ans.append(q.correct)

    X = vstack(X)
    y = np.concatenate(y)
    print('Training')
    lr = RandomForestClassifier(n_estimators=100)
    # lr = ExtraTreesClassifier(1000)

    # lr.fit(train_x, train_y)
    # print('Scoring')


    # lr = ExtraTreesClassifier(n_estimators=1000, n_jobs=8, verbose=3)
    # lr = LogisticRegressionCV()
    # lr = Perceptron('l1', n_jobs=-1)

    lr.fit(train_x, train_y)

    # print('Scoring')
    # print(roc_auc_score(train_y, lr.predict_proba(train_x)[:,1]))
    proba = lr.predict_proba(X)
    # print(proba)
    pred_ans = proba[:,1].reshape(len(questions),4).argmax(axis=1)
    # print pred_ans
    # print true_ans
    # print(y)
    # print(roc_auc_score(y, lr.predict_proba(X)[:,1]))
    print((pred_ans == true_ans).mean())
    return (pred_ans == true_ans).mean()

def get_vector(tfidf, q, a, concatenate):
    if concatenate:
        q_vec = tfidf.transform([q])
        a_vec = tfidf.transform([a])
        return hstack([q_vec, a_vec])
    else:
        return tfidf.transform([q + ' ' + a])


# @memory.cache()
def get_vecs(no_wrong, qs, answers, concatenated, header_context):
    all_vecs = []

    tf = []
    allqs = []
    allas = []
    for idx, rows in qs.iterrows():
        try:
            # add_cont = 'Regarding {}, '.format(context.ix[rows.ix[1]]['header'].lower().encode('ascii', 'ignore'))
            q = rows.ix[1]
            # if len(q) < 100:
            # if header_context:
            #     q = add_cont + q
            a = rows.ix[2]
            allqs.append(q)
            allas.append(a)
            rand_indexs = np.random.permutation(len(answers))[:no_wrong]
            rnds = answers.as_matrix()[rand_indexs]
            tf.append(1)

            for i in range(no_wrong):
                allqs.append(q)
                allas.append(rnds[i])
                tf.append(0)

        except:
            pass

    q_vecs = tfidf.transform(allqs)
    a_vecs = tfidf.transform(allas)

    X = hstack([q_vecs, a_vecs])
    y = np.array(tf)


    return X, y

def get_vecs_no_gen(qs, concatenated, header_context):
    q_vecs = tfidf.transform(qs['0'].as_matrix())
    a_vecs = tfidf.transform(qs['1'].as_matrix())
    X = hstack((q_vecs, a_vecs))
    y = qs['3']
    return X, y

def sanity_check(no_wrong, qs, answers, concatenated, header_context):
    all_vecs = []
    tf = []
    q = "A firm must ensure that information that contains an indication of past performance of relevant business , a relevant investment or a financial index , satisfies the following conditions : the information includes appropriate performance information which covers at least the immediately preceding how many years , or the whole period for which the investment has been offered , the financial index has been established , or the service has been provided if less than five years , or such longer period as the firm may decide , and in every case that performance information must be based on and show complete 12-month periods ?"
    t_a = "at least the immediately preceding five years , or the whole period for which the investment has been offered"
    f_a = "at least the immediately preceding ten years , or the whole period for which the investment has been offered"

    q2 = 'A firm must ensure that information that contains an indication of past performance of relevant business , a relevant investment , a structured deposit or a financial index , satisfies the following conditions : it discloses the effect of commissions , fees or other charges if the indication is based on which of the following ?'
    t2_a = 'gross performance'
    f2_a = 'net performance'

    q3 = "A firm must ensure that information that contains an indication of past performance of relevant business , a relevant investment or a financial index , satisfies which of the following?"
    t3_a = 'the following conditions : the reference period and the source of information are clearly stated'
    f3_a = 'the following conditions : the reference period and the source of information are unclearly stated'

    vec1 = get_vector(tfidf, q, t_a, True)
    vec2 = get_vector(tfidf, q, f_a, True)

    vec3 = get_vector(tfidf, q2, t2_a, True)
    vec4 = get_vector(tfidf, q2, f2_a, True)

    vec5 = get_vector(tfidf, q3, t3_a, True)
    vec6 = get_vector(tfidf, q3, f3_a, True)

    X = vstack([vec1, vec2, vec3, vec4, vec5, vec6])
    y = np.array([1,0,1,0,1,0])

    return X, y

if __name__ == '__main__':
    # statement_checker('ref_perf.txt', 'to_parser_perf.txt', condense_ref)
    # statement_checker('ref_suitability.txt', 'to_parser_suitability.txt', condense_ref)
    # statement_checker('ref_colon.txt', 'to_parser_colon.txt', condense_ref)
    # statement_checker('ref_perf_colon.txt', 'to_parser_perf_colon.txt', condense_ref)
    # statement_checker('ref_everything.txt', 'to_parser_everything.txt', condense_ref)


    # context = load_additional_context()
    _, _, tfidf = generate_clause_set(get_clause_id, 1)
    #
    # q = "A firm must ensure that information that contains an indication of past performance of relevant business , a relevant investment or a financial index , satisfies the following conditions : the information includes appropriate performance information which covers at least the immediately preceding how many years , or the whole period for which the investment has been offered , the financial index has been established , or the service has been provided if less than five years , or such longer period as the firm may decide , and in every case that performance information must be based on and show complete 12-month periods ?"
    # t_a = "at least the immediately preceding five years , or the whole period for which the investment has been offered"
    # f_a = "at least the immediately preceding ten years , or the whole period for which the investment has been offered"
    #
    # q2 = 'A firm must ensure that information that contains an indication of past performance of relevant business , a relevant investment , a structured deposit or a financial index , satisfies the following conditions : it discloses the effect of commissions , fees or other charges if the indication is based on which of the following ?'
    # t2_a = 'gross performance'
    # f2_a = 'net performance'
    #
    # q3 = "A firm must ensure that information that contains an indication of past performance of relevant business , a relevant investment or a financial index , satisfies which of the following?"
    # t3_a = 'the following conditions : the reference period and the source of information are clearly stated'
    # f3_a = 'the following conditions : the reference period and the source of information are unclearly stated'
    #
    # tfidf.fit_transform([q + ' ' + t_a, q + ' ' + f_a,
    # #                              q2 + ' ' + t2_a, q2 + ' ' + f2_a,
    # #                              q3 + ' ' + t3_a, q3 + ' ' + f3_a])
    # #
    #
    #
    #
    #

    factor = 'suitability'
    qs = pd.read_csv('generated_perf.csv')
    # for a in qs['2']:
    #     print(a)
    # qs = qs[(qs['2'].str.split().apply(len) > 1)]
    tfidf.fit(qs['0'] + ' ' + qs['1'])
    # tfidf.transform((qs['0'] + ' ' + qs['1']).as_matrix())
    seed(0)
    res = []
    for i in range(100):
        answers_1 = qs['1'].fillna('')
        for no_wrong in [4]:
            X, y = get_vecs(no_wrong, qs, answers_1, concatenated=True, header_context=False)
            # X, y = get_vecs_no_gen(qs, concatenated=True, header_context=False)
            # X, y = sanity_check(no_wrong, qs, answers_1, concatenated=True, header_context=False)
            # questions = session.query(Question).filter((Question.id.in_([695, 724, 659, 413, 332])))
            # tfidf_test(tfidf, X, y, questions, True)
            questions = session.query(Question).filter(Question.type.in_([0,1])).all()
            questions = session.query(Question).filter(Question.body.ilike('%suitability%')).all()
            et = tfidf_test(tfidf, X, y, questions, True)
            res.append(et)
            # et = pickle.load(open('et.p', 'rb'))
            # X_test, y_test = get_test_vecs(tfidf, questions, True)
            # proba = et.predict_proba(X_test)
            pass

    print np.mean(res)
    # statement_checker('ref_all.txt', 'to_parser_all.txt')
    # questions_checker('q_ref.txt', 'q_to_parser.txt')
    # question_cluster()
    # ibm = get_ibm_model3()
    # semi_sup()