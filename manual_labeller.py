import os
import textwrap
from sqlalchemy import func
__author__ = 'jamesgin'

from features import *
from model import *

# Get a list of all unlabelled questions, display the first n closest matches
# If any are correct, accept prompt to mark as done
# Also display if the question currently correct given a very basic cosine distance method!

def get_all_unlabelled():
    return session.query(Question).filter(Question.related_clause == None)

def get_nearest_neighbours(nn, question_vec, ids):
    indices = nn.kneighbors(question_vec, return_distance=False)
    near_ids = ids[indices]
    clauses = session.query(RawClause).filter(RawClause.id.in_(near_ids[0].tolist()))
    return clauses

def get_mentions(txt):
    clauses = session.query(RawClause).filter(RawClause.cleaned.ilike('%{}%'.format(txt))).all()
    return [c.id for c in clauses]

def print_clauses(n_clauses, highlight):
    for n in n_clauses:
        print('-'*160)
        print(n.id, highlight in n.cleaned.lower(), n.header)
        print('-'*160)
        print(textwrap.fill(n.cleaned, 160))

if __name__ == '__main__':
    X, y, tfidf = generate_clause_set(get_clause_id)

    unlabelled = get_all_unlabelled()
    for q in unlabelled:
        os.system('clear')

        neigh = NearestNeighbors(5, algorithm='brute', metric='cosine')
        neigh.fit(X)
        ans = q.get_correct()
        q_vec = tfidf.transform([q.text()])
        n_clauses = get_nearest_neighbours(neigh, q_vec, y)
        print_clauses(n_clauses, q.get_correct().lower())

        print('-'*160)
        print(q.body)
        print(q.get_correct())

        print('-'*160)
        id = raw_input('Are any correct?')
        if not id.isdigit():
            parts = id.split(',')

            mentioned = get_mentions(parts[0])
            mask = np.in1d(y, mentioned)
            neigh = NearestNeighbors(10, algorithm='brute', metric='cosine')

            try:
                neigh.fit(X[mask,:])
                nn_clauses = get_nearest_neighbours(neigh, q_vec, y[mask])
                print_clauses(nn_clauses, parts[1])
            except:
                print('Nothing')
            id = raw_input('Are any correct?')
        id = int(id)
        if id == 0:
            print('Passed')
            q.related_clause = 54488
            session.commit()
        elif id < 5:
            id = n_clauses[id].id
            q.related_clause = id
            session.commit()
            print(id)
        else:
            q.related_clause = id
            session.commit()
            print(id)