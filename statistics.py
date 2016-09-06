__author__ = 'jamesgin'

from model import *
from features import *

questions = session.query(Question)
i = 0
q_lens = []
a_lens = []
for q in questions:
    for a in q.all_answers():
        ans = tokenise(a)
        ques = tokenise(q.body)
        q_lens.append(len(ques))
        a_lens.append(len(ans))
        for t in ans:
            if t in ques and t not in eng_stop:
                i += 1
                break

print(i)
print(questions.count())
print np.mean(q_lens)
print np.mean(a_lens)
print np.max(q_lens)
print np.max(a_lens)