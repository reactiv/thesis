import spacy

__author__ = 'jamesgin'
from dbconn import *
from model import *

heads = session.query(Statement.head).distinct()
print(heads.count())
# nlp = spacy.load('en')

colons = 0
for h in heads:
    if h[0] is not None:
        text = h[0].strip()
        if text[-1] == ':':
            colons += 1

print(colons)
