import csv
from io import StringIO

__author__ = 'jamesgin'
from features import *
import subprocess as sp
import os
import pandas as pd
from nltk.draw.tree import Tree, TreeWidget
from nltk.draw.util import CanvasFrame
import spacy
import textwrap

nlp = spacy.load('en')


senna_path = '/Users/jamesgin/projects/senna/'

def get_senna_output(sentence):
    p = sp.Popen(['blabla', '-path',  senna_path],
             executable=os.path.join(senna_path, 'senna-osx'),
             stdin=sp.PIPE,
             stdout=sp.PIPE)
    tagged = StringIO(unicode(p.communicate(sentence)[0]))
    # table = csv.reader(tagged, dialect='excel-tab')
    table = pd.read_table(tagged, header=None)
    return table

def dive(tok):
    children = list(tok.children)
    if not children:
        return '({} {})'.format(tok.dep_, tok)
    else:
        return '({} {})'.format(tok.dep_, ''.join([dive(t) for t in children]))


def analyse_questions():
    questions = session.query(Question).all()
    lens = []
    for q in questions:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', q.body)
        lens.append(len(sentences))
        if len(sentences) == 1 and sentences[0][-1] == '?':
            # parse = get_senna_output(sentences[0])
            # tree_string = ''.join(parse.iloc[:, -1])
            # tree_string = tree_string.replace('*', ' %s ')
            # tree_string = tree_string % tuple(parse.iloc[:, 0].str.strip())
            tree_string2 = make_string(nlp(sentences[0]).sents.next())
            # print(tree_string)
            print(tree_string2)
            pass
            # print(parse)
            tree(tree_string2)
    pass

def analyse_clauses():
    clauses = session.query(RawClause).filter(RawClause.section_id == 2962).all()
    lens = []
    for c in clauses:
        if 'deleted' not in c.cleaned:
            doc = nlp(c.no_links)
            for sent in doc.sents:
                print str(sent).strip()
                subject_object_verb(sent)
                tree_string2 = make_string(sent)
                # # print(parse)
                tree(tree_string2)
                # print(str(sent))
                # parse = get_senna_output(str(sent))
                # levels = extract_srl_parse(parse)
                # for l in levels:
                #     if 'A1' in l and 'A0' in l:
                #         print l['A0']
                #         print l['V']
                #         print l['A1']
                # pass
                # print('-'*100)
                # tree_string = ''.join(parse.iloc[:, -1])
                # tree_string = tree_string.replace('*', ' %s ')
                # tree_string = tree_string % tuple(parse.iloc[:, 0].str.strip())
                # print(parse)


def test_sentence(sentence):
    parse = get_senna_output(sentence)
    levels = extract_srl_parse(parse)
    for l in levels:
        print l

def extract_srl_parse(parse):
    srl_parts = parse.iloc[:, 5:-1]
    sentence = parse[0]

    levels = []
    for s in srl_parts:
        tokens = None
        parse_level = {}
        part = srl_parts[s]
        for i in parse.index:
            p = str(part[i]).strip()
            s = str(sentence[i]).strip()
            if p != 'O':
                if p[0] in 'BS':
                    if tokens is not None:
                        tag_name = last_tag[2:]
                        parse_level[tag_name] = tokens
                    tokens = [s]
                else:
                    tokens.append(s)

                last_tag = p
        if tokens:
            tag_name = last_tag[2:]
            parse_level[tag_name] = tokens
        levels.append(parse_level)
    return levels



def make_string(sent):
    string = ''
    state = 0
    for word in sent:
        dep_labels = []
        token = word
        while token.head is not token:
            dep_labels.append(token.dep_)
            token = token.head
        dep_labels.append(token.head.dep_)
        dep_labels.reverse()
        diff = len(dep_labels) - state
        if diff > 0:
            for d in dep_labels[-diff:]:
                string += '({} '.format(d)
        elif diff < 0:
            string += ')' * (-diff)

        string += word.string
        state = len(dep_labels)


    return string + state*')'

def subject_object_verb(sent):
    verb = sent.root.head
    if str(verb) in ['are', 'is']:
        lefts = []
        for l in sent.root.lefts:
            for ll in l.subtree:
                lefts.append(str(ll))
        rights = []
        for r in sent.root.rights:
            for rr in r.subtree:
                rights.append(str(rr))
        print('-'*100)
        print(' '.join(lefts))
        print(verb)
        print(' '.join(rights))

def tree(dep_parse):
    treetok = Tree.fromstring(dep_parse)

    # show parsetree
    cf = CanvasFrame(width=1000, height=450, closeenough=2)
    tc = TreeWidget(cf.canvas(), treetok, draggable=1,
                    node_font=('helvetica', -14, 'bold'),
                    leaf_font=('helvetica', -12, 'italic'),
                    roof_fill='white', roof_color='black',
                    leaf_color='green4', node_color='blue2')
    cf.add_widget(tc,10,10)
    cf.mainloop()

if __name__ == '__main__':
    # analyse_clauses()


    sent = u'When making the personal recommendation or managing his investments, the firm must obtain the necessary information regarding the client\'s knowledge and experience in the investment field relevant to the specific type of designated investment or service.'
    sent = u'If arrangements made by a firm under SYSC1017R to manage conflicts of interest are not sufficient to ensure, with reasonable confidence, that risks of damage to the interests of a client will be prevented, the firm must clearly disclose the general nature and/or sources of conflicts of interest to the client before undertaking business for the client.'
    test_sentence(sent)



    # sent = u'A common platform firm and a management company must establish, implement and maintain an effective conflicts of interest policy that is set out in writing and is appropriate to the size and organisation of the firm and the nature, scale and complexity of its business. Where the common platform firm or the management company is a member of a group, the policy must also take into account any circumstances, of which the firm is or should be aware, which may give rise to a conflict of interest arising as a result of the structure and business activities of other members of the group.'
    # sent = u'The FCA\'s operational objectives are: securing an appropriate degree of protection for consumers, protecting and enhancing the integrity of the UK financial system, and promoting effective competition in the interests of consumers in the markets.'
    # sent = u' A firm must take reasonable steps to ensure that a personal recommendation, or a decision to trade, is suitable for its client.'
    # sent = u'Each of the following is a per se professional client unless and to the extent it is an eligible counterparty or is given a different categorisation under this chapter:'
    # # sent = u'The competent employees rule is the main Handbook requirement relating to the competence of employees. The purpose of this sourcebook is to support the FCA's supervisory function by supplementing the competent employees rule for retail activities.'
    doc = nlp(sent)
    s = doc.sents.next()
    string = make_string(s)
    # subject_object_verb(s)
    tree(string)
    # analyse_clauses()
