import csv
from io import StringIO
from itertools import chain
import collections
import pickle
import pprint
from random import random, choice
import asciitree
import itertools
from nltk import DependencyGraph
from nltk.corpus import wordnet as wn

__author__ = 'jamesgin'
from features import *
import subprocess as sp
import os
import pandas as pd
from nltk.draw.tree import Tree, TreeWidget
from nltk.draw.util import CanvasFrame
import spacy
import textwrap
memory = Memory('temp', verbose=1)



# nlp = None

max_index = 0
senna_path = '/Users/jamesgin/projects/senna/'

class Token(object):
    def __init__(self, index, head, word, pos, tag, dep):
        self.index = index
        self.word = word
        self.pos = pos
        self.tag = tag
        self.dep = dep
        self.head = head
        self.used = False

    def __repr__(self):
        return '{} {} {}'.format(self.word, self.tag, self.dep)

    def __str__(self):
        return self.__repr__()

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
    show_tree(treetok)


def show_tree(treetok):
    cf = CanvasFrame(width=1000, height=450, closeenough=2)
    tc = TreeWidget(cf.canvas(), treetok, draggable=1,
                    node_font=('helvetica', -14, 'bold'),
                    leaf_font=('helvetica', -12, 'italic'),
                    roof_fill='white', roof_color='black',
                    leaf_color='green4', node_color='blue2')
    cf.add_widget(tc,10,10)
    cf.mainloop()

def get_refs():
    with open('ref_all.txt', 'rb') as ref_file:
        lines = ref_file.readlines()
        refs = [int(l.split(',')[0]) for l in lines]
        return refs

# @memory.cache
def check_parsey_output():
    all_parsed = open('out_all.conll', 'rb').read().decode('ascii', 'ignore').split('\n\n')
    all_sents = []
    # refs = get_refs()
    for a in all_parsed:
        # try:
        d, sent_text = to_dict(a)

        d.keys()[0]
        parts = extract_parts(d)
        wanted = ['nsubj', 'neg', 'aux', 'root']
        sent_dict = {}
        sent_dict['pos'] = d.keys()[0].pos
        sent_dict['text'] = sent_text
        sent_dict['dict'] = d
        for p in parts:
            if p[0] in wanted:
                sent_dict[p[0]] = p[1]

        all_sents.append(sent_dict)
        # except:
        #     pass

    df = pd.DataFrame.from_dict(all_sents)
    # df['refs'] = refs
    return df
        # graph = DependencyGraph(a)
        # show_tree(Tree('hello', Tree('goodbye', Tree('awhhhaw', Tree('loawfoawkf', ['blop'])))))
        # show_tree(dense_tree(graph))

def get_rel(root, rels):
    for r in root:
        if r.dep in rels:
            return root, r

    return None, None

def get_word(root, words):
    for r in root:
        if r.word in words:
            return root, r

    return None, None

def get_all_rels(root, rels):
    for r in root:
        if r.dep in rels:
            yield root, r

def interesting_subject(subject_string):
    pass

def ensure_rule(d):
    tr = asciitree.LeftAligned()
    root_tok = d.keys()[0]
    print(tr(d))
    root, advcl = get_rel(d[d.keys()[0]], ['advcl', 'prep'])
    _, subject = get_rel(d[d.keys()[0]], ['nsubj'])

    if advcl:
        string = get_string_from_root(root, advcl)
        if string.endswith(','):
            string = string[:-1]
        return string + ', which of the following must be satisfied?'
    else:
        root, ccomp = get_rel(d[d.keys()[0]], ['ccomp'])
        if ccomp:
            n_root, nsubj = get_rel(root[ccomp], ['nsubj', 'nsubjpass'])
            if nsubj:
                if nsubj.word == 'it':
                    _, dobj = get_rel(root[ccomp], ['dobj'])
                    if dobj:
                        string = get_string_from_root(root[ccomp], dobj)
                        if subject:
                            subject_string = get_string_from_root(d[d.keys()[0]], subject)
                            return 'Which of the following is true regarding {} that {} {}'.format(string, subject_string, ccomp.word)
                else:
                    # if copular, make it about the relationship
                    _, cop = get_word(n_root, ['is', 'are'])
                    if cop or ccomp.word in ['is', 'are']:
                        string = get_string_from_root(n_root, nsubj)
                        return 'Which of the following is true regarding {}'.format(string)
                    else:
                        string = get_string_from_root(n_root, nsubj, ['WDT'])
                        if string.endswith(','):
                            string = string[:-1]
                        return 'Where ' + string + ', which of the following must be satisfied?'
            # else:
            #     root, nsubjpass = get_rel(root[ccomp], 'nsubjpass')
            #     if nsubjpass:
            #         string = get_string_from_root(root, nsubjpass)
            #         subject_string = get_string_from_root(d[d.keys()[0]], subject)
            #         return 'For {} which of the following must be satisfied regarding {}'.format(subject_string, string)

def _get_children(node, k):
    if len(node[k]) == 0:
        return [k]
    else:
        return [k] + list(chain.from_iterable([_get_children(node[k], c) for c in node[k].keys()]))

def get_string_from_root(root, key, filt=[], mark=False):
    if key is None:
        return ''

    children = [c for c in _get_children(root, key) if c.tag not in filt]
    children = sorted(children, key=lambda c: c.index)
    if mark:
        for c in children:
            c.used = True
    return ' '.join(c.word for c in children)

def get_string_from_root(root, key, filt=[], mark=False):
    if key is None:
        return ''

    children = [c for c in _get_children(root, key) if c.tag not in filt]
    children = sorted(children, key=lambda c: c.index)
    if mark:
        for c in children:
            c.used = True
    return ' '.join(c.word for c in children)


def get_noun_chunks(root, key, filt=[]):

    all_children = _get_children(root, key)
    all_children = sorted(all_children, key=lambda k: k.index)

    def _get_nouns(node, k):
        if k.pos == 'NOUN':
            return [(k, node)] + list(chain.from_iterable([_get_nouns(node[k], c) for c in node[k].keys()]))
        else:
            return [None] + list(chain.from_iterable([_get_nouns(node[k], c) for c in node[k].keys()]))

    children = [c for c in _get_nouns(root, key) if c is not None and c[0].tag not in filt]
    chunks = [(_get_path_to_root(all_children, c[0]), c[1], get_string_from_root(c[1], c[0])) for c in children]

    # for c in chunks:
    #     print(c)
    # children = sorted(children, key=lambda c: c.index)
    return chunks

def get_triples(root, key, filt=[]):

    all_children = _get_children(root, key)
    all_children = sorted(all_children, key=lambda k: k.index)

    def _get_triple(node, k):
        children = node[k]
        triple = {}
        triple['root'] = k
        triple['prep'] = []
        triple['advcl'] = []
        last = None
        conj_pos = None
        conj_type = None
        conj_ignore = False
        conj_memory = []
        for n in children:
            string = get_string_from_root(node[k], n)


            if n.dep in ['nsubj', 'nsubjpass']:
                last = 'subject'
                conj_type = None
                conj_pos = n.pos
                triple['subject'] = string
            elif n.dep in ['dobj', 'iobj']:
                last = 'object'
                conj_type = None
                conj_pos = n.pos
                triple['object'] = string
            elif n.dep in ['csubj', 'csubjpass']:
                triple['csubj'] = string
            elif n.dep in ['advcl']:
                triple[n.dep].append(string)
            elif n.dep in ['neg', 'cop', 'aux', 'auxpass']:
                triple[n.dep] = string
            elif n.dep in ['ccomp', 'xcomp']:
                triple['clause'] = n
            elif n.dep in ['prep']:
                triple['prep'].append(string)
            elif n.dep == 'cc':
                if n.word == 'but':
                    conj_ignore = True
                elif last is not None:
                    conj_ignore = False
                    conj_type = n.word
                    triple[last] = {conj_type: conj_memory + [triple[last]]}
            elif n.dep == 'conj' and not conj_ignore and last is not None:
                if conj_type is None:
                    conj_memory.append(string)
                else:
                    triple[last][conj_type].append(string)

        if not triple['prep']:
            triple.pop('prep')
        if not triple['advcl']:
            triple.pop('advcl')

        if triple:
            obligation = False
            copular = False
            clausal = False
            other = False
            if all(a in triple for a in ['subject', 'root', 'clause']):
                clausal = True
            elif all(a in triple for a in ['cop', 'root', 'subject']):
                copular = True
            elif all(a in triple for a in ['root', 'object']):
                obligation = True
            elif all(a in triple for a in ['subject', 'root', 'prep']):
                other = True


            if k.dep == 'advcl':
                return []
            elif obligation or copular:
                return [triple] + list(chain.from_iterable([_get_triple(node[k], c) for c in node[k].keys()]))
            elif other:
                return [triple] + list(chain.from_iterable([_get_triple(node[k], c) for c in node[k].keys()]))
            elif clausal:
                n = triple['clause']
                triple['clause'] = _get_triple(node[k], n)
                return [triple] + list(chain.from_iterable([_get_triple(node[k], c) for c in node[k].keys() if c != n]))
            else:
                return list(chain.from_iterable([_get_triple(node[k], c) for c in node[k].keys()]))
        else:
            return list(chain.from_iterable([_get_triple(node[k], c) for c in node[k].keys()]))

    children = [c for c in _get_triple(root, key) if c is not None]
    # chunks = [(_get_path_to_root(all_children, c[0]), c[1], get_string_from_root(c[1], c[0])) for c in children]
    pass
    # for c in chunks:
    #     print(c)
    # children = sorted(children, key=lambda c: c.index)
    pprint.pprint(children)
    return children

# class Triple(object):
#     def __init__(self, dict):
#         for n in dict




def get_triples_nested(root, key, filt=[]):
    preprocess_conj(root, key)
    all_children = _get_children(root, key)
    all_children = sorted(all_children, key=lambda k: k.index)
    children = recurse_triples(root, key)
    # pprint.pprint(children)
    analyse_triples(children[1])
    return children

def get_name(dep):
    if dep in ['subject', 'object', 'clause']:
        return dep
    if dep in ['nsubj', 'nsubjpass']:
        return 'subject'
    elif dep in ['dobj', 'iobj', 'pobj']:
        return 'object'
    elif dep in ['csubj', 'csubjpass']:
        return 'csubj'
    elif dep in ['neg', 'cop', 'aux', 'auxpass', 'advcl',
                 'parataxis', 'dep', 'rcmod', 'nn', 'conj', 'num', 'mark']:
        return dep
    elif dep in ['ccomp', 'xcomp']:
        return 'clause'
    elif dep in ['prep']:
        return 'prep'
    else:
        return None

def get_conj_split(child_values):
    conj = []
    mem = None
    for c in child_values:
        v = child_values[c]
        if c.dep == 'conj':
            if conj[-1][0] == 'conj':
                if mem:
                    conj.append(mem)
                else:
                    conj.append('placeholder')
            else:
                # print(c, v)
                conj.append((c, v))
        elif c.dep == 'cc':
            for i, t in enumerate(conj):
                if t == 'placeholder':
                    conj[i] = (c, v)

            conj.append((c, v))
            mem = (c, v)
            # print(c, v)
    conj_list = []
    for i, t in enumerate(conj):
        if i % 2 == 1:
            conj_list.append((conj[i-1][0].word, conj[i]))

    return conj_list


def make_triple(key, child_values, string_from_root):
    triple = {'root': (key, string_from_root)}
    root_name = get_name(key.dep)

    for d, c in child_values:
        name = get_name(d)
        if name:
            elem = c
            if name in triple:
                if type(triple[name]) != list:
                    triple[name] = [triple[name]]
                triple[name].append(elem)
            else:
                triple[name] = elem
    return root_name, triple

def preprocess_conj(root, key):
    for k in root[key]:
        for kk in root[key][k]:
            if kk.dep == 'conj':
                bit = root[key][k][kk].copy()
                root[key][k].pop(kk)
                kk.dep = k.dep
                root[key][kk] = bit



    for c in root[key]:
        preprocess_conj(root[key], c)



def recurse_triples(root, key):
    string = get_string_from_root(root, key)
    if key.word == 'relevant':
        pass
    if not is_valid_triple(key, root[key]):
        string = get_string_from_root(root, key)
        return key.dep, string
    else:
        child_elements = []
        for child in root[key]:
            child_elements.append(recurse_triples(root[key], child))
        return make_triple(key, child_elements, string)

def is_conjoined_phrase(root, key):
    return any(k.dep == 'conj' for k in root[key].keys())

def analyse_triples(triple):
    try:
        if 'root' in triple:
            if 'clause' in triple and 'parataxis' in triple:
                pattern = 'If {} then which of {} must be {}?'
                q = pattern.format(triple['clause']['subject'], triple['clause']['object'], triple['clause']['root'][0].word)
                for a in get_all_answer_strings(triple['parataxis']):
                    print (q, a)
    except:
        pass

def get_answer_string(element):
    if type(element) == unicode:
        return element
    else:
        return element['root'][1]

def get_all_answer_strings(element):
    if type(element) == list:
        return [get_answer_string(e) for e in element]
    else:
        return [get_answer_string(element)]

def is_valid_triple(key, children):
    clausal = copular = obligation = other = False
    parts = [get_name(c.dep) for c in children]
    if all(a in parts for a in ['subject', 'object']):
        clausal = True
    if all(a in parts for a in ['nn', 'rcmod']):
        clausal = True
    elif all(a in parts for a in ['cop', 'subject']):
        copular = True
    # elif all(a in parts for a in ['object']):
    #     obligation = True
    elif all(a in parts for a in ['subject', 'prep']):
        other = True
    elif all(a in parts for a in ['dep']):
        other = True
    elif all(a in parts for a in ['aux']):
        other = True


    return clausal or copular or obligation or other


def _get_path_to_root(tokens, start):
    tok_list = []
    tok = start
    while tok.head != -1:
        tok_list.append(tok)
        tok = tokens[tok.head]
    tok_list.reverse()
    return tok_list

def extract_parts(d):
    root = d.keys()[0]
    root_children = d[root].keys()
    parts = [('root', root.word)]
    print(root)
    for r in root_children:
        parts.append((r.dep, get_string_from_root(d[root], r)))

    # pprint.pprint(parts)

    return parts


def to_dict(sentence):
    """Builds a dictionary representing the parse tree of a sentence.

    Args:
    sentence: Sentence protocol buffer to represent.
    Returns:
    Dictionary mapping tokens to children.
    """
    tokens = sentence.split('\n')
    tokens = [t.split('\t') for t in tokens]
    # token_str = ['%s %s %s' % (token[1], token[4], token[7])
    #          for token in tokens]
    token_str = [Token(i, int(token[6])-1, token[1], token[3], token[4], token[7]) for i, token in enumerate(tokens)]
    sent_text = ' '.join([t.word for t in token_str])
    children = [[] for token in tokens]
    root = -1
    for i in range(0, len(tokens)):
        token = tokens[i]
        if int(token[6]) == 0:
            root = i
        else:
            children[int(token[6]) - 1].append(i)

    def _get_dict(i):
        d = collections.OrderedDict()
        for c in children[i]:
            d[token_str[c]] = _get_dict(c)
        return d

    tree = collections.OrderedDict()
    tree[token_str[root]] = _get_dict(root)
    return tree, sent_text

def get_context(d, root_index):
    _, advcl = get_rel(d, ['advcl'])
    _, prep = get_rel(d, ['prep'])
    string = ''
    if prep and prep.index < root_index:
        string += get_string_from_root(d, prep) + ' '
    if advcl:
        string += get_string_from_root(d, advcl) + ', '

    return string

def get_non_context(d, root_index, context):
    new_dict = collections.OrderedDict()
    for i in d[d.keys()[0]]:
        if (i.dep != 'advcl') and not (i.dep == 'prep' and i.index < root_index):
            new_dict[i] = d[d.keys()[0]][i]

    # do aux, subject inversion
    aux = ''
    subj = ''
    obj = ''
    neg = ''
    for k in new_dict:
        if k.dep == 'aux':
            aux = k
            aux = get_string_from_root(new_dict, aux) + ' '
            new_dict.pop(k)
        elif k.dep in ['nsubj', 'nsubjpass']:
            subj = k
            subj = get_string_from_root(new_dict, subj) + ' '
            new_dict.pop(k)
        elif k.dep == 'dobj':
            obj = k
            obj = get_string_from_root(new_dict, obj) + ' '
            new_dict.pop(k)
        elif k.dep == 'neg':
            neg = k
            neg = get_string_from_root(new_dict, neg) + ' '


    root = d.keys()[0].word + ' '

    other = ' '.join([get_string_from_root(new_dict, k) for k in new_dict])

    return 'In which circumstance {}{}{}{}{}{}?'.format(aux, subj, neg, root, obj, other), context


def provide_rules(d):
    root_tok = d.keys()[0]
    chunks = get_noun_chunks(d, root_tok)
    root = d[root_tok]
    for path, answer in chunks:
        if path[0].dep != 'advcl':
            context = get_context(root)
            tok = path[-1]
            if tok.dep != 'nsubj':
                if not any(t.pos == 'NOUN' for t in path[:-1]):
                    _, aux = get_rel(root, 'aux')
                    if aux:
                        aux = aux.word + ' '
                    else:
                        aux = ''
                    _, neg = get_rel(root, 'neg')
                    if neg:
                        neg = neg.word + ' '
                    else:
                        neg = ''
                    _,subj = get_rel(root, 'nsubj')
                    if subj:
                        subj = get_string_from_root(root, subj) + ' '
                    else:
                        subj = ''


                    if path[0].dep == 'prep':
                        _, obj = get_rel(root, 'dobj')
                        if obj:
                            obj = get_string_from_root(root, obj).lower()
                            if context:
                                if subj == 'it':
                                    string = context + ', which of the following {}{}{}{}'.format(aux, subj.lower(), neg, root_tok.word + ' ')
                                    string += obj + ' ' + path[0].word
                                else:
                                    string = 'Which of the following {}{}{}{}'.format(aux, subj.lower(), neg, root_tok.word + ' ')
                                    string += obj + ' ' + path[0].word

                                    string += ', ' + context.lower()

                                print(string + '?', answer)
                    else:
                        string = 'Which of the following {}{}{}{}'.format(aux, subj.lower(), neg, root_tok.word + ' ')
                        if context:
                            string += ', ' + context.lower()

                        print(string + '?', answer)

def get_numeric(d):
    root_tok = d.keys()[0]
    all_children = _get_children(d, root_tok)
    all_children = sorted(all_children, key=lambda k: k.index)
    for a in all_children:
        if a.dep == 'num':
            path = _get_path_to_root(all_children, a)
            string = get_string_from_root(d[root_tok], path[0])
            string.replace(a.word, 'how many')

def context_question(d, context):
    return get_non_context(d, d.keys()[0].index, context)


def rules_s_o_v(d, simple_subj):

    root_tok = d.keys()[0]
    all_children = _get_children(d, root_tok)
    all_children = sorted(all_children, key=lambda k: k.index)
    # print(' '.join([a.word for a in all_children]))
    root = root_tok.word + ' '
    aux = get_string_or_blank(d[root_tok], ['aux'])
    if aux == '':
        aux = 'does '
    auxpass = get_string_or_blank(d[root_tok], ['auxpass'])

    neg = get_string_or_blank(d[root_tok], ['neg'])
    context = get_context(d[root_tok], root_tok.index)

    nsubj = get_string_or_blank(d[root_tok], ['nsubj', 'nsubjpass'])
    aux_phrase = [aux, nsubj.lower()]
    if root_tok.word in ['is']:
        aux = root
        root = ''

    q = None

    if simple_subj:
        '{}, which of the following {}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, auxpass, root)
    else:
        _, clause = get_rel(d[root_tok], ['ccomp'])
        _, xcomp = get_rel(d[root_tok], ['xcomp'])
        _, dobj = get_rel(d[root_tok], ['dobj'])
        _, prep_1 = get_rel(d[root_tok], ['prep'])
        if clause:
            c_nsubj = get_string_or_blank(d[root_tok][clause], ['nsubj', 'nsubjpass', 'csubj'])
            if c_nsubj == '':
                clause_neg = get_word_or_blank(d[root_tok][clause], 'neg')
                q = ('{} which of the following {}{}{}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, auxpass, root, clause_neg, clause.word))
            elif c_nsubj != 'it ':
                q = ('{} which of the following {}{}{}{}{} regarding {}?'.format(context, aux, nsubj.lower(), neg, auxpass, root, c_nsubj))
            else:
                p_root, prep = get_rel(d[root_tok][clause], 'prep')
                if prep:
                    prep_string = get_string_from_root(p_root, prep, mark=True)
                    q = ('{} which of the following {}{}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, auxpass, root, prep_string))
                else:
                    pass
        elif xcomp and root_tok.word != 'is':
            aux2 = get_string_or_blank(d[root_tok][xcomp], 'aux')
            prep = get_string_or_blank(d[root_tok][xcomp], 'prep', True)
            dobj2 = get_word_or_blank(d[root_tok][xcomp], 'dobj')
            subj2 = get_string_or_blank(d[root_tok][xcomp], 'nsubj')
            # if prep == '':
            #     prep = 'about'
            q = '{} which of the following {}{}{}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, root, subj2, aux2, xcomp.word)
            # q = (('{} which of the following {}{}{}{}{}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, auxpass, root, aux2, xcomp.word + ' ', dobj2, prep)))
        elif dobj:
            # find prep
            prep = get_word_or_blank(d[root_tok][dobj], 'prep')
            # prep = get_word_or_blank(d[root_tok], 'prep')
            if prep != '':
                q = ('{} which of the following {}{}{}{}{}a {} {}?'.format(context, aux, nsubj.lower(), neg, auxpass, root, dobj.word, prep))
            else:
                q = ('{} which of the following {}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, auxpass, root))
        else:
            q =('{} which of the following {}{}{}{}{}?'.format(context, aux, nsubj.lower(), neg, auxpass, root))
    print(q)
    a = get_answer(d)

    if q is None:
        pass

    if q is not None and a != '.':
        if context != '':
            return [(q,a), context_question(d, context)]
        else:
            return [(q,a)]

def chunk_approach(d):
    chunks = get_noun_chunks(d, d.keys()[0])
    all_children = _get_children(d, d.keys()[0])
    all_children = sorted(all_children, key=lambda k: k.index)
    string = ' '.join([a.word for a in all_children])
    chunks = filter_chunks(chunks)
    qa_pairs = []
    for c in chunks:
        dist = get_distractor(c[0][-1], c[1])
        if len(dist) > 0:
            q = (string.replace(c[2], get_cloze_phrase(c))[:-1] + '?')
            # for d in dist:
            #     qa_pairs.append((q, d, False))

            qa_pairs.append((q, c[2], True))

    return qa_pairs

num_dist = [10000, 5, 2, 1000, 45000, 75000, 12]
num_word_dist = ['five', 'ten', 'twelve', 'six', 'four', 'three']
# num_word_dist = ['five', 'six']
aux_modals = ['can', 'should', 'must', 'will', 'cannot', 'should not']

def get_distractor(key, chunk):
    all_units = _get_children(chunk, key)
    all_units = sorted(all_units, key= lambda k: k.index)
    string = ' '.join([a.word for a in all_units])
    # print(string)
    subs = []
    _, num = get_rel(chunk[chunk.keys()[0]], ['num'])
    if num:
        for num in [a for a in all_units if a.dep == 'num']:
            if num.word.isdigit():
                other = [n for n in num_dist if n != int(num.word)]
                rep = choice(other)
                subs.append((num.index, str(rep)))
            else:
                other = [n for n in num_word_dist if n != num.word]
                rep = choice(other)
                subs.append((num.index, rep))
    else:
        jj = [a for a in all_units if a.tag == 'JJ']
        if len(jj) > 0:
            for j in jj:
                anto = list(get_antonyms(j.word))
                for a in anto:
                    subs.append((j.index, a))
            pass
        jjr = [a for a in all_units if a.tag == 'JJR']
        if len(jjr) > 0:
            for j in jjr:
                anto = list(get_antonyms(j.word))
                for a in anto:
                    subs.append((j.index, a))
        jjs = [a for a in all_units if a.tag == 'JJS']
        if len(jjs) > 0:
            for j in jjs:
                anto = list(get_antonyms(j.word))
                for a in anto:
                    subs.append((j.index, a))

    ds = []
    for s in subs:
        distractor = ''
        for a in all_units:
            if a.index == s[0]:
                distractor += s[1] + ' '
            else:
                distractor += a.word + ' '
        # print(distractor)
        ds.append(distractor)
    return ds

def get_antonyms(word):
    synsets = wn.synsets(word)
    anto = set()
    for s in synsets:
        lemmas = s.lemmas()
        for l in lemmas:
            antonyms = l.antonyms()
            for a in antonyms:
                anto.add(a.name())
    return anto



def get_cloze_phrase(chunk):
    _, num_rel = get_rel(chunk[1][chunk[0][-1]], 'num')
    rnd = random() > 0.5
    if num_rel is not None:
        if any([p in chunk[2] for p in ['year', 'day', 'month']]) and rnd:
            return 'how long'
        else:
            return chunk[2].replace(num_rel.word, 'how many')
    else:
        return 'which of the following'

def filter_chunks(chunks):
    return [c for c in chunks if _filter_chunk(c)]

def _filter_chunk(chunk):
    if chunk[0][-1].dep == 'nsubj' and chunk[2].lower() in ['a firm', 'the firm']:
        return False
    else:
        return True

def get_answer(d):
    root = d
    key = root.keys()[0]
    all_children = _get_children(root, key)
    all_children = sorted(all_children, key=lambda k: k.index)
    max_index = 0
    for a in all_children:
        if a.used:
            max_index = a.index

    max_index = max(max_index, key.index)

    answer = ' '.join([a.word for a in all_children if a.index > max_index])
    return answer

def get_string_or_blank(root, dep, last=False):
    rels = get_all_rels(root, dep)
    strings = []
    for _, dep_key in rels:
        if dep_key:
            strings.append(get_string_from_root(root, dep_key, mark=True) + ' ')
    if strings:
        if last:
            return strings[-1]
        else:
            return strings[0]
    else:
        return ''

def get_word_or_blank(root, dep, last=False):
    rels = get_all_rels(root, dep)
    strings = []
    for _, dep_key in rels:
        if dep_key:
            dep_key.used = True
            # print(dep_key)
            strings.append(dep_key.word + ' ')
    if strings:
        if last:
            return strings[-1]
        else:
            return strings[0]
    else:
        return ''

def describe_questions():
    nlp = spacy.load('en')
    questions = session.query(Question).all()
    wh_words = ['which', 'who', 'when', 'where', 'what', 'how']
    tups = []
    for q in questions:
        print q.body
        pos = []
        a_type = 'n'
        for a in q.all_answers():
            doc = nlp(a)
            sent = doc.sents.next()
            # print(a, sent.root.pos_)
            pos.append(sent.root.pos_)
        if pos[q.correct] == 'VERB':
            a_type = 'v'
        q.a_type = a_type




        w_type = ' '.join([w for w in wh_words if w in q.body.lower()])
        tups.append((q.body, w_type, q.all_answers()[q.correct], a_type))

    session.commit()
    df = pd.DataFrame(tups, columns=['body', 'wh-word', 'corr_answer', 'a_type'])
    pass


if __name__ == '__main__':
    # describe_questions()
    # analyse_questions()
    output = check_parsey_output()
    print(output['root'].value_counts())
    ensures = output[output['pos'] == 'VERB']
    # ensures = output
    qs = []
    errs = 0
    for idx, row in ensures.iterrows():
        d = row['dict']
        # ref = row['refs']
        tr = asciitree.LeftAligned()
        # print(get_string_from_root(d, d.keys()[0]))
        # print(tr(d))
        # rules_s_o_v(d, True)
        # get_numeric(d)
        get_triples_nested(d, d.keys()[0])
        # new_qs = rules_s_o_v(d, False)
        # if new_qs is not None:
        #     for n in new_qs:
        #         # print n
        #         pass
        #         # qs.append((ref,) + n)
        # else:
        #     errs += 1
        # print(len(qs))
        # print(errs)
    # pd.DataFrame(qs).to_csv('generated_qs.csv')

