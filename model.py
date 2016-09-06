# coding: utf-8
import re
from sqlalchemy import Column, ForeignKey, Integer, Text, text, create_engine
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
import numpy as np

Base = declarative_base()
metadata = Base.metadata

numerals = ['I', 'II', 'III', 'IV']


class ReifiedSubQuestion(object):
    def __init__(self, assertion, truth):
        self.assertion = assertion
        self.truth = truth

class Document(Base):
    __tablename__ = 'document'

    id = Column(Integer, primary_key=True, server_default=text("nextval('tbl_docs_id_seq'::regclass)"))
    name = Column(Text)
    folder_id = Column(ForeignKey(u'document_category.id'))

    folder = relationship(u'DocumentCategory')


class DocumentCategory(Base):
    __tablename__ = 'document_category'

    id = Column(Integer, primary_key=True, server_default=text("nextval('folder_id_seq'::regclass)"))
    name = Column(Text)


class RawClause(Base):
    __tablename__ = 'raw_clause'

    id = Column(Integer, primary_key=True, server_default=text("nextval('tbl_clauses_raw_id_seq'::regclass)"))
    name = Column(Text)
    fca_ref = Column(Text)
    datetime = Column(Text)
    guidance = Column(Text)
    content_html = Column(Text)
    cleaned = Column(Text)
    no_links = Column(Text)
    header = Column(Text, nullable=False)
    url = Column(Text)
    section_id = Column(ForeignKey(u'section.id'))

    section = relationship(u'Section', back_populates='clauses')

class Source(Base):
    __tablename__ = 'source'
    id = Column(Integer, primary_key=True)
    name = Column(Text)
    type = Column(Text)

class Statement(Base):
    __tablename__ = 'statement'
    id = Column(Integer, primary_key=True)
    clause_id = Column(ForeignKey(u'raw_clause.id'))
    head = Column(Text)
    variant1 = Column(Text)
    variant2 = Column(Text)

    def text(self):
        string = self.head
        if self.variant1 is not None:
            string += ' ' + self.variant1
            if self.variant2 is not None:
                string += ' ' + self.variant2
        return string

class StatementPart(Base):
    __tablename__ = 'statement_part'
    id = Column(Integer, primary_key=True)
    clause_id = Column(ForeignKey(u'raw_clause.id'))
    part = Column(Text)
    parent_id = Column(ForeignKey(u'statement_part.id'))
    # parent = relationship('StatementPart', remote_side=[id])
    children = relationship('StatementPart',
                backref=backref('parent', remote_side=[id]))

    def __str__(self):
        return self.part

    def __repr__(self):
        return self.part.encode('ascii', 'ignore')

    def shortest_sentence(self):
        all_children = list(self.yield_children())
        min_index = np.argmin([len(''.join(a)) for a in all_children])
        return all_children[min_index]

    def all_sentences(self):
        return list(self.yield_children())

    def yield_children(self):
        if len(self.children) == 0:
            parts = [self.part]
            par = self.parent
            while par is not None:
                parts.append(par.part)
                par = par.parent
            parts.reverse()
            yield parts
        else:
            for c in self.children:
                for path in c.yield_children():
                    yield path




class Section(Base):
    __tablename__ = 'section'

    id = Column(Integer, primary_key=True, server_default=text("nextval('tbl_section_id_seq'::regclass)"))
    name = Column(Text)
    docpath = Column(Text)
    source_id = Column(ForeignKey(u'source.id'))
    clauses = relationship('RawClause', back_populates='section', cascade='all, delete', order_by='RawClause.id')

class Question(Base):
    __tablename__ = 'question'


    id = Column(Integer, primary_key=True)
    source = Column(Text)
    body = Column(Text)
    num1 = Column(Text)
    num2 = Column(Text)
    num3 = Column(Text)
    num4 = Column(Text)
    ans1 = Column(Text)
    ans2 = Column(Text)
    ans3 = Column(Text)
    ans4 = Column(Text)
    correct = Column(Integer)
    related_clause = Column(ForeignKey(u'raw_clause.id'))
    type = Column(Integer)
    a_type = Column(Text)
    def get_correct(self):
        if self.correct == 0:
            return self.ans1
        elif self.correct == 1:
            return self.ans2
        elif self.correct == 2:
            return self.ans3
        elif self.correct == 3:
            return self.ans4

    def text(self):
        return self.body + '. ' + self.get_correct()

    def all_answers(self):
        return [self.ans1, self.ans2, self.ans3, self.ans4]

    def all_distractors(self):
        return [a for i, a in enumerate(self.all_answers()) if i != self.correct]

    def reify(self):
        if self.num1 is not None:
            # decode the correct answer
            corr = self.get_correct()
            parts = corr.split(' ')
            parts = [numerals.index(p) for p in parts if p not in ['and', '&']]
            subqs = [ReifiedSubQuestion(self.num1, 0 in parts),
                        ReifiedSubQuestion(self.num2, 1 in parts),
                        ReifiedSubQuestion(self.num3, 2 in parts),
                        ReifiedSubQuestion(self.num4, 3 in parts)]
            return subqs

class EntityQuestion(Base):
    __tablename__ = 'entity_question'


    id = Column(Integer, primary_key=True)
    question_id = Column(ForeignKey(u'question.id'))
    source = Column(Text)
    body = Column(Text)
    num1 = Column(Text)
    num2 = Column(Text)
    num3 = Column(Text)
    num4 = Column(Text)
    ans1 = Column(Text)
    ans2 = Column(Text)
    ans3 = Column(Text)
    ans4 = Column(Text)
    correct = Column(Integer)
    related_clause = Column(ForeignKey(u'raw_clause.id'))
    type = Column(Integer)
    a_type = Column(Text)
    def get_correct(self):
        if self.correct == 0:
            return self.ans1
        elif self.correct == 1:
            return self.ans2
        elif self.correct == 2:
            return self.ans3
        elif self.correct == 3:
            return self.ans4

    def text(self):
        return self.body + '. ' + self.get_correct()

    def all_answers(self):
        return [self.ans1, self.ans2, self.ans3, self.ans4]

    def reify(self):
        if self.num1 is not None:
            # decode the correct answer
            corr = self.get_correct()
            parts = corr.split(' ')
            parts = [numerals.index(p) for p in parts if p not in ['and', '&']]
            subqs = [ReifiedSubQuestion(self.num1, 0 in parts),
                        ReifiedSubQuestion(self.num2, 1 in parts),
                        ReifiedSubQuestion(self.num3, 2 in parts),
                        ReifiedSubQuestion(self.num4, 3 in parts)]
            return subqs


if __name__ == '__main__':
    engine = create_engine('postgres:///jamesgin')
    metadata.create_all(engine)