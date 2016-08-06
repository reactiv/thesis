import re
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from model import RawClause, Section, Source, Question, Statement

__author__ = 'jamesgin'
engine = create_engine('postgres:///jamesgin')
Session = scoped_session(sessionmaker(bind=engine))
session = Session()

def get_section_ref_ids():
    return dict(session.query(Section.docpath, Section.id).all())

def get_gloss_tags():
    return dict(session.query(RawClause.url, RawClause.id).filter(RawClause.url != None).all())

def get_gloss_text():
    names = dict(session.query(RawClause.id, RawClause.name).filter(RawClause.url != None).all())
    # for n in names:
    #     names[n] = names[n].replace(' ', '')
    return names

gloss_tags = get_gloss_tags()
gloss_names = get_gloss_text()
ref_tags = get_section_ref_ids()

def get_clauses():
    # get clauses which have html
    return session.query(RawClause).filter(RawClause.content_html.like('<%'))


def basic_clean(clause):
    #load and just get raw text
    soup = BeautifulSoup(clause.content_html, 'lxml')
    soup = replace_all_hrefs(soup)
    return soup.text

def basic_clean_no_links(clause):
    #load and just get raw text
    soup = BeautifulSoup(clause.content_html, 'lxml')
    soup = replace_all_hrefs_text_only(soup)
    text = soup.text
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^)]*\]', '', text)
    text = text.replace('\r\n', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'[0-9]', '', text)
    return text

def replace_all_hrefs(soup):
    for a in soup.find_all('a'):
        if a.has_attr('href'):
            possible_id = replace_href(a)

            if possible_id:
                if possible_id in gloss_names:
                    a.replace_with('[{}:{}]'.format(gloss_names[possible_id].encode('ascii', 'ignore'), possible_id))
                else:
                    a.replace_with('[{}:{}]'.format(a.text.encode('ascii', 'ignore'), possible_id))
            else:
                a.replace_with('[{}]'.format(a.text.encode('ascii', 'ignore')))
    return soup

def replace_all_hrefs_text_only(soup):
    for a in soup.find_all('a'):
        if '.' in a.text:
            a.replace_with('Section')
    return soup

def amplify(clause):
    soup = BeautifulSoup(clause.content_html, 'lxml')
    statements = None
    soup = replace_all_hrefs(soup)
    div = soup.find('div', class_='section-content').find('div')
    if div is not None:
        ols = div.find_all('ol', recursive=False)

        if ols:
            for o in ols:
                if o.previous_sibling is not None:
                    statements = extract_lis(o, o.previous_sibling, True)
    else:
        pass

    if statements is None:
        statements = [(soup.text, None, None)]

    return statements

def extract_lis(ol, head, recurse=False):
    head_text = None
    if hasattr(head, 'text'):
        head_text = head.text
    else:
        sib = head
        while not hasattr(sib, 'text'):
            if sib is None:
                head_text = str(head)
                break
            else:
                sib = sib.previous_sibling
        if head_text is None:
            head_text = sib.text

    head_text = clean_sentence(head_text)

    lis = ol.find_all('li', recursive=False)
    statements = []
    for li in lis:
        if recurse:
            ols = li.find('ol', recursive=False)
            if ols:
                sub_statements = extract_lis(ols, ols.previous_sibling, False)
                for s in sub_statements:
                    # print(s)
                    # print(sub_statements)
                    if type(sub_statements) == list:
                        pass
                    statements.append([head_text, clean_sentence(s)])
            else:
                statements.append([head_text, clean_sentence(li.text)])
        else:
            statements.append([head_text, clean_sentence(li.text)])


    return statements

def amplify_all():
    clauses = session.query(RawClause).join(Section).join(Source).filter(Source.type == 'fca_sourcebook')
    n = float(clauses.count())
    i = 0
    for c in clauses:
        statements = amplify(c)
        for s in statements:
            filt = [st for st in s if st is not None and st.strip() != '']
            filt = filt + [None] * 3
            # session.add(Statement(clause_id=c.id, head=filt[0], variant1=filt[1], variant2=filt[2]))

        i += 1
        if i % 100 == 0:
            # session.commit()
            print(i/n)

def edit_group(group):
    txt = group.group(1)
    if ':' in txt:
        return txt[:txt.index(':')]
    else:
        return txt


def clean_sentence(sent_text):
    """
    Remove numerical references, brackets etc.
    :param sent_text:
    :return:
    """
    print(sent_text)
    txt = re.sub(r'\[[0-9]+\]', '', sent_text)
    txt = re.sub(r'\([^)]*\)', '', txt)
    txt = re.sub(r'\[(.*?)\]', edit_group, txt)
    txt = txt.strip()
    return txt


def replace_href(tag):
    href = tag['href']
    id = None
    if href[:19] == '/handbook/glossary/':
        # is a glossary tag
        if href in gloss_tags:
            id = gloss_tags[href]
    else:
        if href[:5] == '/hand':
            hashmark = href.find('#')
            docpath = href[10:hashmark-1]
            if '.' in docpath:
                docpath = docpath[:docpath.find('.')]
            ref = href[hashmark+1:]
            if docpath in ref_tags:
                sect_id = ref_tags[docpath]
                found = session.query(RawClause).filter((RawClause.section_id == sect_id)
                                                       & (RawClause.fca_ref == ref)).first()
                if found is not None:
                    id = found.id
    return id

def fix_sections_legislation(docpath, sourcename):
    source = session.query(Source).filter(Source.name == sourcename).first()
    if source is None:
        source = Source(name=sourcename, type='legislation')
        session.add(source)
        session.commit()

    # conjoin the subitems of clauses to their parent
    # create sections from the parents
    to_fix = session.query(Section).filter(Section.docpath.like(docpath))
    nc = None
    ns = None
    for s in to_fix:
        if s.name != 'blp':
            for c in s.clauses:
                if '-' not in c.name: # new section
                    if ns is not None:
                        ns.clauses.append(nc)
                        session.add(ns)
                        nc = None

                    ns = Section(name=c.content_html, docpath=s.docpath, source_id=source.id)
                elif c.name.count('-') == 1: # new clause
                    if nc is not None:
                        ns.clauses.append(nc)

                    text = c.content_html
                    nc = RawClause(header=c.header, content_html=text, cleaned=text)
                else:
                    # continuation clause
                    text = '\n#{}#'.format(c.header) + c.content_html
                    nc.content_html += text
                    nc.cleaned += text
    session.commit()




def get_section_name(section):
    if '/' in section.docpath:
        return section.docpath[:section.docpath.find('/')]
    else:
        return section.docpath

def generate_sources():
    book_sections = session.query(Section).filter(Section.docpath != "")
    names = set()
    for s in book_sections:
        name = get_section_name(s)
        if name != '' and name not in names:
            print(name)
            session.add(Source(name=name, type='fca_sourcebook'))
            names.add(name)

    session.commit()

def align_sourcebook_sources():
    book_sections = session.query(Section).filter(Section.docpath != "")
    sourcebooks = session.query(Source).filter(Source.type == 'fca_sourcebook')
    name_dict = {s.name: s.id for s in sourcebooks}
    for s in book_sections:
        name = get_section_name(s)
        if name != '':
            s.source_id = name_dict[name]

    session.commit()

def basic_clean_all():
    clauses = get_clauses()
    n = float(clauses.count())
    i = 0
    for c in clauses:
        c.cleaned = basic_clean(c)
        i += 1
        if i % 100 == 0:
            session.commit()
            print(i / n)

def basic_clean_no_links_all():
    clauses = get_clauses()
    n = float(clauses.count())
    i = 0
    for c in clauses:
        c.no_links = basic_clean_no_links(c)
        i += 1
        if i % 100 == 0:
            session.commit()
            print(i / n)

def fix_duplicate_questions():
    questions = session.query(Question).order_by(Question.id)
    body_set = set()
    count = 0
    for q in questions:
        if q.body in body_set:
            session.delete(q)
            count += 1
        else:
            body_set.add(q.body)

    session.commit()
    print('{} deleted'.format(count))


if __name__ == '__main__':
    amplify_all()
    # basic_clean_no_links_all()
    # clause = session.query(RawClause).filter(RawClause.id == 56438).first()
    #
    # amplify_all()
