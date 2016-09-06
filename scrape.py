__author__ = 'jamesgin'
import requests
from bs4 import BeautifulSoup
from model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import pandas as pd
from gevent.pool import Pool
from multiprocessing import Pool as pPool

engine = create_engine('postgres:///jamesgin')
Session = scoped_session(sessionmaker(bind=engine))
session = Session()
urlroot = 'https://www.handbook.fca.org.uk'

done_set = set(s.docpath for s in session.query(Section).order_by(Section.id))

def get_page(params):
    url, name = params
    urlparts = url.split('/')
    docpath = '/'.join(urlparts[2:])[:-5]

    if docpath not in done_set:
        req = requests.get(urlroot + url)
        page_html = req.text
        # with open('temp.html', 'wb') as myfile:
        #     myfile.write(req.text)

        # with open('temp.html', 'rb') as myfile:
        #     page_html = myfile.read()
        s = Section(name=name, docpath=docpath)

        if req.status_code == 200:
            bs = BeautifulSoup(page_html, 'lxml')
            sections = bs.find_all('section')
            header = None
            for a in range(0, len(sections)):
                # check if we are at header
                if sections[a].select('h1') or sections[a].select('h2'):
                    header = sections[a].text
                else:
                    # check if overriding header
                    if sections[a].select('header'):
                        header = sections[a].select('header')[0].text
                    content = sections[a]
                    if content.select('div.details'):
                        details = content.select('div.details')[0]
                        spans = details.select('span')
                        title = spans[0].text
                        if len(spans) == 3:
                            guidancetype = spans[1].text
                            time = spans[2].text
                        else:
                            guidancetype = None
                            time = spans[1].text

                        ref_name = None
                        if details.select('a'):
                            ref_name = details.select('a')[0].get('name')
                        html = str(content.select('div.section-content')[0])
                        if header is None:
                            header = name
                        clause = RawClause(name=title, fca_ref=ref_name, datetime=time, guidance=guidancetype,
                                  content_html=html, header=header)
                        s.clauses.append(clause)
            if len(s.clauses) > 0:
                session.add(s)
                session.commit()
                print 'Done {}'.format(name)
                return name
            else:
                print 'No data {}'.format(name)
                pass
    else:
        'Already Done {}'.format(name)

def scrape_fca():
    files = pd.read_csv('hbfiles2.csv', header=None)
    arglist = []
    for f in files.index:
        row = files.loc[f]
        arglist.append((row[1], row[0]))

    # arglist = arglist[21:50]
    p = Pool(8)
    done = p.map(get_page, arglist)
    df = pd.DataFrame(pd.Series(done))
    df.to_csv('done.csv', header=None)
    #
    Session.remove()

def scrape_fitch(files):
    all_qs = set([s[0] for s in session.query(Question.body).all()])

    for s in files:
        soup = BeautifulSoup(open(s, 'rb'), 'lxml')
        qs = soup.find_all('div', class_='test-question')
        for q in qs:
            q_text = q.select('span.test-question-text')[0].text


            # print q_text
            answers = []
            numerals = q.findAll('div', class_='test-response-option')
            num_opts = [n.text for n in numerals]
            if not num_opts:
                num_opts = [None for i in range(4)]
            # print num_opts
            for a in range(1,5):
                answers.append(q.findAll('div', class_='test-response-passive')[0].find(id='test-response-{}'.format(a)).select('div.test-response-answer')[0].text)
            # print answers_1
            correct = 'ABCD'.index(q.find('div', class_='test-answer-actual').select('strong')[0].text)
            # print correct
            question = Question(body=q_text, ans1=answers[0], ans2=answers[1], ans3=answers[2], ans4=answers[3],
                                num1=num_opts[0], num2=num_opts[1], num3=num_opts[2], num4=num_opts[3], correct=correct)

            if q_text in all_qs:
                # print 'DUPLICATE FOUND'
                # print q_text
                pass
            else:
                print 'NEW FOUND'
                session.add(question)
            all_qs.add(q_text)

    session.commit()
    print len(all_qs)

def get_fitch_qs(filename):
    soup = BeautifulSoup(open(filename, 'rb'), 'lxml')
    qs = soup.select

def scrape_glossary():
    gloss = session.query(Section).filter(Section.id == 5174).first()
    root = 'https://www.handbook.fca.org.uk/handbook/glossary/?page={}'
    nones = set([c.name for c in gloss.clauses if c.content_html == 'None'])
    for page in range(1, 159):
        req = requests.get(root.format(page))
        main_soup = BeautifulSoup(req.text, 'lxml')
        defs = main_soup.find_all('div', class_='expand')
        link_names = [(d.select('a')[0]['href'], d.text.strip()) for d in defs if d.text.strip() in nones]
        results = map(get_gloss_link, link_names)
        for r in results:
            edit_clause(gloss, r)
        session.commit()
        print page

def fix_glossary_links():
    clauses = session.query(RawClause).filter((RawClause.section_id == 5174)
                                            & (RawClause.url == None))
    root = 'https://www.handbook.fca.org.uk/handbook/glossary/?page={}'
    nones = {c.name: c for c in clauses}
    for page in range(1, 159):
        req = requests.get(root.format(page))
        main_soup = BeautifulSoup(req.text, 'lxml')
        defs = main_soup.find_all('div', class_='expand')
        link_names = [(d.select('a')[0]['href'], d.text.strip()) for d in defs if d.text.strip() in nones]
        for l in link_names:
            if l[1] in nones:
                print l
                nones[l[1]].url = l[0]
        session.commit()
        print page

def edit_clause(gloss, clause):
    for i in gloss.clauses:
        if i.name == clause.name:
            i.content_html = clause.content_html


def get_gloss_link(link_name_tuple):
    link, name = link_name_tuple
    req = requests.get('https://www.handbook.fca.org.uk' + link)
    s = BeautifulSoup(req.text, 'lxml')
    desc = s.find('div', id='desc')
    if desc is None:
        print 'None alert - {}'.format(name)
    else:
        print 'Processed - {}'.format(name)
    return RawClause(name=name, fca_ref='gloss', header=name, content_html=str(desc))

def get_fsma(url, title):
    root = 'http://www.legislation.gov.uk/'
    req = requests.get(root + url)
    sect = Section(name=title, docpath=url)
    soup = BeautifulSoup(req.text, 'lxml')
    contents = soup.find('div',id='viewLegContents')
    spans = contents.find_all('span')
    title = None
    for i in range(len(spans)):
        try:
            s = spans[i]
            id = s.get('id')
            # print(id)
            cls = s.get('class')
            # print cls
            if cls is not None and 'GroupTitle' in ''.join(cls):
                legadd = s.select('span.LegAddition')
                if legadd:
                    title = s.select('span.LegAddition')[0].text
                else:
                    title = s.contents[0].strip()
            if id is not None and len(id) > 7 and id[:7] == 'section':
                if title is not None:
                    section = id[8:-2]
                    clause = title
                    # print(section, clause)
                    sect.clauses.append(RawClause(name=section, fca_ref=None, datetime=None, guidance=None,
                                  content_html=clause, header=section))
                    title = None
                section = id[8:]
                for j in range(5):
                    sp = spans[i + j]
                    if 'LegRHS' in sp.get('class'):
                        clause = sp.text
                        break

                # print(section, clause)
                sect.clauses.append(RawClause(name=section, fca_ref=None, datetime=None, guidance=None,
                                  content_html=clause, header=section))
                next_p = s.find_next('p')
                if next_p.get('class') is not None and 'LegRHS' in next_p.get('class'):
                    end_name = section[:-1] + 'end'
                    sect.clauses.append(RawClause(end_name, fca_ref=None, datetime=None, guidance=None,
                                  content_html=next_p.text, header=end_name))
        except:
            pass

    print(sect.name)
    session.add(sect)
    session.commit()

def get_fsma_index(index_url):
    req = requests.get(index_url)
    soup = BeautifulSoup(req.text, 'lxml')
    links = soup.find_all('p', class_='LegContentsNo')
    for s in links:
        a = s.find('a')
        yield (a['href'], a.text)

def get_all_fsma():
    all_sections = list(get_fsma_index('http://www.legislation.gov.uk/ukpga/2000/8/contents?view=plain'))
    for s in all_sections:
        get_fsma(*s)

def get_all_aml():
    all_sections = list(get_fsma_index('http://www.legislation.gov.uk/uksi/2007/2157/contents/made?view=plain'))
    for s in all_sections:
        get_fsma(*s)
def get_roots():
    pass

if __name__ == '__main__':
    # pass
    # get_all_aml()
    # scrape_glossary()
    # fix_glossary_links()
    # scrape_fca()
    scrape_fitch(['cobsmed.htm'])
    # scrape_fitch(['assoc1.htm','assoc2.htm','assoc3.htm','assoc4.htm',
    #               'fsma1.htm', 'fsma2.htm', 'fsma3.htm', 'fsma4.htm', 'source4.htm'])