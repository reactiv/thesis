from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

__author__ = 'jamesgin'
engine = create_engine('postgres:///jamesgin')
Session = scoped_session(sessionmaker(bind=engine))
session = Session()