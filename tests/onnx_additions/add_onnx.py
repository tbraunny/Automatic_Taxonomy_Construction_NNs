# 1. connect to db
# 2. extract layers using tables.py, or is it extract the jsonb?
# 3. send context to LLM?

import sqlalchemy as db
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

def connect_db():
    engine = db.create_engine('postgresql://postgres:postgres@localhost:2222/graphdb')
    Session = sessionmaker(bind=engine)
    session = Session()

    with engine.connect() as conn:
        result = conn.execute(text('select graph from model'))
        print(result)

def extract_jsonb():
    pass


if __name__ == '__main__':
    connect_db()