from prefect import flow
from db_tasks import fetch_data, process_data

db_path = 'local.db'

@flow(name='SQLite Example')
def work_with_db():
    rows = fetch_data(db_path)
    process_data(rows)

if __name__ == "__main__":
    work_with_db()