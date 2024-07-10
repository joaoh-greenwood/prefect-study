import sqlite3
from prefect import task

@task
def fetch_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    rows = cursor.fetchall()
    conn.close()
    return rows

@task
def process_data(rows):
    for row in rows:
        print(f'User ID: {row[0]}, Name: {row[1]}, Age: {row[2]}')