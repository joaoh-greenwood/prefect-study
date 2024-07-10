import sqlite3

conn = sqlite3.connect('local.db')

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL
)
''')

cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)', ('Paul', 80))
cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)', ('Ringo', 70))

conn.commit()
conn.close()