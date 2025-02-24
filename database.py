# Import module
import sqlite3

# Connecting to sqlite
conn = sqlite3.connect('../voice/modell/chats.db')

# Creating a cursor object using the cursor() method
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS ASSISTANT (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    QUERY TEXT NOT NULL,
    DATE_TIME DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()

def add_data(query):
    table = "INSERT INTO ASSISTANT(QUERY, DATE_TIME) VALUES (?, datetime('now', 'localtime'))"
    cursor.execute(table, (query,))
    conn.commit()
    return True

def get_data():
    data = cursor.execute('SELECT * FROM ASSISTANT')
    table_head = []
    for column in data.description:
        table_head.append(column[0])
    print("{:<14} {:<79} {:<20}".format(table_head[0], table_head[1], table_head[2]))
    print()
    for row in data:
        print("{:<14} {:<79} {:<20}".format(row[0], row[1], row[2]))
    conn.commit()