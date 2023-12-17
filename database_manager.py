import mysql.connector

class DatabaseManager:
    def __init__(self, host, user, password, database):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.connection.cursor()

    def execute_query(self, query, values=None):
        if values:
            self.cursor.execute(query, values)
        else:
            self.cursor.execute(query)
        self.connection.commit()

    def fetch_one(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchone()

    def fetch_all(self, query):
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return result
