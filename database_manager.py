import mysql.connector

import mysql.connector

class DatabaseManager:
    def __init__(self, host, user, password, database):
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.connection.cursor()
            print("Connected to the database successfully.")
        except mysql.connector.Error as err:
            print(f"Error during database connection: {err}")
            raise  # Re-raise the exception after printing the error message

    def execute_query(self, query, values=None):
        try:
            if values:
                self.cursor.execute(query, values)
            else:
                self.cursor.execute(query)
            self.connection.commit()
        except mysql.connector.Error as err:
            print(f"Error executing query: {err}")
            raise

    def fetch_one(self, query):
        try:
            self.cursor.execute(query)
            return self.cursor.fetchone()
        except mysql.connector.Error as err:
            print(f"Error fetching one result: {err}")
            raise

    def fetch_all(self, query):
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return result
        except mysql.connector.Error as err:
            print(f"Error fetching all results: {err}")
            raise

