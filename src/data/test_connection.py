from databricks_client import get_connection


def test_connection():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                current_catalog() AS catalog,
                current_schema() AS schema
        """)

        result = cursor.fetchone()

        print("Databricks connection successful!")
        print(f"Catalog: {result[0]}")
        print(f"Schema: {result[1]}")

    finally:
        connection.close()


if __name__ == "__main__":
    test_connection()