import os
from dotenv import load_dotenv
from psycopg2 import pool,sql

load_dotenv()

db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    user=os.getenv('PG_USER'),
    password=os.getenv('PG_PASSWORD'),
    host=os.getenv('PG_HOST'),
    port=os.getenv('PG_POST'),
    database=os.getenv('PG_DB')
)



def add_job(job_id: str, status: str, image_url: str = None):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            if image_url is not None:
                query = sql.SQL("""
                    INSERT INTO vton_jobs (id, status, vton_image_url)
                    VALUES (%s, %s, %s)
                """)
                cur.execute(query, (job_id, status, image_url))
            else:
                query = sql.SQL("""
                    INSERT INTO vton_jobs (id, status)
                    VALUES (%s, %s)
                """)
                cur.execute(query, (job_id, status))
            conn.commit()
            print(f"Added job {job_id} with status {status} and image_url {image_url}")
    except Exception as e:
        conn.rollback()
        print("Error adding job:", e)
        raise
    finally:
        db_pool.putconn(conn)

