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



def update_job_status(job_id: str, status: str, image_url: str = None, update: bool = False):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            if update:
                if image_url is not None:
                    query = sql.SQL("""
                        UPDATE vton_jobs
                        SET status = %s, vton_image_url = %s
                        WHERE id = %s
                    """)
                    cur.execute(query, (status, image_url, job_id))
                else:
                    query = sql.SQL("""
                        UPDATE vton_jobs
                        SET status = %s
                        WHERE id = %s
                    """)
                    cur.execute(query, (status, job_id))
                print(f"Updated job {job_id} with status {status} and image_url {image_url}")
            else:
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
                print(f"Added job {job_id} with status {status} and image_url {image_url}")
            conn.commit()
    except Exception as e:
        conn.rollback()
        print("Error adding/updating job:", e)
        raise
    finally:
        db_pool.putconn(conn)
