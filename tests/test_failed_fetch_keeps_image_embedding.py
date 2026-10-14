"""A failed image fetch must not null an existing image_embedding.

Logic recW2usIq5jMS1CsH. The Postgres section runs the worker's upsert SQL
(halfvec casts rewritten to text) against a scratch database.
"""
from __future__ import annotations

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

import worker

OID_KEEP = "11111111-1111-4111-8111-111111111111"
OID_OLD = "22222222-2222-4222-8222-222222222222"
OID_OK = "33333333-3333-4333-8333-333333333333"
OID_NONE = "44444444-4444-4444-8444-444444444444"
OID_NEW = "55555555-5555-4555-8555-555555555555"
KEPT = "kept-image-vector"
FETCH_URL = "https://cdn.example.com/catalog/item.jpg"
TEST_DSN = os.environ.get("TEST_DATABASE_URL", "dbname=embed_upsert_test")


def _text_sql(sql: str) -> str:
    """Run the worker statement without pgvector. Only the cast may change."""
    out = sql.replace("::halfvec(768)", "::text")
    if out.replace("::text", "::halfvec(768)") != sql:
        raise AssertionError("halfvec cast rewrite was not exact")
    return out


def _swap_image_assignment(sql: str, assignment: str) -> str:
    prefix, rest = sql.split("image_embedding   = ", 1)
    _case, suffix = rest.split(", embedded_at", 1)
    return prefix + "image_embedding   = " + assignment + ", embedded_at" + suffix


class _Cursor:
    def __init__(self):
        self.sql = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def executemany(self, sql, params):
        self.sql = sql
        self.params = list(params)


class _Conn:
    def __init__(self):
        self.cur = _Cursor()
        self.commits = 0

    def cursor(self):
        return self.cur

    def transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        self.commits += 1


class _Resp:
    def raise_for_status(self):
        return None


class _Session:
    def __init__(self):
        self.posts = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return _Resp()


class _Models:
    def get_text(self):
        return self

    def get_vision(self):
        return self

    def embed(self, texts):
        return [np.zeros(worker.DIM, dtype=np.float32)]


def _result(oid, *, image=None, attempted=False, src=None, text=None):
    return {
        "id": oid,
        "want_hash": "hash-1",
        "text_vec": text if text is not None else np.zeros(4, dtype=np.float32),
        "image_vec": image,
        "image_src": src,
        "image_attempted": attempted,
    }


class UpsertSqlTests(unittest.TestCase):
    def test_conflict_update_keeps_stored_vector_on_failed_fetch(self):
        sql = worker.offering_embedding_upsert_sql("public.offering_embedding")
        self.assertIn("INSERT INTO public.offering_embedding ", sql)
        self.assertNotIn("image_embedding   = EXCLUDED.image_embedding", sql)
        self.assertIn(
            "WHEN EXCLUDED.image_embedding IS NOT NULL THEN EXCLUDED.image_embedding "
            "WHEN EXCLUDED.image_embedded_at IS NOT NULL THEN offering_embedding.image_embedding "
            "ELSE NULL END",
            sql,
        )
        self.assertNotIn("public.offering_embedding.image_embedding", sql)

    def test_postgres_write_uses_that_sql_and_sends_null_on_failure(self):
        src = worker.PostgresSource.__new__(worker.PostgresSource)
        src.conn = _Conn()
        failed = _result(OID_KEEP, attempted=True)
        ok = _result(OID_OK, image=np.ones(4, dtype=np.float32), attempted=True, src=FETCH_URL)
        src.write([failed, ok])
        self.assertEqual(src.conn.cur.sql, worker.offering_embedding_upsert_sql())
        self.assertEqual(src.conn.commits, 1)
        fail_params, ok_params = src.conn.cur.params
        self.assertEqual(fail_params[0], OID_KEEP)
        self.assertIsNone(fail_params[2])
        self.assertIs(fail_params[3], True)
        self.assertIsNone(fail_params[4])
        self.assertTrue(ok_params[2].startswith("["))
        self.assertEqual(ok_params[4], FETCH_URL)

    def test_postgrest_body_omits_image_embedding_only_on_failure(self):
        now = "2026-10-14T17:00:00+00:00"
        failed = worker.embedding_row_body(_result(OID_KEEP, attempted=True), now)
        missing = worker.embedding_row_body(_result(OID_NONE, attempted=False), now)
        ok = worker.embedding_row_body(
            _result(OID_OK, image=np.ones(4, dtype=np.float32), attempted=True, src=FETCH_URL),
            now,
        )
        self.assertNotIn("image_embedding", failed)
        self.assertEqual(failed["image_embedded_at"], now)
        self.assertIsNone(failed["image_vec_source"])
        self.assertIsNone(missing["image_embedding"])
        self.assertIsNone(missing["image_embedded_at"])
        self.assertTrue(ok["image_embedding"].startswith("["))
        self.assertEqual(ok["image_vec_source"], FETCH_URL)

    def test_postgrest_write_posts_the_omitted_body(self):
        src = worker.PostgrestSource.__new__(worker.PostgrestSource)
        src.base = "http://example.test/rest/v1"
        src.table = "offering_embedding"
        src.s = _Session()
        src.write([_result(OID_KEEP, attempted=True)])
        post = src.s.posts[0]
        body = json.loads(post["data"])
        self.assertNotIn("image_embedding", body)
        self.assertEqual(body["offering_id"], OID_KEEP)
        self.assertEqual(
            post["headers"]["Prefer"],
            "return=minimal,resolution=merge-duplicates,missing=default",
        )


class FailureLogTests(unittest.TestCase):
    def test_status_token(self):
        self.assertEqual(worker.image_failure_status("http 404"), "404")
        self.assertEqual(worker.image_failure_status("http 403"), "403")
        self.assertEqual(worker.image_failure_status("http 410"), "410")
        self.assertEqual(worker.image_failure_status("ConnectTimeout"), "ConnectTimeout")
        self.assertEqual(worker.image_failure_status("too large"), "too_large")
        self.assertEqual(worker.image_failure_status(None), "error")
        self.assertEqual(worker.image_failure_status("http "), "error")

    def test_process_batch_logs_offering_id_url_and_status(self):
        cases = [
            ("http 404", "404", "https://cdn.example.com/a.jpg"),
            ("http 403", "403", "https://cdn.shopify.com/s/files/1/item.jpg"),
            ("ConnectTimeout", "ConnectTimeout", "https://images.example.com/b.png"),
            ("too large", "too_large", "https://images.example.com/big.png"),
        ]
        stats = {"img_missing": 0, "img_failed": 0, "img_ok": 0}
        for err, status, url in cases:
            with self.subTest(status=status):
                rows = [{
                    "id": OID_KEEP,
                    "embed_text": "SHOULD_NOT_BE_LOGGED_embed_text",
                    "want_hash": "hash-1",
                    "main_image_url": url,
                }]
                with patch.object(worker, "fetch_image_bytes", return_value=(None, err)):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        with self.assertLogs("embed", level="WARNING") as captured:
                            results = worker.process_batch(rows, _Models(), pool, stats)
                self.assertIsNone(results[0]["image_vec"])
                self.assertTrue(results[0]["image_attempted"])
                self.assertEqual(results[0]["image_error"], err)
                text = "\n".join(captured.output)
                self.assertIn(f"offering_id={OID_KEEP}", text)
                self.assertIn(f"status={status}", text)
                self.assertNotIn("SHOULD_NOT_BE_LOGGED_embed_text", text)
                if "cdn.shopify.com" in url:
                    self.assertIn(url + "?width=512", text)
                else:
                    self.assertIn(f"url={url}", text)
                self.assertIn("image fetch failed ", text)
        self.assertEqual(stats["img_failed"], len(cases))

    def test_decode_failure_logs_offering_id_url_and_status(self):
        url = "https://cdn.example.com/broken.jpg"
        rows = [{
            "id": OID_OLD,
            "embed_text": "x",
            "want_hash": "hash-1",
            "main_image_url": url,
        }]
        stats = {"img_missing": 0, "img_failed": 0, "img_ok": 0}
        with patch.object(worker, "fetch_image_bytes", return_value=(b"not-an-image", None)):
            with patch.object(worker, "decode_image", side_effect=ValueError("bad")):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    with self.assertLogs("embed", level="WARNING") as captured:
                        results = worker.process_batch(rows, _Models(), pool, stats)
        self.assertIsNone(results[0]["image_vec"])
        self.assertTrue(results[0]["image_attempted"])
        text = "\n".join(captured.output)
        self.assertIn(f"image decode/embed failed offering_id={OID_OLD} url={url} status=ValueError", text)
        self.assertNotIn("image fetch failed ", text)


class PostgresUpsertTests(unittest.TestCase):
    conn = None

    @classmethod
    def setUpClass(cls):
        try:
            import psycopg
            cls.conn = psycopg.connect(TEST_DSN, autocommit=False)
        except Exception as exc:
            raise unittest.SkipTest(f"no scratch postgres at {TEST_DSN}: {exc}") from exc
        with cls.conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            db = cur.fetchone()[0]
            if db != "embed_upsert_test":
                raise unittest.SkipTest(f"refusing to touch database {db}")
            cur.execute("DROP TABLE IF EXISTS public.offering_embedding")
            cur.execute(
                """
                CREATE TABLE public.offering_embedding (
                    offering_id uuid PRIMARY KEY,
                    text_embedding text,
                    image_embedding text,
                    embedded_at timestamptz,
                    image_embedded_at timestamptz,
                    image_vec_source text,
                    text_hash text,
                    updated_at timestamptz
                )
                """
            )
        cls.conn.commit()
        cls.sql = worker.offering_embedding_upsert_sql("public.offering_embedding")

    @classmethod
    def tearDownClass(cls):
        if cls.conn is None or cls.conn.closed:
            return
        with cls.conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            if cur.fetchone()[0] == "embed_upsert_test":
                cur.execute("DROP TABLE IF EXISTS public.offering_embedding")
        cls.conn.commit()
        cls.conn.close()

    def setUp(self):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE public.offering_embedding")
        self.conn.commit()

    def _seed(self, oid, image):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.offering_embedding
                    (offering_id, text_embedding, image_embedding,
                     image_embedded_at, image_vec_source, text_hash)
                VALUES (%s::uuid, 'old-text', %s, now() - interval '2 days',
                        'https://cdn.example/old.jpg', 'old-hash')
                """,
                (oid, image),
            )
        self.conn.commit()

    def _exec(self, sql, params):
        with self.conn.cursor() as cur:
            cur.executemany(_text_sql(sql), [params])
        self.conn.commit()

    def _row(self, oid):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_embedding, text_embedding, image_vec_source,
                       (image_embedded_at IS NOT NULL)
                FROM public.offering_embedding
                WHERE offering_id = %s::uuid
                """,
                (oid,),
            )
            return cur.fetchone()

    def test_failed_fetch_keeps_vector_that_the_old_upsert_nulled(self):
        failed = (OID_KEEP, "updated-text", None, True, None, "new-hash")
        self._seed(OID_KEEP, KEPT)
        self._exec(_swap_image_assignment(self.sql, "EXCLUDED.image_embedding"), failed)
        self.assertEqual(self._row(OID_KEEP)[0], None)

        self.setUp()
        self._seed(OID_KEEP, KEPT)
        self._exec(self.sql, failed)
        image, text, source, attempted = self._row(OID_KEEP)
        self.assertEqual(image, KEPT)
        self.assertEqual(text, "updated-text")
        self.assertIsNone(source)
        self.assertIs(attempted, True)

    def test_successful_fetch_replaces_image_embedding(self):
        self._seed(OID_OK, KEPT)
        self._exec(
            self.sql,
            (OID_OK, "updated-text", "new-image-vector", True, FETCH_URL, "new-hash"),
        )
        image, text, source, attempted = self._row(OID_OK)
        self.assertEqual(image, "new-image-vector")
        self.assertEqual(text, "updated-text")
        self.assertEqual(source, FETCH_URL)
        self.assertIs(attempted, True)

    def test_missing_url_still_clears_image_embedding(self):
        self._seed(OID_NONE, KEPT)
        self._exec(self.sql, (OID_NONE, "updated-text", None, False, None, "new-hash"))
        image, _text, source, attempted = self._row(OID_NONE)
        self.assertIsNone(image)
        self.assertIsNone(source)
        self.assertIs(attempted, False)

    def test_failed_fetch_with_no_existing_row_stores_null(self):
        self._exec(self.sql, (OID_NEW, "updated-text", None, True, None, "new-hash"))
        image, text, source, attempted = self._row(OID_NEW)
        self.assertIsNone(image)
        self.assertEqual(text, "updated-text")
        self.assertIsNone(source)
        self.assertIs(attempted, True)


if __name__ == "__main__":
    unittest.main()
