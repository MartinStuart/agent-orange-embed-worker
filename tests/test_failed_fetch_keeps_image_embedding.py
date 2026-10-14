"""A failed image fetch keeps the stored image vector, timestamp and source.

Logic recW2usIq5jMS1CsH. Those three columns change only when a new vector is
written. A missing URL nulls all three. The Postgres section runs the worker's
upsert SQL (halfvec casts rewritten to text) against a scratch database.
"""
from __future__ import annotations

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
STORED_SRC = "https://cdn.example/old.jpg"
STORED_AT = datetime(2020, 1, 15, 12, 0, tzinfo=timezone.utc)
TEST_DSN = os.environ.get("TEST_DATABASE_URL", "dbname=embed_upsert_test")
NEW_VEC = np.ones(4, dtype=np.float32)


def sql_placeholder_count(sql: str) -> int:
    return sql.count("%s")


def _text_sql(sql: str) -> str:
    """Run the worker statement without pgvector. Only the cast may change."""
    out = sql.replace("::halfvec(768)", "::text")
    if out.replace("::text", "::halfvec(768)") != sql:
        raise AssertionError("halfvec cast rewrite was not exact")
    return out


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
    def test_conflict_update_keeps_vector_timestamp_and_source(self):
        sql = worker.offering_embedding_upsert_sql("public.offering_embedding")
        self.assertIn("INSERT INTO public.offering_embedding ", sql)
        for column in ("image_embedding", "image_embedded_at", "image_vec_source"):
            self.assertIn(
                "WHEN EXCLUDED.image_embedding IS NOT NULL THEN EXCLUDED." + column + " "
                "WHEN %s THEN offering_embedding." + column + " "
                "ELSE NULL END",
                sql,
            )
        self.assertNotIn("image_embedding   = EXCLUDED.image_embedding", sql)
        self.assertNotIn("image_embedded_at = EXCLUDED.image_embedded_at", sql)
        self.assertNotIn("image_vec_source  = EXCLUDED.image_vec_source", sql)
        self.assertNotIn("public.offering_embedding.image_embedding", sql)

    def test_postgres_write_preserves_on_failure_and_writes_on_success(self):
        src = worker.PostgresSource.__new__(worker.PostgresSource)
        src.conn = _Conn()
        failed = _result(OID_KEEP, attempted=True)
        missing = _result(OID_NONE, attempted=False)
        ok = _result(OID_OK, image=NEW_VEC, attempted=True, src=FETCH_URL)
        src.write([failed, missing, ok])
        self.assertEqual(src.conn.cur.sql, worker.offering_embedding_upsert_sql())
        self.assertEqual(src.conn.commits, 1)
        fail_params, miss_params, ok_params = src.conn.cur.params
        self.assertEqual(sql_placeholder_count(src.conn.cur.sql), len(fail_params))
        self.assertEqual(fail_params[0], OID_KEEP)
        self.assertIsNone(fail_params[2])
        self.assertIs(fail_params[3], False)
        self.assertIsNone(fail_params[4])
        self.assertEqual(fail_params[6:], (True, True, True))
        self.assertIs(miss_params[3], False)
        self.assertEqual(miss_params[6:], (False, False, False))
        self.assertTrue(ok_params[2].startswith("["))
        self.assertIs(ok_params[3], True)
        self.assertEqual(ok_params[4], FETCH_URL)
        self.assertEqual(ok_params[6:], (False, False, False))

    def test_postgrest_body_omits_image_columns_on_failure(self):
        now = "2026-10-14T17:00:00+00:00"
        failed = worker.embedding_row_body(_result(OID_KEEP, attempted=True), now)
        missing = worker.embedding_row_body(_result(OID_NONE, attempted=False), now)
        ok = worker.embedding_row_body(
            _result(OID_OK, image=NEW_VEC, attempted=True, src=FETCH_URL),
            now,
        )
        for column in ("image_embedding", "image_embedded_at", "image_vec_source"):
            self.assertNotIn(column, failed)
        self.assertIsNone(missing["image_embedding"])
        self.assertIsNone(missing["image_embedded_at"])
        self.assertIsNone(missing["image_vec_source"])
        self.assertTrue(ok["image_embedding"].startswith("["))
        self.assertEqual(ok["image_embedded_at"], now)
        self.assertEqual(ok["image_vec_source"], FETCH_URL)

    def test_postgrest_write_posts_the_omitted_body(self):
        src = worker.PostgrestSource.__new__(worker.PostgrestSource)
        src.base = "http://example.test/rest/v1"
        src.table = "offering_embedding"
        src.s = _Session()
        src.write([_result(OID_KEEP, attempted=True)])
        post = src.s.posts[0]
        body = json.loads(post["data"])
        self.assertEqual(body["offering_id"], OID_KEEP)
        for column in ("image_embedding", "image_embedded_at", "image_vec_source"):
            self.assertNotIn(column, body)
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

    def _seed(self, oid, image=KEPT):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.offering_embedding
                    (offering_id, text_embedding, image_embedding,
                     image_embedded_at, image_vec_source, text_hash)
                VALUES (%s::uuid, 'old-text', %s, %s, %s, 'old-hash')
                """,
                (oid, image, STORED_AT, STORED_SRC),
            )
        self.conn.commit()

    def _exec(self, results):
        params = worker.offering_embedding_upsert_params(results)
        self.assertEqual(sql_placeholder_count(self.sql), len(params[0]))
        with self.conn.cursor() as cur:
            cur.executemany(_text_sql(self.sql), params)
        self.conn.commit()

    def _row(self, oid):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT image_embedding, text_embedding, image_vec_source, image_embedded_at
                FROM public.offering_embedding
                WHERE offering_id = %s::uuid
                """,
                (oid,),
            )
            return cur.fetchone()

    def test_failed_fetch_keeps_vector_timestamp_and_source(self):
        self._seed(OID_KEEP)
        self._exec([_result(OID_KEEP, attempted=True)])
        image, text, source, embedded_at = self._row(OID_KEEP)
        self.assertEqual(image, KEPT)
        self.assertNotEqual(text, "old-text")
        self.assertEqual(source, STORED_SRC)
        self.assertEqual(embedded_at, STORED_AT)

    def test_new_vector_updates_embedding_timestamp_and_source(self):
        self._seed(OID_OK)
        self._exec([_result(OID_OK, image=NEW_VEC, attempted=True, src=FETCH_URL)])
        image, text, source, embedded_at = self._row(OID_OK)
        self.assertEqual(image, worker.to_vec_literal(NEW_VEC))
        self.assertNotEqual(image, KEPT)
        self.assertNotEqual(text, "old-text")
        self.assertEqual(source, FETCH_URL)
        self.assertNotEqual(embedded_at, STORED_AT)
        self.assertIsNotNone(embedded_at)
        age = abs((datetime.now(timezone.utc) - embedded_at).total_seconds())
        self.assertLess(age, 60)

    def test_missing_url_nulls_embedding_timestamp_and_source(self):
        self._seed(OID_NONE)
        self._exec([_result(OID_NONE, attempted=False)])
        image, text, source, embedded_at = self._row(OID_NONE)
        self.assertIsNone(image)
        self.assertIsNone(source)
        self.assertIsNone(embedded_at)
        self.assertNotEqual(text, "old-text")

    def test_failed_fetch_with_no_existing_row_stores_nulls(self):
        self._exec([_result(OID_NEW, attempted=True)])
        image, text, source, embedded_at = self._row(OID_NEW)
        self.assertIsNone(image)
        self.assertIsNone(source)
        self.assertIsNone(embedded_at)
        self.assertNotEqual(text, "old-text")

    def test_batch_keeps_one_row_and_rewrites_the_other(self):
        self._seed(OID_KEEP)
        self._seed(OID_OK)
        self._exec([
            _result(OID_KEEP, attempted=True),
            _result(OID_OK, image=NEW_VEC, attempted=True, src=FETCH_URL),
        ])
        kept_image, _text, kept_source, kept_at = self._row(OID_KEEP)
        new_image, _text, new_source, new_at = self._row(OID_OK)
        self.assertEqual(kept_image, KEPT)
        self.assertEqual(kept_source, STORED_SRC)
        self.assertEqual(kept_at, STORED_AT)
        self.assertEqual(new_image, worker.to_vec_literal(NEW_VEC))
        self.assertEqual(new_source, FETCH_URL)
        self.assertNotEqual(new_at, STORED_AT)


if __name__ == "__main__":
    unittest.main()
