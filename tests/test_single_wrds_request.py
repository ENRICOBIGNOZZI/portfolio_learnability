import unittest
from unittest.mock import patch

import pandas as pd

from download_JKP.dataset import MAX_WRDS_REQUESTS, SingleRequestWRDS


class _FakeConnection:
    def execution_options(self, **kwargs):
        return self


class _FakeConnectContext:
    def __init__(self, engine):
        self.engine = engine
        self.connection = _FakeConnection()

    def __enter__(self):
        self.engine.connect_calls += 1
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self):
        self.connect_calls = 0
        self.dispose_calls = 0

    def connect(self):
        return _FakeConnectContext(self)

    def dispose(self):
        self.dispose_calls += 1


class SingleRequestWRDSTest(unittest.TestCase):
    def test_request_budget_is_one(self):
        self.assertEqual(MAX_WRDS_REQUESTS, 1)

    def test_second_request_is_blocked_before_connect(self):
        engine = _FakeEngine()
        db = SingleRequestWRDS(engine)

        chunks = [pd.DataFrame({"x": [1, 2]})]

        with patch(
            "download_JKP.dataset.pd.read_sql_query",
            return_value=iter(chunks),
        ) as read_sql:
            first = list(db.stream_sql("SELECT 1"))

            self.assertEqual(len(first), 1)
            self.assertEqual(db.request_count, 1)
            self.assertEqual(db.connection_attempts, 1)
            self.assertEqual(engine.connect_calls, 1)
            self.assertEqual(read_sql.call_count, 1)

            with self.assertRaises(RuntimeError):
                list(db.stream_sql("SELECT 2"))

            # The guard must fail before any second connection/query reaches WRDS.
            self.assertEqual(db.request_count, 1)
            self.assertEqual(db.connection_attempts, 1)
            self.assertEqual(engine.connect_calls, 1)
            self.assertEqual(read_sql.call_count, 1)

    def test_single_request_assertion(self):
        engine = _FakeEngine()
        db = SingleRequestWRDS(engine)

        with self.assertRaises(AssertionError):
            db.assert_single_request()

        with patch(
            "download_JKP.dataset.pd.read_sql_query",
            return_value=iter([pd.DataFrame({"x": [1]})]),
        ):
            list(db.stream_sql("SELECT 1"))

        db.assert_single_request()


if __name__ == "__main__":
    unittest.main()
