import pytest
import sqlite3
from anysqlite3 import connect


@pytest.mark.anyio
async def test_connect():
    async with await connect(":memory:") as conn:
        assert conn is not None


@pytest.mark.anyio
async def test_execute():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
        cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
        row = await cursor.fetchone()
        assert row == ("test_value",)


@pytest.mark.anyio
async def test_fetchall():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.executemany(
            "INSERT INTO test (value) VALUES (?)", [("value1",), ("value2",)]
        )
        cursor = await conn.execute("SELECT value FROM test")
        rows = await cursor.fetchall()
        assert rows == [("value1",), ("value2",)]


@pytest.mark.anyio
async def test_transaction_commit_rollback():
    async with await connect(":memory:") as conn:
        async with conn.transaction() as t:
            await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            await conn.execute(
                "INSERT INTO test (value) VALUES (?)", ("value_before_commit",)
            )
            await t.commit()
            cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
            row = await cursor.fetchone()
            assert row == ("value_before_commit",)

            await conn.execute(
                "UPDATE test SET value = ? WHERE id = 1", ("value_after_rollback",)
            )
            await t.rollback()
            cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
            row = await cursor.fetchone()
            assert row == ("value_before_commit",)


@pytest.mark.anyio
async def test_close():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    with pytest.raises(sqlite3.ProgrammingError):
        await conn.execute(
            "INSERT INTO test (value) VALUES (?)", ("value_after_close",)
        )


@pytest.mark.anyio
async def test_row_factory():
    async with await connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
        cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
        row = await cursor.fetchone()
        assert isinstance(row, sqlite3.Row)
        conn.row_factory = None
        cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
        row = await cursor.fetchone()
        assert isinstance(row, tuple)


@pytest.mark.anyio
async def test_iter():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.executemany(
            "INSERT INTO test (value) VALUES (?)", [("value1",), ("value2",)]
        )
        cursor = await conn.execute("SELECT value FROM test")
        values = [row[0] async for row in cursor]
        assert values == ["value1", "value2"]


@pytest.mark.anyio
async def test_fetchmany():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.executemany(
            "INSERT INTO test (value) VALUES (?)",
            [("value1",), ("value2",), ("value3",)],
        )
        cursor = await conn.execute("SELECT value FROM test")
        rows = await cursor.fetchmany(2)
        assert rows == [("value1",), ("value2",)]
        rows = await cursor.fetchmany(2)
        assert rows == [("value3",)]


@pytest.mark.anyio
async def test_executemany():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.executemany(
            "INSERT INTO test (value) VALUES (?)",
            [("value1",), ("value2",), ("value3",)],
        )
        cursor = await conn.execute("SELECT value FROM test")
        rows = await cursor.fetchall()
        assert rows == [("value1",), ("value2",), ("value3",)]


@pytest.mark.anyio
async def test_transaction():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.execute("INSERT INTO test (value) VALUES (?)", ("value1",))
        async with conn.transaction():
            await conn.execute("UPDATE test SET value = ?", ("value2",))
        cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
        row = await cursor.fetchone()
        assert row == ("value2",)


@pytest.mark.anyio
async def test_cursor_in_transaction():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.execute("INSERT INTO test (value) VALUES (?)", ("value1",))
        async with conn.transaction():
            cursor = conn.cursor()
            await cursor.execute("SELECT value FROM test WHERE id = 1")
            row = await cursor.fetchone()
            assert row == ("value1",)
            await cursor.executemany(
                "INSERT INTO test (value) VALUES (?)", [("value2",), ("value3",)]
            )
            await cursor.execute("SELECT value FROM test")
            rows = await cursor.fetchall()
            assert rows == [("value1",), ("value2",), ("value3",)]
            await cursor.executescript("DELETE FROM test")
            await cursor.execute("SELECT value FROM test")
            rows = await cursor.fetchall()
            assert rows == []


@pytest.mark.anyio
async def test_nested_transaction():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.execute("INSERT INTO test (value) VALUES (?)", ("value1",))
        async with conn.transaction():
            await conn.execute("UPDATE test SET value = ?", ("value2",))
            with pytest.raises(
                RuntimeError, match="Cannot start a transaction in another transaction"
            ):
                async with conn.transaction():
                    await conn.execute("UPDATE test SET value = ?", ("value3",))


@pytest.mark.anyio
async def test_transaction_exception():
    async with await connect(":memory:") as conn:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        await conn.execute("INSERT INTO test (value) VALUES (?)", ("value1",))
        with pytest.raises(ValueError):
            async with conn.transaction():
                await conn.execute("UPDATE test SET value = ?", ("value2",))
                raise ValueError
        cursor = await conn.execute("SELECT value FROM test WHERE id = 1")
        row = await cursor.fetchone()
        assert row == ("value1",)


@pytest.mark.anyio
async def test_sqlite_not_threadsafe():
    old = sqlite3.threadsafety
    try:
        sqlite3.threadsafety = 0
        with pytest.raises(RuntimeError, match="SQLite is not thread-safe"):
            await connect(":memory:")
    finally:
        sqlite3.threadsafety = old


@pytest.mark.anyio
async def test_unsupported_transaction_methods():
    async with await connect(":memory:") as db:
        with pytest.raises(NotImplementedError):
            db.commit()
        with pytest.raises(NotImplementedError):
            db.rollback()


@pytest.mark.anyio
async def test_interrupt():
    async with await connect(":memory:") as conn:
        await conn.interrupt()


@pytest.mark.anyio
async def test_executescript():
    async with await connect(":memory:") as conn:
        await conn.executescript(
            """
            CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO test (value) VALUES ('value1');
            INSERT INTO test (value) VALUES ('value2');
            """
        )
        cursor = await conn.execute("SELECT value FROM test")
        rows = await cursor.fetchall()
        assert rows == [("value1",), ("value2",)]


@pytest.mark.anyio
async def test_db_aclose():
    conn = await connect(":memory:")
    await conn.aclose()
    with pytest.raises(sqlite3.ProgrammingError):
        await conn.execute("SELECT * FROM sqlite_master")


@pytest.mark.anyio
async def test_cursor_aclose():
    async with await connect(":memory:") as conn:
        cursor = await conn.execute("SELECT 1")
        await cursor.aclose()
        with pytest.raises(sqlite3.ProgrammingError):
            await cursor.fetchone()
        await cursor.aclose()
