import sqlite3
import functools
import contextvars

import anyio
import anyio.to_thread

from typing import Any, TypeAlias, Self
from .version import __version__
from pathlib import Path

__all__ = ["connect", "Cursor", "Connection", "__version__"]

RowFactoryType: TypeAlias = sqlite3.Row | tuple

_current_transaction = contextvars.ContextVar["_Transaction | None"](
    "_current_transaction", default=None
)


class _Transaction:
    def __init__(self, conn: "Connection"):
        self._conn = conn

    async def __aenter__(self):
        _current_transaction.set(self)
        try:
            await self._conn._transaction_lock.acquire()
        except RuntimeError:
            raise RuntimeError("Cannot start a transaction in another transaction")
        await self._conn.execute("BEGIN")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                await self.commit()
            else:
                await self.rollback()
        finally:
            self._conn._transaction_lock.release()
            _current_transaction.set(None)

    async def commit(self):
        await anyio.to_thread.run_sync(sqlite3.Connection.commit, self._conn)

    async def rollback(self):
        await anyio.to_thread.run_sync(sqlite3.Connection.rollback, self._conn)


class Cursor(sqlite3.Cursor):
    connection: "Connection"  # type: ignore

    def __aiter__(self):
        return self

    async def __anext__(self) -> RowFactoryType:
        row = await self.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row

    async def fetchall(self):  # type: ignore
        return await anyio.to_thread.run_sync(super().fetchall)

    async def fetchmany(self, size: int = -1) -> list[RowFactoryType]:  # type: ignore
        return await anyio.to_thread.run_sync(super().fetchmany, size)

    async def fetchone(self) -> RowFactoryType | None:
        return await anyio.to_thread.run_sync(super().fetchone)

    async def execute(self, sql: str, parameters: tuple[Any] = ()) -> None:  # type: ignore
        if _current_transaction.get() is None:
            async with self.connection.transaction():
                await anyio.to_thread.run_sync(super().execute, sql, parameters)
        else:
            await anyio.to_thread.run_sync(super().execute, sql, parameters)

    async def executemany(self, sql: str, parameters: list[tuple[Any]]) -> None:  # type: ignore
        if _current_transaction.get() is None:
            async with self.connection.transaction():
                await anyio.to_thread.run_sync(super().executemany, sql, parameters)
        else:
            await anyio.to_thread.run_sync(super().executemany, sql, parameters)

    async def executescript(self, sql: str) -> None:  # type: ignore
        if _current_transaction.get() is None:
            async with self.connection.transaction():
                await anyio.to_thread.run_sync(super().executescript, sql)
        else:
            await anyio.to_thread.run_sync(super().executescript, sql)

    async def aclose(self) -> None:
        await anyio.to_thread.run_sync(super().close)


class Connection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transaction_lock = anyio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def cursor(self, factory=None) -> Cursor:
        if factory is None:
            factory = Cursor
        return super().cursor(factory)

    async def execute(self, sql: str, parameters: tuple = ()) -> Cursor:  # type: ignore
        c = self.cursor()
        await c.execute(sql, parameters)
        return c

    async def executemany(self, sql: str, parameters: list[tuple]) -> Cursor:  # type: ignore
        c = self.cursor()
        await c.executemany(sql, parameters)
        return c

    async def executescript(self, sql: str) -> Cursor:  # type: ignore
        c = self.cursor()
        await c.executescript(sql)
        return c

    async def aclose(self) -> None:
        await anyio.to_thread.run_sync(self.close)

    async def interrupt(self) -> None:  # type: ignore
        await anyio.to_thread.run_sync(super().interrupt)

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def commit(self) -> None:
        raise NotImplementedError("Use a transaction instead")

    def rollback(self) -> None:
        raise NotImplementedError("Use a transaction instead")


async def connect(db: str | bytes | Path, **kwargs) -> Connection:
    if sqlite3.threadsafety < 3:
        raise RuntimeError("SQLite is not thread-safe")
    conn = await anyio.to_thread.run_sync(
        functools.partial(
            sqlite3.connect,
            db,
            check_same_thread=False,
            factory=Connection,
            **kwargs,
        )
    )
    return conn  # type: ignore
