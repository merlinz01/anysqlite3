# anysqlite3

**anysqlite3** is a wrapper around `sqlite3` that allows you to interact with SQLite databases from async code.
It is built on top of the built-in `sqlite3` module and is compatible with both `asyncio` and `trio` code.

## Installation

```bash
pip install anysqlite3
```

## Usage

**anysqlite3** provides essentially the same API as the built-in `sqlite3` module, but with async versions of most methods.

```python
import anyio # or asyncio or trio
import anysqlite3

async def main():
    async with await anysqlite3.connect(":memory:") as db:
        await db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await db.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
        await db.execute("INSERT INTO users (name) VALUES (?)", ("Bob",))

        async for row in db.execute("SELECT * FROM users"):
            print(row)

anyio.run(main())
```

### Transactions

**anysqlite3** provides a context manager for transactions.
Use this instead of the database's `commit` and `rollback` methods.
Transactions hold a lock on the database, so you should always use them in a `with` block.

```python
async with db.transaction() as t:
    await db.execute("INSERT INTO users (name) VALUES (?)", ("Charlie",))
    await t.rollback()
    await db.execute("INSERT INTO users (name) VALUES (?)", ("David",))
    await t.commit() # Optional, as the transaction will commit automatically the with block exits without an exception
```

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.
