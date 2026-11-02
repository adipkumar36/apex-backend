# Keep your existing reviewer account

The account database used by older prototype copies was stored next to `main.py`. This version stores it in:

`%LOCALAPPDATA%\ApexResearch\apex.db`

For the first update, stop the old server, then extract/overwrite the updated files in the same project folder that contains your existing `apex.db`. Keep that database file. On its first start, the app copies it safely into the shared LocalAppData location. Later extracted versions will use that same database automatically.

If you extracted the update into a new folder already, copy the old `apex.db` into that new project folder before the first launch. The app will migrate it on startup. Keep a backup of the original database. The ZIP intentionally does not contain account data or password hashes.

If you already started the new version before copying the old database and created a new empty shared database, stop the server and back up `%LOCALAPPDATA%\ApexResearch\apex.db` before replacing it with the old database and restarting.
