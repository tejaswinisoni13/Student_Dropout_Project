# EduLens setup

## Folder structure
```
edulens/
├── app.py                      # frontend (replaces your old app.py)
├── pipeline.py                 # backend - unchanged
├── make_user.py                # creates a password hash for a new account
├── requirements.txt            # unchanged (no new dependency; hashing uses Python's standard library)
├── test_edulens.py             # optional automated checks
├── .gitignore                  # keeps secrets.toml out of GitHub
└── .streamlit/
    ├── config.toml             # Navy / Ivory / Sage theme
    ├── secrets.toml.example    # template (placeholders only, safe to commit)
    └── secrets.toml            # YOUR real accounts - create it, never commit it
```

## Create the first account
1. `python make_user.py`
2. Enter a username, display name, role number(s) and a password (10+ characters, typed twice, hidden).
3. Copy the printed `[edulens.users.<username>]` block into `.streamlit/secrets.toml`
   (create the file by copying `secrets.toml.example`; delete the placeholder example block).
4. Repeat for each person; one block per account. To remove an account, delete its block.

Only a salted PBKDF2-SHA256 hash (600,000 iterations) is stored. No password appears in app.py or in Git.
Deploying on Streamlit Community Cloud: paste the same TOML into the app's *Secrets* box instead of a file.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Roles
| Role | Pages | Downloads |
|---|---|---|
| Faculty Member | all 10 tabs | student results, intervention list, class summary |
| Course Coordinator | all 10 tabs | intervention list and class summary only |

An account can sign in only under roles listed in its `roles = [...]`. Permissions live in the `ROLES` dictionary at the
top of app.py.

## Limits to know about
- Sessions last for the browser session only: refreshing the page or closing the browser signs the user out.
- The 5-failed-attempts lock (30 s) is tracked per browser session, so it slows guessing but is not server-wide rate
  limiting. Serve the app over HTTPS if it is ever exposed beyond your college network.
- There is no password-reset screen; change a password by generating a new hash with make_user.py.
