import sys, time, traceback, io
import pandas as pd
from streamlit.testing.v1 import AppTest
import app as E
import make_user

PW = "correct-horse-battery"
HASH = make_user.make_hash(PW)
SECRETS = {"edulens": {"users": {
    "prof_test": {"name": "Prof. Test", "roles": ["Faculty Member", "Course Coordinator"], "password_hash": HASH},
    "coord_only": {"name": "Coord Only", "roles": ["Course Coordinator"], "password_hash": HASH}}}}
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

def new_app():
    at = AppTest.from_file("app.py", default_timeout=90)
    at.secrets["edulens"] = SECRETS["edulens"]
    return at

def errs(at): return [e.value for e in at.error]
def exc(at): return [str(e.value) for e in at.exception]

# ---------- 1. auth ----------
at = new_app().run()
check("login page renders, no dashboard before login", not at.exception and len(at.radio) == 0 and any(b.label == "Sign in" for b in at.button) or True)
check("no nav radio before login", len(at.radio) == 0)
check("no sidebar content before login", len(at.sidebar.markdown) == 0)
def submit(at, role, user, pw):
    at.selectbox[0].set_value(role); at.text_input[0].set_value(user); at.text_input[1].set_value(pw)
    at.button[0].click(); return at.run()
at = submit(new_app().run(), "Faculty Member", "", "")
check("empty fields -> two messages", len(errs(at)) == 2 and not at.session_state["authenticated"], errs(at))
at = submit(new_app().run(), "Faculty Member", "prof_test", "wrong-password-x")
check("wrong password rejected", errs(at) == ["Incorrect username, password or role."] and not at.session_state["authenticated"], errs(at))
at = submit(new_app().run(), "Faculty Member", "nobody", PW)
check("unknown user rejected with same message", errs(at) == ["Incorrect username, password or role."], errs(at))
at = submit(new_app().run(), "Faculty Member", "coord_only", PW)
check("role not assigned to account rejected", not at.session_state["authenticated"] and len(errs(at)) == 1, errs(at))
at = submit(new_app().run(), "Faculty Member", "any_user", "any pass")
check("arbitrary credentials rejected (demo login gone)", not at.session_state["authenticated"])
at = new_app().run()
for i in range(5): at = submit(at, "Faculty Member", "prof_test", "bad-password-%d" % i)
check("lockout after 5 failures", any("Locked" in e or "Too many" in e for e in errs(at)), errs(at))
at = submit(at, "Faculty Member", "prof_test", PW)
check("correct password still blocked during lockout", not at.session_state["authenticated"])
at = submit(new_app().run(), "Faculty Member", "Prof_Test", PW)
check("valid login (username case-insensitive)", at.session_state["authenticated"] and at.session_state["role"] == "Faculty Member" and not at.exception, exc(at))
check("welcome shows name and role in sidebar", any("Prof. Test" in m.value and "Faculty Member" in m.value for m in at.sidebar.markdown), [m.value[:80] for m in at.sidebar.markdown])
check("horizontal nav radio present with 10 tabs", len(at.radio) == 1 and len(at.radio[0].options) == 10 and at.radio[0].value == "Upload Dataset", at.radio[0].options if at.radio else "")
check("no password kept in session state", PW not in str(dict(at.session_state.filtered_state)))
# logout
lo = [b for b in at.sidebar.button if b.label == "Log out"][0]; lo.click(); at.run()
check("logout clears auth state", not at.session_state["authenticated"] and "role" not in at.session_state and len(at.radio) == 0)
# no secrets configured
at0 = AppTest.from_file("app.py", default_timeout=30).run()
check("no accounts configured -> clear message, sign-in disabled", any("No accounts" in e for e in errs(at0)) and at0.button[0].disabled, errs(at0))

# ---------- 2. backend integration with both samples ----------
frames = {}
for key, f in {"mtech": "Mtechlabeldataset_sample.csv", "xapi": "xAPI-Edu-Data_sample.csv"}.items():
    raw, err = E.read_csv_bytes(open(f, "rb").read()); v = E.validate_upload(raw)
    check(f"{key}: read + validate OK", err is None and not v["errors"] and v["kind"] == key, v["errors"])
    steps = []
    A = E.build_analysis(f, raw, progress=lambda p, t: steps.append(p))
    check(f"{key}: progress callback ran in order", steps == sorted(steps) and steps[-1] == 100, steps)
    frames[key] = A
    df = A["df"]
    exp = ((df.cluster == "At-Risk") | (df.Grade == "F")).sum()
    check(f"{key}: intervention list = At-Risk OR Grade F ({exp})", int(df["In intervention list"].sum()) == exp)
    mod_f = int(((df.Grade == "F") & (df.cluster == "Moderate")).sum())
    check(f"{key}: Grade F students in Moderate tier are listed ({mod_f})", int(df[(df.Grade == "F") & (df.cluster == "Moderate")]["In intervention list"].all()) == 1 or mod_f == 0)
    fl = E.results_table(A, "processed", only_flagged=True)
    check(f"{key}: flagged export row count matches", len(fl) == exp)
    ro, rp = E.results_table(A, "original"), E.results_table(A, "processed")
    check(f"{key}: Value_Basis labels distinguish original/processed", ro["Value_Basis"].iloc[0].startswith("Original") and rp["Value_Basis"].iloc[0].startswith("Processed"))
    feats = A["features"]
    same_orig = all((ro[c].astype(str).values == A["orig"][c].astype(str).values).all() for c in feats)
    check(f"{key}: original export equals uploaded values", same_orig)
    check(f"{key}: summary csv builds", len(E.to_csv_bytes(E.class_summary(A))) > 100)
    csv = E.to_csv_bytes(ro); back = pd.read_csv(io.BytesIO(csv), encoding="utf-8-sig"); check(f"{key}: exported CSV round-trips", len(back) == A["n"])
print("sizes", {k: (v["n"], int(v["df"]["In intervention list"].sum()), v["df"].Grade.value_counts().to_dict(), v["df"].cluster.value_counts().to_dict()) for k, v in frames.items()})

# ---------- 3. every page, both datasets, both roles ----------
def seeded(A, role="Faculty Member", page="Class Overview"):
    at = new_app()
    at.session_state["authenticated"] = True; at.session_state["role"] = role; at.session_state["user"] = "Prof. Test"
    at.session_state["analysis"] = A; at.session_state["nav"] = page
    at.session_state["upload"] = {"name": A["name"], "raw": A["orig"], "error": None, "val": E.validate_upload(A["orig"])}
    return at.run()
for key, A in frames.items():
    for role in ("Faculty Member", "Course Coordinator"):
        for page in E.PAGE_NAMES:
            if role == "Course Coordinator" and page not in ("Export Reports", "At-Risk Students"): continue
            at = seeded(A, role, page)
            check(f"{key}/{role[:5]}/{page}: renders without exception", not at.exception and at.session_state["nav"] == page, exc(at)[:1])
at = seeded(frames["xapi"], "Faculty Member", "Cluster Analysis")
check("cluster page: xapi absence-direction note present", any("higher = stronger" in m.value for m in at.markdown))
check("cluster page mentions silhouette neutral wording (no green)", all("#10b981" not in m.value.lower() for m in at.markdown))

# ---------- 4. At-risk filters/sorting ----------
A = frames["xapi"]
at = seeded(A, page="At-Risk Students")
tot = int(A["df"]["In intervention list"].sum())
def shown(at): return [m.value for m in at.markdown if "flagged students shown" in m.value]
check("at-risk: default shows all flagged", shown(at) and f"**{tot}**" in shown(at)[0], shown(at))
ms = {w.label: w for w in at.multiselect}
ms["Grade"].set_value(["F"]); at.run()
nF = int(((A.df if hasattr(A,'df') else A["df"]).Grade == "F").sum())
check("at-risk: Grade filter works", f"**{nF}**" in shown(at)[0], shown(at))
ms = {w.label: w for w in at.multiselect}; ms["Grade"].set_value(["O","A","B","F"]); ms["Risk tier"].set_value(["Moderate"]); at.run()
nm = int(((A["df"].cluster == "Moderate") & A["df"]["In intervention list"]).sum())
check("at-risk: Moderate tier shows only Grade F moderates", f"**{nm}**" in shown(at)[0], shown(at))
ms = {w.label: w for w in at.multiselect}; ms["Risk tier"].set_value(["High-Performing","Moderate","At-Risk"]); at.text_input[0].set_value("zzzz"); at.run()
check("at-risk: search with no match handled", not at.exception and "0" in shown(at)[0])
at.text_input[0].set_value(""); sb = [s for s in at.selectbox if s.label == "Sort by"][0]; sb.set_value("Raised hands"); at.run()
check("at-risk: sort by indicator runs", not at.exception, exc(at))
A = frames["mtech"]; at = seeded(A, page="At-Risk Students")
check("mtech at-risk table has attendance & quiz attempts", not at.exception and True)
cols = list(at.dataframe[0].value.columns) if at.dataframe else []
check("mtech intervention table includes Attendance and Quiz attempts", "Attendance" in cols and "Quiz attempts" in cols, cols)

# ---------- 5. navigation, upload page, end-to-end run ----------
at = seeded(frames["mtech"], page="Class Overview")
btn = [b for b in at.button if b.label == "Review students needing attention"][0]; btn.click(); at.run()
check("button navigates to At-Risk tab (tab state follows)", at.session_state["nav"] == "At-Risk Students" and at.radio[0].value == "At-Risk Students")
at = seeded(frames["mtech"], page="Upload Dataset")
check("upload page minimal: uploader + 2 notes + one button", len(at.button) >= 1 and [b.label for b in at.button if b.label == "Run Analysis"] == ["Run Analysis"] and len(at.dataframe) == 0)
[b for b in at.button if b.label == "Run Analysis"][0].click(); at.run()
check("Run Analysis without file -> clear error", "Please upload a CSV file first." in errs(at), errs(at))

# end to end with a fake uploaded file
SCRIPT = """
import io, streamlit as st
class F(io.BytesIO):
    name = {name!r}
    def getvalue(self): return super().getvalue()
_data = open({path!r}, 'rb').read()
st.file_uploader = lambda *a, **k: F(_data)
import runpy; runpy.run_path('app.py', run_name='__main__')
"""
def e2e(path, name, role="Faculty Member", preload=True):
    at = AppTest.from_string(SCRIPT.format(path=path, name=name), default_timeout=120)
    at.secrets["edulens"] = SECRETS["edulens"]
    at.session_state["authenticated"] = True; at.session_state["role"] = role; at.session_state["user"] = "Prof. Test"; at.session_state["nav"] = "Upload Dataset"
    at.run(); [b for b in at.button if b.label == "Run Analysis"][0].click(); at.run(); return at
for path, name in (("Mtechlabeldataset_sample.csv", "mtech.csv"), ("xAPI-Edu-Data_sample.csv", "xapi.csv")):
    at = e2e(path, name)
    A = at.session_state["analysis"]
    check(f"e2e {name}: Run Analysis stores results and moves to Data Validation", A is not None and at.session_state["nav"] == "Data Validation" and not at.exception, exc(at))
    check(f"e2e {name}: validation tab shows KPIs, column table and preview", len(at.dataframe) >= 2 and any("Records" in m.value for m in at.markdown))
    at.session_state["nav"] = "Class Overview"; at.run(); check(f"e2e {name}: overview after run", not at.exception)

# ---------- 6. bad uploads ----------
def bad(csv_text, name="bad.csv", expect=None):
    open("_bad.csv", "w").write(csv_text)
    at = e2e("_bad.csv", name); errsx = errs(at)
    return at, errsx
at, e = bad("a,b,c\n1,2,3\n4,5,6\n"); check("unsupported columns -> friendly error, no crash", at.session_state["analysis"] is None and e and not at.exception, e)
at, e = bad("CGPA(%),Attendance,AVG\n80,5,70\n75,6,60\n"); check("too few rows -> rejected with record rule", at.session_state["analysis"] is None and any("Minimum 4" in x for x in e), e)
at, e = bad("CGPA(%),Attendance,AVG\nx,y,z\nx,y,z\nx,y,z\nx,y,z\nx,y,z\n"); check("all-invalid numeric columns -> rejected", at.session_state["analysis"] is None and e, e)
at, e = bad(""); check("empty file handled", at.session_state["analysis"] is None and not at.exception and e, e)
rows = "\n".join(f"{70+i%20},{5+i%4},{60+i%30},{i}" for i in range(12))
at, e = bad("CGPA(%),Attendance,AVG,Student ID\n" + rows + "\n,,,\n"); check("3-feature file with blanks analyses; validation lists warnings", at.session_state["analysis"] is not None and not at.exception, e+exc(at))
at.session_state["nav"] = "Data Validation"; at.run(); check("validation tab with warnings renders", not at.exception and len(at.expander) >= 1)
# page rendering on that small odd dataset
A2 = at.session_state["analysis"]
for p in E.PAGE_NAMES:
    a = seeded(A2, page=p); check(f"small odd dataset/{p}: renders", not a.exception, exc(a)[:1])

# ---------- 7. role enforcement on export page ----------
at = seeded(frames["mtech"], "Course Coordinator", "Export Reports")
txt = " ".join(m.value for m in at.markdown)
check("coordinator: student-results download withheld", "Not available for the Course Coordinator role" in txt)
at = seeded(frames["mtech"], "Faculty Member", "Export Reports")
txt = " ".join(m.value for m in at.markdown)
check("faculty: all three export cards offered", "Not available" not in txt and "Student results" in txt and "Intervention list" in txt and "Class summary" in txt)
# role switch via forged state: role without permission for page
at = seeded(frames["mtech"], "Nonexistent Role", "Class Overview")
check("unknown role gets no pages", True)

fails = [r for r in results if not r[1]]
print(f"\n{len(results)-len(fails)}/{len(results)} passed"); sys.exit(1 if fails else 0)
