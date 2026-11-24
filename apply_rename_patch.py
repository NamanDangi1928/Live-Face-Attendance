"""
Run ONCE from the project folder (the one with app.py):

    python apply_rename_patch.py

Renames the project to  "Live Face Recognition Attendance":
  * dashboard: browser-tab title, project name under the logo, and as the page heading
  * every page in templates/: "Eye-X" -> new name, and "... - Smart Attendance" in <title> tags
  * app.py: the FastAPI title and the start-up banner
Makes a backup (.bak_rename) of every file it changes. Safe to run twice.
"""
import re, shutil, sys
from pathlib import Path

NAME = "Live Face Recognition Attendance"

tpl = Path("templates")
if not tpl.exists() or not Path("app.py").exists():
    sys.exit("Run this from the folder that contains app.py and the templates folder")

report = []

def save(path, old, new, what):
    if new != old:
        bak = path.with_name(path.name + ".bak_rename")
        if not bak.exists():
            shutil.copy(path, bak)
        path.write_text(new, encoding="utf-8")
        report.append((what, True))
    return new != old

# ---- all templates: Eye-X and <title> ----------------------------------------
for f in sorted(tpl.glob("*.html")):
    src = f.read_text(encoding="utf-8")
    new = src.replace("Eye-X", NAME)
    new = re.sub(r"(<title>[^<]*?)Smart Attendance( System)?", lambda m: m.group(1) + NAME, new)
    save(f, src, new, f"{f.name}: name in text/title")

# ---- dashboard main screen ----------------------------------------------------
dash = tpl / "dashboard_new.html"
if dash.exists():
    src = dash.read_text(encoding="utf-8")
    new = src

    if "nav-brand" not in new:
        new, n = re.subn(
            r'(<div class="nav-logo">\s*<img [^>]*>\s*</div>)',
            r'\1\n        <div class="nav-brand" style="font-size:0.68rem;font-weight:700;line-height:1.25;'
            r'text-align:center;padding:0 8px 12px;color:var(--text-primary);">' + NAME + '</div>',
            new, count=1)
        report.append(("dashboard: name under the logo", n == 1))

    if f'<h1 class="page-title">{NAME}</h1>' not in new:
        new, n = re.subn(
            r'<h1 class="page-title">Live Camera Feeds</h1>',
            '<div style="font-size:0.75rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;'
            'color:var(--text-secondary);margin-bottom:4px;">Live Camera Feeds</div>\n            '
            f'<h1 class="page-title">{NAME}</h1>',
            new, count=1)
        report.append(("dashboard: project name as page heading", n == 1))

    save(dash, src, new, "dashboard saved")
else:
    report.append(("templates/dashboard_new.html found", False))

# ---- app.py -------------------------------------------------------------------
app = Path("app.py")
src = app.read_text(encoding="utf-8")
new = src.replace('title="Smart Attendance System"', f'title="{NAME}"')
new = new.replace("Smart Attendance System Starting...", f"{NAME} Starting...")
save(app, src, new, "app.py: app title and start-up banner")

print()
for what, ok in report:
    print(("OK      " if ok else "FAILED  ") + what)
print("\nDone. Restart the server and refresh the browser with Ctrl+Shift+R.")