"""
Run ONCE from the project folder (the one with app.py):

    python apply_alarm_patch.py

Adds a sound alarm + on-screen banner + ON/OFF and Test buttons to
templates/dashboard_new.html. Makes a backup first. Safe to run twice.
"""
import re, shutil, sys
from pathlib import Path

path = Path("templates/dashboard_new.html")
if not path.exists():
    sys.exit("templates/dashboard_new.html not found. Run this from the folder that contains app.py")

src = path.read_text(encoding="utf-8")
bak = Path("templates/dashboard_new.html.bak")
if not bak.exists():
    shutil.copy(path, bak)
    print("Backup saved as templates/dashboard_new.html.bak")

report = []
def done(name, ok): report.append((name, ok))

ALARM_JS = r'''
        // ===== Unknown-person alarm (free: sound is generated in the browser, no file needed) =====
        let alarmCtx = null;
        let alarmEnabled = localStorage.getItem('alarmEnabled') !== 'false';
        let alarmPlaying = false;

        function getAlarmCtx() {
            if (!alarmCtx) {
                const AC = window.AudioContext || window.webkitAudioContext;
                if (AC) alarmCtx = new AC();
            }
            if (alarmCtx && alarmCtx.state === 'suspended') alarmCtx.resume();
            return alarmCtx;
        }
        // Browsers only allow sound after the user has clicked the page once
        document.addEventListener('click', getAlarmCtx);
        document.addEventListener('keydown', getAlarmCtx);

        function refreshAlarmButton() {
            const b = document.getElementById('alarm-toggle');
            if (b) b.textContent = alarmEnabled ? 'Alarm: ON' : 'Alarm: OFF';
        }
        function toggleAlarm() {
            alarmEnabled = !alarmEnabled;
            localStorage.setItem('alarmEnabled', alarmEnabled);
            refreshAlarmButton();
        }

        function showAlarmBanner() {
            let banner = document.getElementById('alarm-banner');
            if (!banner) {
                banner = document.createElement('div');
                banner.id = 'alarm-banner';
                banner.textContent = 'ALERT: Unknown person detected!';
                banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;padding:14px;' +
                    'text-align:center;font-weight:700;color:white;background:#dc2626;' +
                    'box-shadow:0 4px 20px rgba(220,38,38,.6);';
                document.body.appendChild(banner);
            }
            banner.style.display = 'block';
            clearTimeout(banner._t);
            banner._t = setTimeout(() => { banner.style.display = 'none'; }, 4000);
        }

        function triggerAlarm(force = false) {
            if (!alarmEnabled && !force) return;
            showAlarmBanner();
            if (alarmPlaying) return;

            const ctx = getAlarmCtx();
            if (!ctx || ctx.state !== 'running') {
                showToast('Click anywhere on the page once to allow alarm sound', 'error');
                return;
            }
            alarmPlaying = true;
            const dur = 3;                       // seconds
            const t0 = ctx.currentTime;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'square';
            for (let i = 0; i < dur / 0.4; i++) {   // siren: two tones alternating
                osc.frequency.setValueAtTime(i % 2 ? 660 : 880, t0 + i * 0.4);
            }
            gain.gain.setValueAtTime(0.25, t0);
            gain.gain.linearRampToValueAtTime(0, t0 + dur);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.onended = () => { alarmPlaying = false; };
            osc.start(t0);
            osc.stop(t0 + dur);
        }
        document.addEventListener('DOMContentLoaded', refreshAlarmButton);

'''

BUTTONS = '''<div style="display:flex;gap:var(--spacing-md);margin-bottom:var(--spacing-lg);">
            <button id="alarm-toggle" onclick="toggleAlarm()"
                style="background:var(--bg-tertiary);color:var(--text-primary);padding:8px 16px;border-radius:8px;border:none;font-weight:500;cursor:pointer;">Alarm: ON</button>
            <button onclick="triggerAlarm(true)"
                style="background:var(--bg-tertiary);color:var(--text-primary);padding:8px 16px;border-radius:8px;border:none;font-weight:500;cursor:pointer;">Test alarm</button>
        </div>

        '''

# 1. alarm code
if "function triggerAlarm" not in src:
    new, n = re.subn(r"^([ \t]*)function showToast\(", lambda m: ALARM_JS + m.group(0), src, count=1, flags=re.M)
    src = new; done("1 alarm code added", n == 1)
else:
    done("1 alarm code (already done)", True)

# 2. call it when an unknown-person alert arrives
if "triggerAlarm();" not in src:
    new, n = re.subn(r"showToast\('Unknown person detected!', 'error'\);",
                     "showToast('Unknown person detected!', 'error');\n                    triggerAlarm();",
                     src, count=1)
    src = new; done("2 alarm triggered on intrusion_alert", n == 1)
else:
    done("2 trigger (already done)", True)

# 3. buttons
if 'id="alarm-toggle"' not in src:
    new, n = re.subn(r'<div class="camera-grid">', lambda m: BUTTONS + m.group(0), src, count=1)
    src = new; done("3 Alarm ON/OFF and Test buttons added", n == 1)
else:
    done("3 buttons (already done)", True)

# 4. small crash fix: 'gate-video' element does not exist on this page
if "const _gv = " not in src:
    new, n = re.subn(r"document\.getElementById\('gate-video'\)\.style\.display = 'none';",
                     "const _gv = document.getElementById('gate-video'); if (_gv) _gv.style.display = 'none';",
                     src, count=1)
    src = new; done("4 small crash fix on camera disconnect", n == 1)
else:
    done("4 crash fix (already done)", True)

path.write_text(src, encoding="utf-8")
print()
for name, ok in report:
    print(("OK      " if ok else "FAILED  ") + name)
print("\nAll edits applied." if all(ok for _, ok in report) else "\nSome edits FAILED - send me this output.")
print("Now refresh the dashboard page with Ctrl+Shift+R.")