"""
Capture the four Fig. 6 states from the running LungLens app.

Drives the machine's installed Chrome in headless mode over the DevTools
Protocol. Playwright and Selenium are not installed here, and CDP needs nothing
beyond `websockets`, which the Gradio stack already pulls in. The win over a
plain screenshot tool is deviceScaleFactor: every shot is rendered at 2x (3x for
the close-up), so the text survives being scaled into a two-column figure.

Capture rules: one window size for every shot (800x1200 CSS), light mode via
Gradio's ?__theme=light, no browser chrome (CDP captures the page only), PNG.

The upload hands the app the real test file, fetched through Gradio's own
/gradio_api/file= route, so it classifies exactly the pixels the offline scoring
did rather than a re-encoded copy.

Usage:  python capture_ui.py [A] [B] [C] [D]     (default: all four)
"""
import os, sys, json, time, base64, socket, subprocess, tempfile, asyncio, shutil
import urllib.request
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ui_shots")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "capture.log"), "a", encoding="utf-8")

APP = "http://127.0.0.1:7861/?__theme=light"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W, H = 800, 1200
CALL_TIMEOUT = 120

CASE_CONFIDENT = (r"C:\Users\evanl\.cache\kagglehub\datasets\tawsifurrahman"
                  r"\covid19-radiography-database\versions\5"
                  r"\COVID-19_Radiography_Dataset\COVID\images\COVID-3348.png")
CASE_UNCERTAIN = (r"C:\Users\evanl\.cache\kagglehub\datasets\raddar"
                  r"\tuberculosis-chest-xrays-shenzhen\versions\1\images\images"
                  r"\CHNCXR_0619_1.png")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    async def _call(self, method, params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def call(self, method, **params):
        return await asyncio.wait_for(self._call(method, params), CALL_TIMEOUT)

    async def js(self, expr, awaitp=False):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=awaitp,
                            returnByValue=True, userGesture=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("text", "js error"))
        return r["result"].get("value")

    async def shot(self, path, clip, scale):
        r = await self.call("Page.captureScreenshot", format="png",
                            captureBeyondViewport=True, clip=dict(clip, scale=scale))
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))
        log("  saved", os.path.basename(path), os.path.getsize(path), "bytes")


UPLOAD_JS = """(async () => {
  const p = %s;
  const blob = await (await fetch("/gradio_api/file=" + p)).blob();
  const dt = new DataTransfer();
  dt.items.add(new File([blob], "case.png", {type: "image/png"}));
  const inp = document.querySelector('input[type=file]');
  inp.files = dt.files;
  inp.dispatchEvent(new Event('change', {bubbles: true}));
  await new Promise(r => setTimeout(r, 3000));
  const btn = [...document.querySelectorAll('button')]
      .find(b => b.textContent.trim() === 'Analyze');
  if (!btn) return "NO_BUTTON";
  btn.click();
  for (let i = 0; i < 160; i++) {
    await new Promise(r => setTimeout(r, 500));
    const t = document.body.innerText;
    if (/Prediction:/.test(t)) { await new Promise(r => setTimeout(r, 1200)); return t; }
  }
  return "TIMEOUT";
})()"""

# One call that returns every rectangle we might clip to, and never throws.
RECTS_JS = """(() => {
  const R = e => { if (!e) return null; const r = e.getBoundingClientRect();
    return {x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height,
            bottom: r.bottom + scrollY}; };
  const panels = [...document.querySelectorAll('.custom-panel')];
  const parts = [...document.querySelectorAll('.ll-header, .tabs, .custom-panel')];
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const e of parts) { const r = e.getBoundingClientRect();
    x0 = Math.min(x0, r.x + scrollX); y0 = Math.min(y0, r.y + scrollY);
    x1 = Math.max(x1, r.right + scrollX); y1 = Math.max(y1, r.bottom + scrollY); }
  const pad = 14;
  const content = {x: Math.max(0, x0 - pad), y: Math.max(0, y0 - pad),
                   width: Math.min(innerWidth, x1 - x0 + 2 * pad),
                   height: y1 - y0 + 2 * pad};
  return {content: content,
          results: R(panels[1] || null),
          label: R(document.querySelector('.conf-label')),
          // The Results column holds several markdown blocks ("Results" heading
          // included). Pick the one that actually carries the generated summary,
          // not whichever comes first in the DOM - selecting by position gave a
          // clip with negative height, which hangs Page.captureScreenshot.
          md: R([...document.querySelectorAll('.md, .prose')]
                  .filter(e => /Prediction:/.test(e.innerText)).pop() || null),
          scrollH: document.documentElement.scrollHeight};
})()"""


async def load(c):
    await c.call("Page.navigate", url=APP)
    for _ in range(160):
        await asyncio.sleep(0.5)
        if await c.js("!!document.querySelector('input[type=file]')"):
            await asyncio.sleep(2.0)
            return
    raise RuntimeError("app never finished loading")


async def run(c, panels):
    texts = {}

    if "A" in panels:
        log("Panel A - upload state")
        await load(c)
        r = await c.js(RECTS_JS)
        log("   rects:", json.dumps(r)[:300])
        await c.shot(os.path.join(OUT, "A_upload.png"), r["content"], 2)

    if "B" in panels or "C" in panels:
        log("Panel B - confident correct prediction")
        await load(c)
        t = await c.js(UPLOAD_JS % json.dumps(CASE_CONFIDENT), awaitp=True)
        texts["confident"] = t
        log("   ", " ".join(t.split())[:160])
        await asyncio.sleep(1.5)
        r = await c.js(RECTS_JS)
        log("   rects:", json.dumps(r)[:400])
        if "B" in panels:
            await c.shot(os.path.join(OUT, "B_result.png"), r["content"], 2)
        if "C" in panels:
            log("Panel C - clinical summary close-up")
            lab, md, res = r.get("label"), r.get("md"), r.get("results")
            # Tight on the generated description only - the class-confidence card
            # above it is already legible in panel B, and including it here makes
            # the crop tall and narrow instead of readable at figure scale.
            if md:
                clip = {"x": md["x"] - 12, "y": md["y"] - 12,
                        "width": md["width"] + 24, "height": md["height"] + 24}
            else:
                clip = None
            # A non-positive height hangs Page.captureScreenshot outright, so
            # never send one: fall back to the lower part of the Results card.
            if clip is None or clip["height"] <= 0 or clip["width"] <= 0:
                log("   (fallback clip: usable label/summary rects not found)")
                clip = {"x": res["x"], "y": res["y"] + res["height"] * 0.42,
                        "width": res["width"], "height": res["height"] * 0.58}
            log("   clip:", json.dumps(clip))
            await c.shot(os.path.join(OUT, "C_summary.png"), clip, 3)

    if "D" in panels:
        log("Panel D - uncertain response")
        await load(c)
        t = await c.js(UPLOAD_JS % json.dumps(CASE_UNCERTAIN), awaitp=True)
        texts["uncertain"] = t
        log("   ", " ".join(t.split())[:260])
        await asyncio.sleep(1.5)
        r = await c.js(RECTS_JS)
        log("   rects:", json.dumps(r)[:400])
        await c.shot(os.path.join(OUT, "D_uncertain.png"), r["content"], 2)
        # Also a tight crop of the Uncertain response. The full window shrinks to
        # about an inch across in a four-panel figure, at which point its text is
        # unreadable; the crop is what actually goes into Fig. 6.
        md = r.get("md")
        if md:
            await c.shot(os.path.join(OUT, "D_summary.png"),
                         {"x": md["x"] - 12, "y": md["y"] - 12,
                          "width": md["width"] + 24, "height": md["height"] + 24}, 3)

    if texts:
        old = {}
        p = os.path.join(OUT, "texts.json")
        if os.path.exists(p):
            old = json.load(open(p, encoding="utf-8"))
        old.update(texts)
        json.dump(old, open(p, "w", encoding="utf-8"), indent=2)


async def main():
    panels = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "D"]
    port = free_port()
    prof = tempfile.mkdtemp(prefix="lunglens_shot_")
    proc = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
        "--disable-extensions", "--hide-scrollbars", "--force-color-profile=srgb",
        f"--window-size={W},{H}", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"\n=== capture {time.strftime('%H:%M:%S')} panels={panels} chrome pid={proc.pid} ===")
    try:
        ws_url = None
        for _ in range(80):
            try:
                data = json.loads(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list", timeout=1).read())
                pages = [t for t in data if t["type"] == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not ws_url:
            raise RuntimeError("Chrome DevTools endpoint never came up")
        async with websockets.connect(ws_url, max_size=400_000_000,
                                      close_timeout=5) as ws:
            c = CDP(ws)
            await c.call("Page.enable")
            await c.call("Runtime.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H,
                         deviceScaleFactor=2, mobile=False)
            await run(c, panels)
    except Exception as e:
        log("ERROR:", type(e).__name__, e)
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
        log("chrome closed")


asyncio.run(main())
log("done ->" + OUT)
