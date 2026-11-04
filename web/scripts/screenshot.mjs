// Records the portrait map for the repository README: the `?screenshot` page (src/screenshot.ts) in headless
// Chrome, stepped frame by frame through one wave cycle into an animated WebP that loops without a seam, or
// with --still into a PNG. It needs Node 22 or later, Chrome or chrome-headless-shell, and ffmpeg with libwebp.
//
//   weclone-cli server --demo en --port 5180
//   WECLONE_PASSWORD=demo node web/scripts/screenshot.mjs http://127.0.0.1:5180 atlas-light-en.webp
//
// --locale (en, zh-CN) and --theme (light, dark) pick the page, --size and --scale the picture: the README
// shows the default 1280x840 at 880 wide. CHROME names the browser when none is on PATH.
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { parseArgs } from 'node:util';

/** Time between frames in ms: about 29 frames a second keep the wave smooth. */
const FRAME_MS = 35;

const { values: options, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    locale: { type: 'string', default: 'en' },
    theme: { type: 'string', default: 'light' },
    size: { type: 'string', default: '1280x840' },
    scale: { type: 'string', default: '1.5' },
    still: { type: 'boolean', default: false },
  },
});
const [address, output] = positionals;
const [width, height] = options.size.split('x').map(Number);
if (!address || !output || !width || !height) {
  console.error(
    'Usage: node web/scripts/screenshot.mjs <server address> <output.webp|.png>' +
      ' [--locale en] [--theme light] [--size 1280x840] [--scale 1.5] [--still]',
  );
  process.exit(2);
}

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const names = ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome-headless-shell'];
  const dirs = (process.env.PATH ?? '').split(delimiter).filter(Boolean);
  return names.flatMap((name) => dirs.map((dir) => join(dir, name))).find((path) => existsSync(path));
}

/** A DevTools connection to the browser's first page. */
async function connect(chrome, profile) {
  const portFile = join(profile, 'DevToolsActivePort');
  for (let i = 0; !existsSync(portFile); i++) {
    if (chrome.exitCode !== null || i > 150) throw new Error('Chrome did not start.');
    await sleep(100);
  }
  const port = readFileSync(portFile, 'utf8').split('\n')[0];
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const socket = new WebSocket(targets.find((target) => target.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = () => reject(new Error('Could not connect to Chrome.'));
  });
  const calls = new Map();
  const waits = new Map();
  let lastId = 0;
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.id) calls.get(message.id)?.(message);
    else waits.get(message.method)?.(message.params);
  };
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++lastId;
      calls.set(id, ({ result, error }) => {
        calls.delete(id);
        if (error) reject(new Error(`${method}: ${error.message}`));
        else resolve(result);
      });
      socket.send(JSON.stringify({ id, method, params }));
    });
  const event = (method) =>
    new Promise((resolve) =>
      waits.set(method, (params) => {
        waits.delete(method);
        resolve(params);
      }),
    );
  return {
    send,
    close: () => socket.close(),
    async goto(url) {
      const loaded = event('Page.loadEventFired');
      await send('Page.navigate', { url });
      await loaded;
    },
    async evaluate(expression) {
      const { result, exceptionDetails } = await send('Runtime.evaluate', {
        expression,
        awaitPromise: true,
        returnByValue: true,
      });
      if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
      return result.value;
    },
    async screenshot() {
      const { data } = await send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: true });
      return Buffer.from(data, 'base64');
    },
  };
}

async function record(page) {
  const url = new URL('/?screenshot', address).href;
  await page.send('Page.enable');
  await page.send('Emulation.setDeviceMetricsOverride', {
    width,
    height,
    deviceScaleFactor: Number(options.scale),
    mobile: false,
  });
  const saved = {
    'weclone.locale': options.locale,
    'weclone.theme': options.theme,
    'weclone.profile.grouping': 'dimension',
  };
  await page.send('Page.addScriptToEvaluateOnNewDocument', {
    source: `for (const [key, value] of Object.entries(${JSON.stringify(saved)})) localStorage.setItem(key, value);`,
  });
  await page.goto(url);
  const password = process.env.WECLONE_PASSWORD;
  if (password) {
    const status = await page.evaluate(
      `fetch('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-WeClone-Request': '1' },
        body: ${JSON.stringify(JSON.stringify({ password }))} }).then((response) => response.status)`,
    );
    if (status !== 200) throw new Error(`Signing in failed with HTTP ${status}.`);
    await page.goto(url);
  }
  const ready = await page.evaluate(`new Promise((resolve) => {
    const start = performance.now();
    const check = () => window.atlasCapture?.ready() ? resolve(true)
      : performance.now() - start > 30000 ? resolve(false) : setTimeout(check, 100);
    check();
  })`);
  if (!ready) {
    throw new Error(`The map did not show at ${url}: is the server running, and does it need WECLONE_PASSWORD?`);
  }
  // step(0) stops the page's own frame loop, so that only the steps move the map from here on.
  const { frames, step } = await page.evaluate(`(() => {
    const capture = window.atlasCapture;
    capture.step(0);
    for (let left = capture.warmUp(); left > 0; left -= ${FRAME_MS}) capture.step(Math.min(${FRAME_MS}, left));
    const frames = Math.round(capture.period() / ${FRAME_MS});
    return { frames, step: capture.period() / frames };
  })()`);
  if (options.still) {
    writeFileSync(output, await page.screenshot());
    return;
  }
  const input = `-v error -y -f image2pipe -c:v png -framerate ${1000 / step} -i -`;
  const encoding = '-c:v libwebp_anim -pix_fmt yuv420p -quality 90 -loop 0';
  const ffmpeg = spawn('ffmpeg', [...`${input} ${encoding}`.split(' '), output], {
    stdio: ['pipe', 'inherit', 'inherit'],
  });
  const encoded = once(ffmpeg, 'exit');
  // A failed ffmpeg reports its own error and exit code.
  ffmpeg.stdin.on('error', () => {});
  // The frame after the last is the first again.
  for (let i = 0; i < frames && ffmpeg.exitCode === null; i++) {
    if (i) await page.evaluate(`window.atlasCapture.step(${step})`);
    if (!ffmpeg.stdin.write(await page.screenshot())) await Promise.race([once(ffmpeg.stdin, 'drain'), encoded]);
  }
  ffmpeg.stdin.end();
  const [code] = await encoded;
  if (code !== 0) throw new Error(`ffmpeg exited with ${code}.`);
}

const chromePath = findChrome();
if (!chromePath) {
  console.error('No Chrome found: set CHROME to a Chrome or chrome-headless-shell binary.');
  process.exit(1);
}
if (!options.still && spawnSync('ffmpeg', ['-hide_banner', '-h', 'encoder=libwebp_anim']).status !== 0) {
  console.error('Recording a WebP needs ffmpeg with libwebp, or pass --still for a PNG.');
  process.exit(1);
}
const profile = mkdtempSync(join(tmpdir(), 'weclone-screenshot-'));
const flags = '--headless --remote-debugging-port=0 --hide-scrollbars --no-first-run --no-default-browser-check';
const chrome = spawn(chromePath, [...flags.split(' '), `--user-data-dir=${profile}`, 'about:blank'], {
  stdio: 'ignore',
});
let page;
try {
  page = await connect(chrome, profile);
  await record(page);
  console.log(`Wrote ${output}`);
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
} finally {
  page?.close();
  chrome.kill();
  if (chrome.exitCode === null) await once(chrome, 'exit');
  rmSync(profile, { recursive: true, force: true });
}
