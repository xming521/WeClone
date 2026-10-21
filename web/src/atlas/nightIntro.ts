// The map's first-visit intro (design: WC-exp/design, "从夜到昼"; schedule and particles in
// nightIntroPlan.ts). On the first visit the page turns to night as soon as the map mounts. Once the
// map has its first layout the intro plays above it and hands over the way the entrance does
// (entrance.ts): by the last frame the sky and the stars have faded, and the map's nodes, lines and
// labels are drawn with the map's own primitives (paint.ts), so the map takes over in one frame.
import { api } from '../api';
import { DIMENSIONS, ROOT_COLOR } from '../data';
import type { ProfileModel } from '../data';
import type { EntranceConfig, EntranceItem } from './entrance';
import { hexToRgb } from './graphStyle';
import { canvasDpr, drawEdge, drawLabel, drawNode, type Transform } from './paint';
import {
  DAY_LEAD,
  SPIN,
  dayAt,
  dayProgress,
  easeInOutCubic,
  easeOutCubic,
  layoutWords,
  planParticles,
  progress,
  sampleGlyphs,
  smoothstep,
  spinWindow,
  timeline,
  titleLine,
  type GlyphRaster,
  type IntroSentence,
  type NodeTiming,
  type ParticlePlan,
  type Timeline,
  type WordSpan,
} from './nightIntroPlan';
import './nightIntro.css';

const STORAGE_KEY = 'weclone.atlas.intro.v1';
const WORD_FONT = "'Noto Sans SC', 'Noto Sans CJK SC', 'Noto Sans', system-ui, sans-serif";
const WORD_COLOR = '#dfe9e6';
const DATE_COLOR = '#86a39c';
/** Particles: one per message, within these bounds; small touch screens get a fixed number. */
const MIN_PARTICLES = 500;
const MAX_PARTICLES = 60000;
const TOUCH_PARTICLES = 8000;
/** How long the intro waits for its numbers and lines before it plays without them. */
const DATA_WAIT_MS = 2500;
const FONT_WAIT_MS = 1200;
/** A map that has not laid out by then gets the usual entrance instead. */
const LAYOUT_WAIT_MS = 10000;
/** A skipped intro fades out over this long; keep in step with nightIntro.css. */
const LEAVE_MS = 400;
const CHROME_FADE_MS = 650;

type IntroData = {
  message_total: number | null;
  first_time: string | null;
  last_time: string | null;
  sentences: IntroSentence[];
};

// ───────────────────────── shaders ─────────────────────────

const DAY_GLSL = `
uniform vec2 uCenter;
uniform vec4 uDay; // global progress, centre lead, max distance, night variant
// Daylight: one even dawn over the whole sky, a little ahead near the root.
float dayAt(vec2 p) {
  if (uDay.w < 0.5) return 1.0;
  float d = length(p - uCenter) / uDay.z;
  float n = (sin(p.x * 0.0061 + p.y * 0.0042) + sin(p.y * 0.0093 - p.x * 0.0031)) * 0.02;
  return clamp(uDay.x * (1.0 + uDay.y) - uDay.y * d + n * (1.0 - uDay.x), 0.0, 1.0);
}`;

const POINT_VERT = `
precision highp float;
attribute vec4 aA; // homeR, homeTheta, clusterX, clusterY
attribute vec4 aB; // offsetX, offsetY, nodeX, nodeY
attribute vec4 aC; // depart, arrive, collapse, collapse duration
attribute vec4 aD; // seed, size, palette slot, base alpha
attribute vec2 aE; // unwrapped cluster angle, orbit speed
attribute vec4 aF; // opening words: glyph x, y, break-up start (<0: not a word), arrival | star lights up
attribute vec2 aG; // opening words: dot opacity and diameter while still part of a line

uniform float uTime;
uniform vec2 uViewport;
uniform float uDpr;
uniform float uSpin;
uniform float uSpinBoost;
uniform float uSpinFalloff;
uniform vec2 uSpinWindow;
uniform float uDrift;
uniform float uStreak;
uniform float uGlobalAlpha;
uniform float uSizeScale;
uniform vec3 uPalette[7];
uniform vec3 uPaletteNight[7];
uniform vec3 uBase;
uniform vec3 uBaseNight;
uniform float uBaseAlpha;
uniform float uArriveAlpha;
uniform float uArriveSize;
uniform vec4 uTextMask;
uniform float uTextMaskK;
uniform float uMaxPoint;
uniform float uArms;
uniform float uTwinkle;
uniform float uWords;
uniform vec3 uWordColor;
${DAY_GLSL}

varying vec4 vColor;
varying vec2 vDir;
varying float vHalf;
varying float vRadius;
varying float vSprite;

float spinPhase(float t) {
  float L = uSpinWindow.y - uSpinWindow.x;
  float x = clamp((t - uSpinWindow.x) / L, 0.0, 1.0);
  float extra = L * (x * x * x - 0.5 * x * x * x * x) + max(0.0, t - uSpinWindow.y);
  return t + uSpinBoost * extra;
}

// accelerate towards the target (speed peaks ~90% of the way), then a short settle
float easeArrive(float u) {
  const float p = 1.9;
  const float w1 = 0.88;
  const float u1 = 0.8745;
  if (u < u1) return w1 * pow(u / u1, p);
  float s = (u - u1) / (1.0 - u1);
  return w1 + (1.0 - w1) * (1.0 - (1.0 - s) * (1.0 - s));
}

// smooth, divergence-light flow field: large slow eddies plus a finer ripple
vec2 flowField(vec2 p, float t) {
  vec2 big = vec2(
    sin(p.y * 0.0047 + t * 0.9 + 1.7) + 0.6 * sin((p.x + p.y) * 0.0031 - t * 0.6),
    cos(p.x * 0.0043 - t * 0.8 + 0.4) + 0.6 * cos((p.x - p.y) * 0.0036 + t * 0.7));
  vec2 fine = vec2(
    sin(p.y * 0.013 + t * 1.6),
    cos(p.x * 0.012 - t * 1.4));
  return big * 2.6 + fine * 0.8;
}

vec2 position(float t, out float w, out float k) {
  float r0 = aA.x;
  float seed = aD.x;
  float omega = uSpin * (0.3 + 0.7 * exp(-r0 / uSpinFalloff));
  float th = aA.y + omega * spinPhase(t);
  vec2 drift = uDrift * vec2(
    sin(t * (0.45 + 0.35 * seed) + seed * 31.0) + 0.5 * sin(t * 1.27 + seed * 7.0),
    cos(t * (0.40 + 0.45 * seed) + seed * 13.0) + 0.5 * cos(t * 1.13 + seed * 3.0));
  w = 0.0;
  k = 0.0;
  vec2 p;
  if (aC.z < 0.0) {
    float f = max(0.0, t - aC.x);
    float r = r0 + 16.0 * f * f;
    p = uCenter + r * vec2(cos(th), sin(th)) + drift;
  } else {
    float u = clamp((t - aC.x) / max(0.001, aC.y - aC.x), 0.0, 1.0);
    w = easeArrive(u);
    float r = mix(r0, length(aA.zw - uCenter), w);
    float a = mix(th, aE.x, w);
    p = uCenter + r * vec2(cos(a), sin(a));
    p += flowField(p, t) * (9.0 * sin(3.14159265 * w) * (1.0 - 0.6 * w));
    p += drift * (1.0 - w);
    float rho = length(aB.xy);
    float oa = aE.y * (1.25 - 0.5 * clamp(rho / 40.0, 0.0, 1.0)) * (t - aC.x);
    float co = cos(oa);
    float so = sin(oa);
    p += vec2(co * aB.x - so * aB.y, so * aB.x + co * aB.y) * w;
    k = clamp((t - aC.z) / max(0.001, aC.w), 0.0, 1.0);
    p = mix(p, aB.zw, k * k);
  }
  // opening words: a dot sits on its glyph, then lifts off and drifts to its place in the sky
  if (uWords > 0.5 && aF.z >= 0.0) {
    float kw = clamp((t - aF.z) / max(0.001, aF.w - aF.z), 0.0, 1.0);
    float ew = kw < 0.5 ? 4.0 * kw * kw * kw : 1.0 - pow(-2.0 * kw + 2.0, 3.0) * 0.5;
    vec2 q = mix(aF.xy, p, ew);
    q += (flowField(q, t) * 4.0 + vec2(0.0, -12.0)) * sin(3.14159265 * kw);
    p = q;
  }
  return p;
}

void main() {
  float w;
  float k;
  float w0;
  float k0;
  vec2 p = position(uTime, w, k);
  vec2 v = p - position(uTime - 0.0166667, w0, k0);
  float speed = length(v);

  int si = int(aD.z + 0.5);
  vec3 dimDay = uPalette[0];
  vec3 dimNight = uPaletteNight[0];
  for (int i = 1; i < 7; i++) {
    if (i == si) {
      dimDay = uPalette[i];
      dimNight = uPaletteNight[i];
    }
  }
  // opening words: how much this particle is still a dot of a sentence, and whether it is lit yet
  float textness = 0.0;
  float lit = 1.0;
  if (uWords > 0.5) {
    if (aF.z >= 0.0) {
      float kw = clamp((uTime - aF.z) / max(0.001, aF.w - aF.z), 0.0, 1.0);
      lit = smoothstep(aF.z - 0.02, aF.z + 0.02, uTime);
      textness = 1.0 - smoothstep(0.0, 0.65, kw);
    } else {
      lit = aF.w <= 0.0 ? 1.0 : smoothstep(aF.w, aF.w + 0.45, uTime);
    }
  }
  float sky = dayAt(p);
  float day = smoothstep(0.35, 0.75, sky);
  float near = smoothstep(0.28, 0.96, w);
  vec3 base = mix(uBaseNight, uBase, day);
  // the closer a particle gets, the more it takes the node's own (saturated) colour, day or night
  vec3 target = mix(dimNight, dimDay, max(day, 0.75 * near));
  float speedN = clamp(speed / 7.0, 0.0, 1.0);
  vec3 col = mix(mix(base, target, near), uWordColor, textness);

  float alpha = aD.w * uBaseAlpha * mix(1.0, uArriveAlpha, near) * (1.0 + 0.3 * speedN);
  // two soft spiral arms that slowly turn: makes the vortex readable before anything moves
  vec2 rel = p - uCenter;
  float rr = max(length(rel), 40.0);
  float arm = 0.5 + 0.5 * cos(2.0 * (atan(rel.y, rel.x) - uSpin * 0.7 * uTime - log(rr / 70.0) * 1.7));
  alpha *= mix(1.0, 0.45 + 1.1 * arm, uArms * (1.0 - near) * (1.0 - textness));
  // stars twinkle a little while it is night
  alpha *= 1.0 - uTwinkle * (1.0 - sky) * (1.0 - textness) * (0.5 + 0.5 * sin(uTime * (1.3 + 2.1 * aD.x) + aD.x * 40.0));
  alpha = mix(alpha, aG.x, textness) * lit;
  if (aC.z < 0.0) alpha *= 1.0 - smoothstep(aC.x, aC.y, uTime);
  alpha *= 1.0 - smoothstep(0.55, 1.0, k);
  vec2 q = (p - uTextMask.xy) / uTextMask.zw;
  alpha *= 1.0 - uTextMaskK * 0.72 * (1.0 - smoothstep(0.5, 1.05, length(q)));
  alpha *= uGlobalAlpha;

  float size = mix(aD.y * uSizeScale * mix(1.0, uArriveSize, near) * mix(1.0, 0.4, k), aG.y, textness);
  float radius = 0.5 * size * uDpr;
  float flying = aC.z < 0.0 ? 0.0 : step(aC.x, uTime) * (1.0 - step(aC.y + 0.05, uTime));
  float len = flying * min(max(0.0, speed - 1.4) * uStreak * uDpr, 16.0 * uDpr);
  alpha *= pow(2.0 * radius / (2.0 * radius + len), 0.5);
  if (radius < 0.65) {
    alpha *= (radius / 0.65) * (radius / 0.65);
    radius = 0.65;
  }

  vDir = speed > 0.0001 ? v / speed : vec2(1.0, 0.0);
  vHalf = 0.5 * len;
  vRadius = radius;
  vSprite = min(2.0 * radius + len + 2.0, uMaxPoint);
  vColor = vec4(col, clamp(alpha, 0.0, 1.0));
  gl_PointSize = vSprite;
  gl_Position = vec4(p.x / uViewport.x * 2.0 - 1.0, 1.0 - p.y / uViewport.y * 2.0, 0.0, 1.0);
  if (vColor.a < 0.003) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
}`;

// A round dot, drawn out into a capsule along its motion while it flies.
const POINT_FRAG = `
precision mediump float;
varying vec4 vColor;
varying vec2 vDir;
varying float vHalf;
varying float vRadius;
varying float vSprite;
void main() {
  vec2 q = (gl_PointCoord - 0.5) * vSprite;
  float s = clamp(dot(q, vDir), -vHalf, vHalf);
  float d = length(q - vDir * s);
  float a = vColor.a * (1.0 - smoothstep(vRadius - 0.75, vRadius + 0.75, d));
  if (a < 0.002) discard;
  gl_FragColor = vec4(vColor.rgb * a, a);
}`;

const SKY_VERT = `
precision highp float;
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const SKY_FRAG = `
precision highp float;
varying vec2 vUv;
uniform vec2 uViewport;
uniform float uAlpha;
${DAY_GLSL}
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
float glow(vec2 uv, vec2 at, vec2 radius) {
  vec2 d = (uv - at) / radius;
  return exp(-dot(d, d) * 2.2);
}
void main() {
  vec2 px = vec2(vUv.x, 1.0 - vUv.y) * uViewport;
  vec2 uv = px / uViewport;
  vec3 night = mix(vec3(0.043, 0.075, 0.086), vec3(0.071, 0.110, 0.122), uv.y);
  vec3 glows = vec3(0.055, 0.160, 0.140) * 0.55 * glow(uv, vec2(0.23, 0.24), vec2(0.42, 0.48))
    + vec3(0.060, 0.090, 0.180) * 0.55 * glow(uv, vec2(0.78, 0.72), vec2(0.40, 0.46))
    + vec3(0.140, 0.100, 0.060) * 0.35 * glow(uv, vec2(0.66, 0.18), vec2(0.30, 0.34));
  float d = dayAt(px);
  // night → blue hour → dawn → pale paper, instead of a grey cross-fade
  vec3 col = night + glows;
  float centre = 1.0 - clamp(length(px - uCenter) / uDay.z, 0.0, 1.0);
  col = mix(col, vec3(0.118, 0.205, 0.255) + glows * 0.8, smoothstep(0.0, 0.3, d));
  col = mix(col, vec3(0.600, 0.680, 0.700) + glows * 0.35, smoothstep(0.24, 0.56, d));
  col = mix(col, vec3(0.905, 0.910, 0.890), smoothstep(0.5, 0.8, d));
  // first light: a faint warm glow around the root while the sky turns
  col += vec3(0.120, 0.072, 0.024) * centre * centre * smoothstep(0.2, 0.5, d) * (1.0 - smoothstep(0.55, 0.9, d));
  col += (hash(gl_FragCoord.xy) - 0.5) / 160.0;
  float a = (1.0 - smoothstep(0.78, 1.0, d)) * uAlpha;
  gl_FragColor = vec4(col * a, a);
}`;

// ───────────────────────── sky and stars (WebGL) ─────────────────────────

type Rgb = [number, number, number];
const rgb01 = (hex: string) => hexToRgb(hex).map((value) => value / 255) as Rgb;
const mixRgb = (a: Rgb, b: Rgb, t: number) => a.map((value, i) => value + (b[i] - value) * t) as Rgb;
const WHITE: Rgb = [1, 1, 1];

type StarFrame = { time: number; globalAlpha: number; textMaskK: number; dayProgress: number };

type StarScene = {
  plan: ParticlePlan;
  timeline: Timeline;
  width: number;
  height: number;
  maxDistance: number;
  textMask: [number, number, number, number];
};

type StarLayer = {
  load: (scene: StarScene) => void;
  render: (frame: StarFrame) => void;
  destroy: () => void;
};

const POINT_ATTRIBUTES = [
  ['aA', 4, 'a'],
  ['aB', 4, 'b'],
  ['aC', 4, 'c'],
  ['aD', 4, 'd'],
  ['aE', 2, 'e'],
  ['aF', 4, 'f'],
  ['aG', 2, 'g'],
] as const;

/** Compiles the sky and the stars; null when WebGL or the shaders are unavailable. */
function createStarLayer(canvas: HTMLCanvasElement, dpr: number): StarLayer | null {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: 'high-performance',
  });
  if (!gl) return null;
  const compile = (vertex: string, fragment: string) => {
    const shader = (type: number, source: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, source);
      gl.compileShader(s);
      return s;
    };
    const v = shader(gl.VERTEX_SHADER, vertex);
    const f = shader(gl.FRAGMENT_SHADER, fragment);
    const program = gl.createProgram()!;
    gl.attachShader(program, v);
    gl.attachShader(program, f);
    gl.linkProgram(program);
    gl.deleteShader(v);
    gl.deleteShader(f);
    if (gl.getProgramParameter(program, gl.LINK_STATUS)) return program;
    gl.deleteProgram(program);
    return null;
  };
  const points = compile(POINT_VERT, POINT_FRAG);
  const sky = compile(SKY_VERT, SKY_FRAG);
  const release = () => {
    gl.deleteProgram(points);
    gl.deleteProgram(sky);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  };
  if (!points || !sky) {
    release();
    return null;
  }
  const buffers: WebGLBuffer[] = [];
  const pointAttributes = POINT_ATTRIBUTES.map(([name, size, key]) => ({
    location: gl.getAttribLocation(points, name),
    size,
    key,
    buffer: null as WebGLBuffer | null,
  }));
  const skyLocation = gl.getAttribLocation(sky, 'aPos');
  const skyBuffer = gl.createBuffer()!;
  buffers.push(skyBuffer);
  gl.bindBuffer(gl.ARRAY_BUFFER, skyBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const at = (program: WebGLProgram, name: string) => gl.getUniformLocation(program, name);
  let count = 0;
  let pointTime: WebGLUniformLocation | null = null;
  let pointAlpha: WebGLUniformLocation | null = null;
  let pointMask: WebGLUniformLocation | null = null;
  let pointDay: WebGLUniformLocation | null = null;
  let skyDay: WebGLUniformLocation | null = null;
  let day: [number, number, number, number] = [0, DAY_LEAD, 1, 1];

  return {
    load({ plan, timeline: T, width, height, maxDistance, textMask }) {
      for (const attribute of pointAttributes) {
        const buffer = gl.createBuffer()!;
        buffers.push(buffer);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, plan[attribute.key], gl.STATIC_DRAW);
        attribute.buffer = buffer;
      }
      count = plan.count;
      day = [0, DAY_LEAD, maxDistance, 1];
      const palette = [
        ...DIMENSIONS.map((entry) => rgb01(entry.color)),
        mixRgb(rgb01(ROOT_COLOR), WHITE, 0.18),
        rgb01('#8a9894'),
      ];
      const night = palette.map((color, i) => mixRgb(color, WHITE, i === 5 ? 0.72 : 0.32));
      const pointSizes = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array;
      gl.useProgram(points);
      gl.uniform2f(at(points, 'uViewport'), width, height);
      gl.uniform1f(at(points, 'uDpr'), dpr);
      gl.uniform2f(at(points, 'uCenter'), plan.center.x, plan.center.y);
      gl.uniform1f(at(points, 'uSpin'), SPIN.rate);
      gl.uniform1f(at(points, 'uSpinBoost'), SPIN.boost);
      gl.uniform1f(at(points, 'uSpinFalloff'), SPIN.falloff);
      gl.uniform2f(at(points, 'uSpinWindow'), ...spinWindow(T));
      gl.uniform1f(at(points, 'uDrift'), 5.5);
      gl.uniform1f(at(points, 'uStreak'), 0.55);
      gl.uniform1f(at(points, 'uSizeScale'), plan.count < 15000 ? 1.15 : 1);
      gl.uniform3fv(at(points, 'uPalette'), palette.flat());
      gl.uniform3fv(at(points, 'uPaletteNight'), night.flat());
      gl.uniform3fv(at(points, 'uBase'), rgb01('#8d9a96'));
      gl.uniform3fv(at(points, 'uBaseNight'), rgb01('#dce6ec'));
      gl.uniform1f(at(points, 'uBaseAlpha'), 0.62);
      gl.uniform1f(at(points, 'uArriveAlpha'), 1.45);
      gl.uniform1f(at(points, 'uArriveSize'), 1.15);
      gl.uniform4fv(at(points, 'uTextMask'), textMask);
      gl.uniform1f(at(points, 'uMaxPoint'), Math.max(4, Math.min(64, pointSizes[1] || 64)));
      gl.uniform1f(at(points, 'uArms'), 0.55);
      gl.uniform1f(at(points, 'uTwinkle'), 0.35);
      gl.uniform1f(at(points, 'uWords'), plan.words ? 1 : 0);
      gl.uniform3fv(at(points, 'uWordColor'), rgb01(WORD_COLOR));
      pointTime = at(points, 'uTime');
      pointAlpha = at(points, 'uGlobalAlpha');
      pointMask = at(points, 'uTextMaskK');
      pointDay = at(points, 'uDay');
      gl.useProgram(sky);
      gl.uniform2f(at(sky, 'uViewport'), width, height);
      gl.uniform2f(at(sky, 'uCenter'), plan.center.x, plan.center.y);
      gl.uniform1f(at(sky, 'uAlpha'), 1);
      skyDay = at(sky, 'uDay');
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
    },
    render({ time, globalAlpha, textMaskK, dayProgress: progressOfDay }) {
      day[0] = progressOfDay;
      gl.clear(gl.COLOR_BUFFER_BIT);
      for (const attribute of pointAttributes) gl.disableVertexAttribArray(attribute.location);
      // Once the dawn has reached every corner the sky is fully transparent.
      if (progressOfDay < 1) {
        gl.useProgram(sky);
        gl.uniform4fv(skyDay, day);
        gl.bindBuffer(gl.ARRAY_BUFFER, skyBuffer);
        gl.enableVertexAttribArray(skyLocation);
        gl.vertexAttribPointer(skyLocation, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.disableVertexAttribArray(skyLocation);
      }
      if (globalAlpha <= 0 || !count) return;
      gl.useProgram(points);
      gl.uniform1f(pointTime, time);
      gl.uniform1f(pointAlpha, globalAlpha);
      gl.uniform1f(pointMask, textMaskK);
      gl.uniform4fv(pointDay, day);
      for (const attribute of pointAttributes) {
        gl.bindBuffer(gl.ARRAY_BUFFER, attribute.buffer);
        gl.enableVertexAttribArray(attribute.location);
        gl.vertexAttribPointer(attribute.location, attribute.size, gl.FLOAT, false, 0, 0);
      }
      gl.drawArrays(gl.POINTS, 0, count);
    },
    destroy() {
      for (const buffer of buffers) gl.deleteBuffer(buffer);
      release();
    },
  };
}

// ───────────────────────── opening lines (Canvas 2D) ─────────────────────────

function setWordFont(c: CanvasRenderingContext2D, span: Pick<WordSpan, 'weight' | 'size' | 'tracking'>) {
  c.font = `${span.weight} ${span.size}px ${WORD_FONT}`;
  c.letterSpacing = `${span.tracking}px`;
  c.textBaseline = 'middle';
  c.textAlign = 'left';
}

/** Measures and rasterizes the lines in CSS pixels, the unit their dots live in. */
function glyphTools(width: number, height: number) {
  const canvas = document.createElement('canvas');
  const c = canvas.getContext('2d', { willReadFrequently: true })!;
  const measure = (text: string, size: number, weight: number, tracking: number) => {
    c.font = `${weight} ${size}px ${WORD_FONT}`;
    c.letterSpacing = '0px';
    return c.measureText(text).width + tracking * Math.max(0, [...text].length - 1);
  };
  const raster = (span: WordSpan): GlyphRaster => {
    const x0 = Math.max(0, Math.floor(span.x) - 2);
    const x1 = Math.min(Math.ceil(width) - 1, Math.ceil(span.x + span.width) + 2);
    const y0 = Math.max(0, Math.floor(span.y - span.size * 0.85));
    const y1 = Math.min(Math.ceil(height) - 1, Math.ceil(span.y + span.size * 0.85));
    const w = Math.max(1, x1 - x0 + 1);
    const h = Math.max(1, y1 - y0 + 1);
    if (canvas.width < w || canvas.height < h) {
      canvas.width = Math.max(canvas.width, w);
      canvas.height = Math.max(canvas.height, h);
    }
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, w, h);
    setWordFont(c, span);
    c.fillStyle = '#ffffff';
    c.fillText(span.text, span.x - x0, span.y - y0);
    return { x0, y0, width: w, height: h, data: c.getImageData(0, 0, w, h).data };
  };
  return { measure, raster };
}

/** The lines fade in, then a soft edge erases them from the left while their dots lift off. */
function drawWords(c: CanvasRenderingContext2D, spans: WordSpan[], t: number) {
  const edge = 8;
  for (const span of spans) {
    const shown = easeOutCubic(progress(t, span.appear, span.fade));
    if (shown <= 0.002) continue;
    const sweep = span.x + ((t - span.dissolve) / span.sweep) * span.width;
    if (sweep > span.x + span.width + edge) continue;
    setWordFont(c, span);
    const color = span.role === 'date' ? DATE_COLOR : WORD_COLOR;
    c.globalAlpha = span.alpha * shown;
    if (sweep < span.x - edge) c.fillStyle = color;
    else {
      const gradient = c.createLinearGradient(sweep - edge, 0, sweep + edge, 0);
      gradient.addColorStop(0, `${color}00`);
      gradient.addColorStop(1, color);
      c.fillStyle = gradient;
    }
    c.fillText(span.text, span.x, span.y + (1 - shown) * (span.role === 'faint' ? 3 : 6));
  }
  c.globalAlpha = 1;
  c.letterSpacing = '0px';
}

// ───────────────────────── the map's last phase (Canvas 2D) ─────────────────────────

type MapNode = EntranceItem & { r: number; parent: MapNode | null; timing: NodeTiming };

/**
 * Lines grow out of their parents, nodes fade in as their stars collapse into them, and names
 * follow. `t` null draws the settled map.
 */
function drawMap(c: CanvasRenderingContext2D, tf: Transform, nodes: MapNode[], t: number | null) {
  for (const n of nodes) {
    if (!n.parent) continue;
    const grow =
      t === null ? 1 : easeInOutCubic(progress(t, n.timing.edgeStart, n.timing.edgeEnd - n.timing.edgeStart));
    drawEdge(c, tf, n, n.parent, n, grow, Math.min(1, grow * 1.6));
  }
  for (const n of nodes) {
    const shown = t === null ? 1 : easeOutCubic(progress(t, n.timing.fadeIn, n.timing.fadeOut - n.timing.fadeIn));
    drawNode(c, tf, n, n.x, n.y, 0.86 + 0.14 * shown, shown);
  }
  for (const n of nodes) {
    const shown =
      t === null ? 1 : easeOutCubic(progress(t, n.timing.labelStart, n.timing.labelEnd - n.timing.labelStart));
    drawLabel(c, tf, n, n.x, n.y, shown);
  }
  c.setTransform(1, 0, 0, 1, 0, 0);
}

// ───────────────────────── overlay ─────────────────────────

const SKIP_ICON =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 17 5-5-5-5"></path><path d="m13 17 5-5-5-5"></path></svg>';

function createOverlay() {
  const element = document.createElement('div');
  element.className = 'night-intro';
  element.dataset.state = 'waiting';
  const glCanvas = document.createElement('canvas');
  glCanvas.className = 'night-intro-sky';
  const canvas = document.createElement('canvas');
  const title = document.createElement('div');
  title.className = 'night-intro-title';
  const lines = [0, 1, 2].map(() => title.appendChild(document.createElement('div')));
  for (const node of [glCanvas, canvas]) node.setAttribute('aria-hidden', 'true');
  const skip = document.createElement('button');
  skip.type = 'button';
  skip.className = 'night-intro-skip';
  skip.setAttribute('aria-label', '跳过开场动画');
  skip.innerHTML = `跳过${SKIP_ICON}`;
  element.append(glCanvas, canvas, title, skip);
  return { element, glCanvas, canvas, title, lines, skip };
}

const formatCount = (value: number) => value.toLocaleString('en-US');
const yearOf = (time: string | null | undefined) => (time ? Number(time.slice(0, 4)) || null : null);

/** "2021 — 2025", "你说过 58,516 句话", "其中 1,234 件事被记住了". */
function fillTitle(lines: HTMLDivElement[], data: IntroData | null, model: ProfileModel) {
  const times = Object.values(model.sources)
    .map((source) => source.sample_time)
    .filter((time): time is string => !!time)
    .sort();
  const now = new Date().getFullYear();
  const first = yearOf(data?.first_time) ?? yearOf(times[0]) ?? now;
  const last = yearOf(data?.last_time) ?? yearOf(times[times.length - 1]) ?? now;
  lines[0].textContent = first === last ? String(first) : `${first} — ${last}`;
  const total = data?.message_total;
  lines[1].textContent = typeof total === 'number' && total > 0 ? `你说过 ${formatCount(total)} 句话` : '你说过的话';
  const kept = model.facts.filter((fact) => fact.status !== 'rejected').length;
  const accent = document.createElement('span');
  accent.textContent = formatCount(kept);
  lines[2].replaceChildren('其中 ', accent, ' 件事被记住了');
}

function particleCount(data: IntroData | null, model: ProfileModel): number {
  const touch = window.matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) <= 820;
  if (touch) return TOUCH_PARTICLES;
  const total = data?.message_total ?? Object.keys(model.sources).length;
  return Math.round(Math.min(MAX_PARTICLES, Math.max(MIN_PARTICLES, total)));
}

const wait = <T>(promise: Promise<T>, ms: number, fallback: T) =>
  Promise.race([promise, new Promise<T>((resolve) => window.setTimeout(() => resolve(fallback), ms))]);

// ───────────────────────── first visit ─────────────────────────

// `?intro` in the address, or replayNightIntro(), makes the next open play even after a visit.
let forced = new URLSearchParams(window.location.search).has('intro');

/** Whether the intro may play at all: not when the user prefers reduced motion. */
export const nightIntroAllowed = () => !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Makes the next openNightIntro() play, as on a first visit. */
export function replayNightIntro() {
  forced = true;
}

function seen(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'seen';
  } catch {
    return false;
  }
}

function markSeen() {
  try {
    window.localStorage.setItem(STORAGE_KEY, 'seen');
  } catch {
    // Storage may be unavailable, e.g. in a private window: the intro then plays on every visit.
  }
}

/** The page around the map is hidden while the intro plays and fades in when the map takes over. */
function revealChrome() {
  const root = document.documentElement;
  root.dataset.atlasIntro = 'revealing';
  window.setTimeout(() => {
    if (root.dataset.atlasIntro === 'revealing') delete root.dataset.atlasIntro;
  }, CHROME_FADE_MS);
}

/**
 * What playEntrance gets, and the model for the numbers and the stars. Only the host's position on
 * the page is used: the intro covers the whole window. As with playEntrance, at the end the overlay
 * goes as soon as `onReveal(true)` returns.
 */
export type IntroStart = EntranceConfig & { model: ProfileModel };

export interface NightIntro {
  /**
   * Plays the intro over the freshly laid out map and returns a function that ends it, or null
   * when it cannot play (no WebGL, nothing to show, already skipped): the caller then plays the
   * usual entrance.
   */
  play: (start: IntroStart) => (() => void) | null;
  /** Ends the intro wherever it is; the overlay fades out. */
  close: () => void;
}

/**
 * On the first visit, or once after `?intro` or replayNightIntro(), turns the page to night at once
 * and starts loading the intro's numbers and lines. Returns null otherwise, and when the user
 * prefers reduced motion.
 */
export function openNightIntro(): NightIntro | null {
  if (!nightIntroAllowed()) {
    markSeen();
    return null;
  }
  if (!forced && seen()) return null;
  forced = false;
  const data = api.get<IntroData>('profile/intro').catch(() => null);
  const overlay = createOverlay();
  document.body.append(overlay.element);
  document.documentElement.dataset.atlasIntro = 'playing';

  let state: 'waiting' | 'preparing' | 'playing' | 'done' = 'waiting';
  let reveal: ((finished: boolean) => void) | null = null;
  let stars: StarLayer | null = null;
  let raf = 0;
  const watchdog = window.setTimeout(() => {
    if (state === 'waiting') leave();
  }, LAYOUT_WAIT_MS);

  const listeners: [EventTarget, string, EventListener, AddEventListenerOptions?][] = [];
  const listen = (target: EventTarget, type: string, listener: EventListener, options?: AddEventListenerOptions) => {
    target.addEventListener(type, listener, options);
    listeners.push([target, type, listener, options]);
  };

  function teardown() {
    cancelAnimationFrame(raf);
    window.clearTimeout(watchdog);
    for (const [target, type, listener, options] of listeners) target.removeEventListener(type, listener, options);
    stars?.destroy();
    stars = null;
    overlay.element.remove();
  }

  /** Ends early: the map appears under the fading overlay. */
  function leave() {
    if (state === 'done') return;
    const revealMap = reveal;
    state = 'done';
    markSeen();
    cancelAnimationFrame(raf);
    overlay.element.dataset.state = 'leaving';
    revealChrome();
    revealMap?.(false);
    window.setTimeout(teardown, LEAVE_MS);
  }

  listen(overlay.skip, 'pointerdown', (event) => event.stopPropagation());
  listen(overlay.skip, 'click', () => leave());
  listen(overlay.element, 'pointerdown', () => leave());
  listen(overlay.element, 'wheel', () => leave(), { passive: true });
  listen(window, 'keydown', (event) => {
    if ((event as KeyboardEvent).key === 'Escape') leave();
  });
  // The stars are placed in window pixels.
  listen(window, 'resize', () => leave());

  async function prepare(start: IntroStart, layer: StarLayer, glDpr: number) {
    const [intro] = await Promise.all([
      wait(data, DATA_WAIT_MS, null),
      wait(document.fonts.ready, FONT_WAIT_MS, undefined),
    ]);
    if (state !== 'preparing') return;
    const { host, items, camera, model } = start;
    const width = overlay.element.clientWidth || window.innerWidth;
    const height = overlay.element.clientHeight || window.innerHeight;
    // Map points in window pixels: the camera places them in the host, the host sits at `offset`.
    const offset = host.getBoundingClientRect();
    const camX = camera.x + offset.left;
    const camY = camera.y + offset.top;
    fillTitle(overlay.lines, intro, model);
    // Stars dim behind the title while it shows.
    const title = overlay.title.getBoundingClientRect();
    const textMask: [number, number, number, number] = [
      title.left + title.width / 2,
      title.top + title.height / 2,
      title.width / 2 + 70,
      title.height / 2 + 60,
    ];

    const tools = glyphTools(width, height);
    // Without lines the intro opens on the title.
    const lines = intro && Array.isArray(intro.sentences) ? intro.sentences : [];
    const spans = lines.length ? layoutWords(lines, width, height, tools.measure) : null;
    const T = timeline(!!spans?.length);
    const words = spans?.length ? sampleGlyphs(spans, tools.raster, T.flowStart - 0.12) : null;
    const planNodes = items.map((item) => ({
      id: item.id,
      kind: item.kind,
      dim: item.dim,
      x: item.x * camera.zoom + camX,
      y: item.y * camera.zoom + camY,
      r: (item.look.size / 2) * camera.zoom,
    }));
    const { particles, timings } = planParticles(model, planNodes, {
      timeline: T,
      width,
      height,
      count: particleCount(intro, model),
      zoom: camera.zoom,
      dims: DIMENSIONS.map((entry) => entry.dim),
      words,
    });
    const center = particles.center;
    const maxDistance = Math.max(
      Math.hypot(center.x, center.y),
      Math.hypot(width - center.x, center.y),
      Math.hypot(center.x, height - center.y),
      Math.hypot(width - center.x, height - center.y),
    );
    overlay.glCanvas.width = Math.max(1, Math.round(width * glDpr));
    overlay.glCanvas.height = Math.max(1, Math.round(height * glDpr));
    layer.load({ plan: particles, timeline: T, width, height, maxDistance, textMask });

    // Shadow blur is in device pixels: the map's last phase is drawn at the map's own pixel ratio,
    // so the picture handed over is identical.
    const dpr = canvasDpr();
    overlay.canvas.width = Math.max(1, Math.round(width * dpr));
    overlay.canvas.height = Math.max(1, Math.round(height * dpr));
    const c = overlay.canvas.getContext('2d')!;
    const tf: Transform = { k: dpr * camera.zoom, ox: dpr * camX, oy: dpr * camY };
    const byId = new Map<string, MapNode>();
    const mapNodes = items.map((item): MapNode => {
      const node = { ...item, r: item.look.size / 2, parent: null, timing: timings.get(item.id)! };
      byId.set(item.id, node);
      return node;
    });
    for (const node of mapNodes) node.parent = node.parentId ? (byId.get(node.parentId) ?? null) : null;
    let settled = 0;
    for (const timing of timings.values()) settled = Math.max(settled, timing.fadeOut, timing.edgeEnd, timing.labelEnd);

    // Dawn reaches the skip button as it reaches this point.
    const skipDistance = Math.hypot(width - 60 - center.x, height - 40 - center.y);
    let painted: 'none' | 'words' | 'map' | 'settled' = 'none';
    const clear = () => {
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, overlay.canvas.width, overlay.canvas.height);
    };
    const render = (t: number) => {
      let textMaskK = 0;
      overlay.lines.forEach((line, index) => {
        const { opacity, y } = titleLine(T, index, t);
        textMaskK = Math.max(textMaskK, opacity);
        line.style.opacity = opacity.toFixed(3);
        line.style.transform = `translate3d(0, ${y.toFixed(2)}px, 0)`;
      });
      overlay.skip.style.opacity = (1 - smoothstep(T.skipFadeStart, T.total, t)).toFixed(3);
      overlay.element.style.setProperty('--intro-progress', (t / T.total).toFixed(4));
      overlay.element.style.setProperty('--intro-day', dayAt(T, t, skipDistance, maxDistance).toFixed(3));
      // The sky has faded in over the backdrop by now, and dawn must uncover the page later.
      if (t > 0.4 && !overlay.element.dataset.sky) overlay.element.dataset.sky = 'on';
      layer.render({
        time: t,
        globalAlpha: 1 - smoothstep(T.particlesFadeStart, T.particlesFadeEnd, t),
        textMaskK,
        dayProgress: dayProgress(T, t),
      });
      if (t >= T.settleStart - 0.02) {
        if (painted === 'settled') return;
        clear();
        drawMap(c, tf, mapNodes, t >= settled ? null : t);
        painted = t >= settled ? 'settled' : 'map';
      } else if (words && t < T.wordsEnd) {
        clear();
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawWords(c, words.spans, t);
        painted = 'words';
      } else if (painted !== 'none') {
        clear();
        painted = 'none';
      }
    };

    state = 'playing';
    overlay.element.dataset.state = 'playing';
    render(0);
    let t = 0;
    let last = performance.now();
    const frame = (now: number) => {
      if (state !== 'playing') return;
      t += Math.min(0.05, Math.max(0, (now - last) / 1000));
      last = now;
      if (t < T.total) {
        render(t);
        raf = requestAnimationFrame(frame);
        return;
      }
      // By now the sky and the stars are gone and the overlay shows the map as the map draws it,
      // so the map takes over in this frame.
      state = 'done';
      markSeen();
      revealChrome();
      reveal?.(true);
      teardown();
    };
    raf = requestAnimationFrame((now) => {
      last = now;
      frame(now);
    });
  }

  return {
    play(start) {
      if (state !== 'waiting') return null;
      const glDpr = Math.min(2, window.devicePixelRatio || 1);
      const hasRoot = start.items.some((item) => item.kind === 'root');
      const layer = hasRoot && start.model.facts.length ? createStarLayer(overlay.glCanvas, glDpr) : null;
      if (!layer) {
        leave();
        return null;
      }
      stars = layer;
      state = 'preparing';
      reveal = start.onReveal;
      void prepare(start, layer, glDpr).catch((error: unknown) => {
        console.error('Atlas intro failed', error);
        leave();
      });
      return leave;
    },
    close: leave,
  };
}
