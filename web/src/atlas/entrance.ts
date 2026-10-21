// Plays the map's entrance on overlay canvases above the map (design: WC-exp/design, "画像入场 A";
// schedule in entrancePlan.ts). Nodes and lines are drawn with Canvas 2D: settled ones are baked
// into an offscreen layer, so each frame only draws the few that are animating. Particles run on a
// WebGL canvas whose buffers are fixed up front and animated by one time uniform. When everything
// has appeared the map takes over underneath and the overlay goes, so the last frame draws nodes,
// lines and labels with the map's own primitives (paint.ts).
import {
  buildEntrancePlan,
  clamp,
  easeInOutCubic,
  mulberry32,
  nodeFrame,
  smooth,
  type EntranceInput,
  type EntranceNode,
  type EntrancePlan,
  type NodeFrame,
} from './entrancePlan';
import { glowColor, hexToRgb, nodeFill, nodeStroke } from './graphStyle';
import { canvasDpr, drawEdge, drawLabel, drawNode, type PearlSlice, type Transform } from './paint';

export type EntranceItem = EntranceInput & {
  dim: number | null;
  label: string;
  selected: boolean;
  /** The centre's dimensions, which tint the pearl. */
  slices?: PearlSlice[];
};

/** Viewport position of a map point: point × zoom + (x, y). */
export type Camera = { zoom: number; x: number; y: number };

type Node = EntranceNode<EntranceItem>;

const TAU = Math.PI * 2;
/** Stars below this salience surface without bloom or sparkles, which keeps a busy sweep calm. */
const BLOOM_SALIENCE = 0.7;

// ───────────────────────── particles ─────────────────────────
//   kind 0  bloom   – soft disc that collapses into the node: the eye focusing
//   kind 1  sparkle – specks gathering into the node; low-confidence stars stay scattered
//   kind 2  glint   – four-point flare on the brightest stars (echoes the ✦ mark)
const PARTICLE_FLOATS = 14; // target 2, offset 2, time 2, colour 3, size 2, misc 3

/** Particle vertices, and the time the last particle has faded. */
function buildParticles(plan: EntrancePlan<EntranceItem>): { data: Float32Array; end: number } {
  const rand = mulberry32(99);
  const out: number[] = [];
  let end = 0;
  const add = (
    tx: number,
    ty: number,
    ox: number,
    oy: number,
    t0: number,
    dur: number,
    hex: string,
    s0: number,
    s1: number,
    alpha: number,
    kind: number,
    screen: boolean,
  ) => {
    const [r, g, b] = hexToRgb(hex);
    end = Math.max(end, t0 + dur);
    out.push(tx, ty, ox, oy, t0, dur, r / 255, g / 255, b / 255, s0, s1, alpha, kind, screen ? 1 : 0);
  };
  // Dimension colour, or the root's own.
  const tint = (n: Node) => (n.kind === 'root' ? nodeFill(n.kind, n.dim) : nodeStroke(n.kind, n.dim));
  const sparkles = (n: Node, count: number, t0: number, spread: number, reachIn: number, life: number) => {
    for (let i = 0; i < count; i++) {
      const th = rand() * TAU;
      const from = n.r + 16 + rand() * spread;
      const to = n.r * reachIn;
      const ex = n.x + Math.cos(th) * to;
      const ey = n.y + Math.sin(th) * to;
      add(
        ex,
        ey,
        Math.cos(th) * (from - to),
        Math.sin(th) * (from - to),
        t0 - rand() * 0.08,
        life * (1 + rand() * 0.7),
        tint(n),
        3 + rand() * 1.6,
        1.4,
        0.72,
        1,
        true,
      );
    }
  };
  const { root, dimensions, stars } = plan;
  const size = (n: Node) => n.look.size;
  add(
    root.x,
    root.y,
    0,
    0,
    root.appearAt,
    0.95,
    glowColor('root', null),
    size(root) * 3.2,
    size(root) * 1.15,
    0.34,
    0,
    false,
  );
  add(root.x, root.y, 0, 0, root.appearAt + 0.12, 0.8, tint(root), size(root) * 2.6, size(root) * 3.4, 0.3, 2, false);
  sparkles(root, 16, root.appearAt - 0.04, 110, 0.8, 0.42);
  for (const d of dimensions) {
    const color = tint(d);
    // a stream of light thrown from the root to where the dimension lands
    for (let j = 0; j < 10; j++) {
      add(
        d.x,
        d.y,
        root.x - d.x,
        root.y - d.y,
        d.appearAt + j * 0.03 - 0.02,
        d.dur * (0.5 + rand() * 0.35),
        color,
        2.8 + rand(),
        1.4,
        0.7,
        1,
        true,
      );
    }
    add(d.x, d.y, 0, 0, d.appearAt + d.dur * 0.38, 0.75, color, size(d) * 2.8, size(d) * 1.1, 0.24, 0, false);
  }
  for (const n of stars) {
    const blur = 1 - n.look.clarity;
    const sal = n.look.sal01;
    const small = n.kind === 'attribute';
    if (sal < BLOOM_SALIENCE) continue;
    // A vague star never fully focuses: its bloom stays wider and lingers. Star particles live
    // shorter than in the design, so the last ones have faded by the end of the faster entrance.
    add(
      n.x,
      n.y,
      0,
      0,
      n.appearAt - 0.04,
      n.dur * 1.2 + 0.1,
      tint(n),
      size(n) * (3.0 + 1.4 * blur),
      size(n) * (1.1 + 1.3 * blur),
      (small ? 0.16 : 0.2) + 0.22 * sal,
      0,
      false,
    );
    const count = small ? Math.round(1 + 2 * sal) : Math.round(3 + 6 * sal);
    sparkles(n, count, n.appearAt - 0.18 * n.dur, !small ? 26 + 30 * sal : 18, 0.55 + 1.1 * blur, 0.32);
  }
  for (const n of stars
    .filter((star) => star.kind !== 'attribute')
    .sort((a, b) => b.look.sal01 - a.look.sal01)
    .slice(0, 7)) {
    add(n.x, n.y, 0, 0, n.appearAt + 0.02, 0.8, tint(n), size(n) * 2.6, size(n) * 4.4, 0.5, 2, false);
  }
  return { data: new Float32Array(out), end };
}

const PARTICLE_VERT = `
precision highp float;
attribute vec2 a_target, a_offset, a_time, a_size;
attribute vec3 a_color, a_misc;
uniform float u_time, u_dpr, u_max;
uniform vec3 u_cam;
uniform vec2 u_view;
varying vec3 v_color;
varying float v_alpha, v_kind;
void main() {
  v_color = a_color;
  v_kind = a_misc.y;
  float p = (u_time - a_time.x) / a_time.y;
  if (p <= 0.0 || p >= 1.0) { gl_Position = vec4(-2.0, -2.0, 0.0, 1.0); gl_PointSize = 0.0; v_alpha = 0.0; return; }
  float e = 1.0 - pow(1.0 - p, 3.0);
  vec2 screen = (a_target + a_offset * (1.0 - e)) * u_cam.x + u_cam.yz;
  gl_Position = vec4(screen.x / u_view.x * 2.0 - 1.0, 1.0 - screen.y / u_view.y * 2.0, 0.0, 1.0);
  float size = mix(a_size.x, a_size.y, e) * (a_misc.z > 0.5 ? 1.0 : u_cam.x);
  gl_PointSize = min(size * u_dpr, u_max);
  float env;
  if (a_misc.y < 0.5) env = smoothstep(0.0, 0.14, p) * pow(1.0 - p, 1.5);
  else if (a_misc.y < 1.5) env = smoothstep(0.0, 0.18, p) * (1.0 - smoothstep(0.62, 1.0, p));
  else env = smoothstep(0.0, 0.1, p) * pow(1.0 - p, 2.0);
  v_alpha = a_misc.x * env;
}`;

const PARTICLE_FRAG = `
precision mediump float;
varying vec3 v_color;
varying float v_alpha, v_kind;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(q, q);
  if (d2 > 1.0) discard;
  float a;
  if (v_kind < 0.5) a = exp(-d2 * 4.2);
  else if (v_kind < 1.5) a = exp(-d2 * 5.0);
  else {
    float sx = exp(-q.y * q.y * 700.0) * (1.0 - abs(q.x));
    float sy = exp(-q.x * q.x * 700.0) * (1.0 - abs(q.y));
    a = max(sx, sy) + exp(-d2 * 26.0) * 0.5;
  }
  a *= v_alpha;
  gl_FragColor = vec4(v_color * a, a);
}`;

type ParticleLayer = { frame: (time: number) => void; destroy: () => void };

function createParticleLayer(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  camera: Camera,
  dpr: number,
): ParticleLayer | null {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    depth: false,
    stencil: false,
  });
  if (!gl) return null;
  const shader = (type: number, source: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, source);
    gl.compileShader(s);
    return s;
  };
  const vert = shader(gl.VERTEX_SHADER, PARTICLE_VERT);
  const frag = shader(gl.FRAGMENT_SHADER, PARTICLE_FRAG);
  const program = gl.createProgram()!;
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  const buffer = gl.createBuffer();
  const release = () => {
    gl.deleteBuffer(buffer);
    gl.deleteProgram(program);
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  };
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    release();
    return null;
  }
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  let offset = 0;
  for (const [name, size] of [
    ['a_target', 2],
    ['a_offset', 2],
    ['a_time', 2],
    ['a_color', 3],
    ['a_size', 2],
    ['a_misc', 3],
  ] as const) {
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, PARTICLE_FLOATS * 4, offset * 4);
    offset += size;
  }
  const maxPoint = (gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array)[1] || 256;
  gl.uniform1f(gl.getUniformLocation(program, 'u_dpr'), dpr);
  gl.uniform1f(gl.getUniformLocation(program, 'u_max'), maxPoint);
  gl.uniform3f(gl.getUniformLocation(program, 'u_cam'), camera.zoom, camera.x, camera.y);
  gl.uniform2f(gl.getUniformLocation(program, 'u_view'), canvas.width / dpr, canvas.height / dpr);
  const timeLocation = gl.getUniformLocation(program, 'u_time');
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  const count = data.length / PARTICLE_FLOATS;
  return {
    frame(time) {
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(timeLocation, time);
      gl.drawArrays(gl.POINTS, 0, count);
    },
    destroy: release,
  };
}

// ───────────────────────── playback ─────────────────────────

export interface EntranceConfig {
  /** Element above the map that receives the overlay canvases; it must not catch pointer events. */
  host: HTMLElement;
  items: EntranceItem[];
  camera: Camera;
  width: number;
  height: number;
  /**
   * Called once the map underneath should be shown, at the end or when the entrance is stopped.
   * At the end (`finished`) the overlay removes its picture as soon as this returns, so the map
   * must be drawn by then: the two pictures are identical, but overlapping they would darken
   * every translucent pixel for a moment.
   */
  onReveal: (finished: boolean) => void;
}

/** Starts the entrance and returns a function that ends it at once. */
export function playEntrance({ host, items, camera, width, height, onReveal }: EntranceConfig): () => void {
  // Shadow blur is in device pixels: the overlay must use the map's pixel ratio to hand over an
  // identical picture.
  const dpr = canvasDpr();
  const makeCanvas = () => {
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    return canvas;
  };
  const glCanvas = makeCanvas();
  const canvas = makeCanvas();
  // Settled lines and nodes, kept apart so every line stays below every node, as on the map.
  const bakedEdges = makeCanvas();
  const bakedNodes = makeCanvas();
  host.append(glCanvas, canvas);
  const ctx = canvas.getContext('2d')!;
  const bakedEdgesCtx = bakedEdges.getContext('2d')!;
  const bakedNodesCtx = bakedNodes.getContext('2d')!;
  const tf: Transform = { k: dpr * camera.zoom, ox: dpr * camera.x, oy: dpr * camera.y };
  const plan = buildEntrancePlan(items);
  const { data: particleData, end: particlesEnd } = buildParticles(plan);
  const particles = createParticleLayer(glCanvas, particleData, camera, dpr);
  const frameOf = new Map<Node, NodeFrame>(plan.nodes.map((n) => [n, { x: n.x, y: n.y, alpha: 1, scale: 1, glow: 1 }]));

  const activeNodes: Node[] = [];
  const activeEdges: Node[] = [];
  let nodePtr = 0;
  let edgePtr = 0;
  let t = 0;
  let last = performance.now();
  let raf = 0;
  // After the nodes settle, the last particles play out above the map.
  let state: 'playing' | 'trailing' | 'done' = 'playing';

  // Current position and radius of a node that may still be animating.
  const live = (n: Node, time: number) => {
    const f = frameOf.get(n)!;
    if (time < n.end && nodeFrame(n, time, plan.root, f)) return { x: f.x, y: f.y, r: n.r * f.scale };
    return n;
  };

  function advance(time: number) {
    const nq = plan.nodeQueue;
    while (nodePtr < nq.length && nq[nodePtr].appearAt <= time) activeNodes.push(nq[nodePtr++]);
    const eq = plan.edgeQueue;
    while (edgePtr < eq.length && eq[edgePtr].edgeAt <= time) activeEdges.push(eq[edgePtr++]);
    // A line bakes once both its ends have stopped moving, so a baked line never trails a node.
    let j = 0;
    for (const n of activeEdges) {
      if (time >= n.edgeEnd && time >= n.end && time >= n.parent!.end) drawEdge(bakedEdgesCtx, tf, n, n.parent!, n);
      else activeEdges[j++] = n;
    }
    activeEdges.length = j;
    j = 0;
    for (const n of activeNodes) {
      if (time >= n.end) drawNode(bakedNodesCtx, tf, n, n.x, n.y);
      else activeNodes[j++] = n;
    }
    activeNodes.length = j;
  }

  function composite(time: number) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bakedEdges, 0, 0);
    for (const n of activeEdges) {
      let grow = 1;
      let alpha = 1;
      if (n.kind === 'dimension') alpha = smooth(0, 0.25, (time - n.appearAt) / n.dur);
      else grow = easeInOutCubic(clamp((time - n.edgeAt) / (n.edgeEnd - n.edgeAt), 0, 1));
      drawEdge(ctx, tf, n, live(n.parent!, time), live(n, time), grow, alpha);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(bakedNodes, 0, 0);
    for (const n of activeNodes) {
      const f = frameOf.get(n)!;
      if (nodeFrame(n, time, plan.root, f)) drawNode(ctx, tf, n, f.x, f.y, f.scale, f.alpha, f.glow);
    }
    for (const n of plan.nodes) {
      if (time < n.labelAt) continue;
      const at = live(n, time);
      drawLabel(ctx, tf, n, at.x, at.y, smooth(n.labelAt, n.labelEnd, time));
    }
  }

  function cleanup() {
    cancelAnimationFrame(raf);
    particles?.destroy();
    glCanvas.remove();
    canvas.remove();
  }

  // The map shows the same picture, so it takes over in the same frame. Fading across instead
  // would stack the two and briefly darken every translucent fill, glow and line.
  function finish() {
    onReveal(true);
    if (state !== 'playing') return;
    canvas.remove();
    state = particles && t < particlesEnd ? 'trailing' : 'done';
    if (state === 'done') cleanup();
  }

  function frame(now: number) {
    t += Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    if (state === 'playing') {
      if (t >= plan.end) finish();
      else {
        advance(t);
        composite(t);
      }
    } else if (state === 'trailing' && t >= particlesEnd) {
      state = 'done';
      cleanup();
    }
    if (state === 'done') return;
    particles?.frame(t);
    raf = requestAnimationFrame(frame);
  }

  raf = requestAnimationFrame((now) => {
    last = now;
    frame(now);
  });

  return () => {
    if (state === 'done') return;
    const wasPlaying = state === 'playing';
    state = 'done';
    cleanup();
    if (wasPlaying) onReveal(false);
  };
}
