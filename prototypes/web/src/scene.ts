/** Seeded ball scene. Constants match prototypes/python/demo.py. */

const N_BALLS = 6;
const SEED = 42;

export interface Ball {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  hue: number;
  flash: number;
}

export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  hue: number;
  life: number;
  maxLife: number;
}

export interface Scene {
  balls: Ball[];
  particles: Particle[];
  state: number;
  width: number;
  height: number;
}

function xorshift32(state: { s: number }): number {
  let x = state.s >>> 0;
  x = (x ^ (x << 13)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ (x << 5)) >>> 0;
  state.s = x;
  return x / 4294967296;
}

export function makeScene(width: number, height: number): Scene {
  const state = { s: SEED };
  const balls: Ball[] = [];
  for (let i = 0; i < N_BALLS; i++) {
    const radius = 22 + xorshift32(state) * 18;
    const x = radius + xorshift32(state) * (width - 2 * radius);
    const y = radius + xorshift32(state) * (height - 2 * radius);
    const ang = xorshift32(state) * Math.PI * 2;
    const speed = 180 + xorshift32(state) * 220;
    const hue = xorshift32(state) * 360;
    balls.push({
      x,
      y,
      vx: Math.cos(ang) * speed,
      vy: Math.sin(ang) * speed,
      r: radius,
      hue,
      flash: 0,
    });
  }
  return { balls, particles: [], state: state.s, width, height };
}

function onBeat(ball: Ball) {
  ball.vx = -ball.vx;
  ball.vy = -ball.vy;
  const speed = Math.hypot(ball.vx, ball.vy);
  const target = Math.min(640, Math.max(220, speed * 1.12));
  if (speed > 1e-6) {
    ball.vx *= target / speed;
    ball.vy *= target / speed;
  }
  ball.hue = (ball.hue + 47) % 360;
  ball.flash = 1;
}

function spawn(scene: Scene, ball: Ball) {
  const state = { s: scene.state };
  for (let i = 0; i < 28; i++) {
    const ang = xorshift32(state) * Math.PI * 2;
    const speed = 220 + xorshift32(state) * 520;
    const radius = 6 + xorshift32(state) * 8;
    scene.particles.push({
      x: ball.x,
      y: ball.y,
      vx: Math.cos(ang) * speed,
      vy: Math.sin(ang) * speed - 30,
      r: radius,
      hue: ball.hue,
      life: 0.45,
      maxLife: 0.45,
    });
  }
  scene.state = state.s;
}

export function stepScene(scene: Scene, beat: boolean, strong: boolean, dt: number): number {
  if (beat) {
    for (const ball of scene.balls) onBeat(ball);
  }
  if (strong) {
    for (const ball of scene.balls) spawn(scene, ball);
  }
  for (const ball of scene.balls) {
    ball.x += ball.vx * dt;
    ball.y += ball.vy * dt;
    if (ball.x < ball.r) {
      ball.x = ball.r;
      ball.vx = Math.abs(ball.vx);
    } else if (ball.x > scene.width - ball.r) {
      ball.x = scene.width - ball.r;
      ball.vx = -Math.abs(ball.vx);
    }
    if (ball.y < ball.r) {
      ball.y = ball.r;
      ball.vy = Math.abs(ball.vy);
    } else if (ball.y > scene.height - ball.r) {
      ball.y = scene.height - ball.r;
      ball.vy = -Math.abs(ball.vy);
    }
    ball.flash *= 0.86;
  }
  for (const particle of scene.particles) {
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;
    particle.vy += 280 * dt;
    particle.life -= dt;
  }
  scene.particles = scene.particles.filter((particle) => particle.life > 0);
  return beat ? 1 : 0;
}

export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  h = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const m = l - c / 2;
  return [r + m, g + m, b + m];
}

export function cssColor(r: number, g: number, b: number): string {
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}
