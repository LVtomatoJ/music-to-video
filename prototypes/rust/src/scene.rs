//! Seeded ball / particle scene. Physics constants match the Python prototype.

use std::f64::consts::TAU;

const N_BALLS: usize = 6;
const SEED: u32 = 42;

pub struct Ball {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub r: f64,
    pub hue: f64,
    pub flash: f64,
}

pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub r: f64,
    pub hue: f64,
    pub life: f64,
    pub max_life: f64,
}

pub struct Scene {
    pub balls: Vec<Ball>,
    pub particles: Vec<Particle>,
    state: u32,
    pub width: f64,
    pub height: f64,
}

fn xorshift32(state: &mut u32) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x as f64 / 4294967296.0
}

impl Scene {
    pub fn new(width: u32, height: u32) -> Self {
        let mut state = SEED;
        let width = width as f64;
        let height = height as f64;
        let mut balls = Vec::with_capacity(N_BALLS);
        for _ in 0..N_BALLS {
            let radius = 22.0 + xorshift32(&mut state) * 18.0;
            let x = radius + xorshift32(&mut state) * (width - 2.0 * radius);
            let y = radius + xorshift32(&mut state) * (height - 2.0 * radius);
            let ang = xorshift32(&mut state) * TAU;
            let speed = 180.0 + xorshift32(&mut state) * 220.0;
            let hue = xorshift32(&mut state) * 360.0;
            balls.push(Ball {
                x,
                y,
                vx: ang.cos() * speed,
                vy: ang.sin() * speed,
                r: radius,
                hue,
                flash: 0.0,
            });
        }
        Self {
            balls,
            particles: Vec::new(),
            state,
            width,
            height,
        }
    }

    pub fn step(&mut self, beat: bool, strong: bool, dt: f64) -> f64 {
        if beat {
            for ball in &mut self.balls {
                on_beat(ball);
            }
        }
        if strong {
            let mut spawned = Vec::new();
            for ball in &self.balls {
                spawn(ball, &mut self.state, &mut spawned);
            }
            self.particles.extend(spawned);
        }
        let width = self.width;
        let height = self.height;
        for ball in &mut self.balls {
            ball.x += ball.vx * dt;
            ball.y += ball.vy * dt;
            if ball.x < ball.r {
                ball.x = ball.r;
                ball.vx = ball.vx.abs();
            } else if ball.x > width - ball.r {
                ball.x = width - ball.r;
                ball.vx = -ball.vx.abs();
            }
            if ball.y < ball.r {
                ball.y = ball.r;
                ball.vy = ball.vy.abs();
            } else if ball.y > height - ball.r {
                ball.y = height - ball.r;
                ball.vy = -ball.vy.abs();
            }
            ball.flash *= 0.86;
        }
        for particle in &mut self.particles {
            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;
            particle.vy += 280.0 * dt;
            particle.life -= dt;
        }
        self.particles.retain(|particle| particle.life > 0.0);
        if beat {
            1.0
        } else {
            0.0
        }
    }
}

fn on_beat(ball: &mut Ball) {
    ball.vx = -ball.vx;
    ball.vy = -ball.vy;
    let speed = ball.vx.hypot(ball.vy);
    let target = (speed * 1.12).clamp(220.0, 640.0);
    if speed > 1e-6 {
        ball.vx *= target / speed;
        ball.vy *= target / speed;
    }
    ball.hue = (ball.hue + 47.0) % 360.0;
    ball.flash = 1.0;
}

fn spawn(ball: &Ball, state: &mut u32, particles: &mut Vec<Particle>) {
    for _ in 0..28 {
        let ang = xorshift32(state) * TAU;
        let speed = 220.0 + xorshift32(state) * 520.0;
        let radius = 6.0 + xorshift32(state) * 8.0;
        particles.push(Particle {
            x: ball.x,
            y: ball.y,
            vx: ang.cos() * speed,
            vy: ang.sin() * speed - 30.0,
            r: radius,
            hue: ball.hue,
            life: 0.45,
            max_life: 0.45,
        });
    }
}

pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let h = h.rem_euclid(360.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    (r + m, g + m, b + m)
}
