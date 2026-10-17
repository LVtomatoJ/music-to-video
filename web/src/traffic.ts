import { sampleCurve, timesPulse } from "./math";
import { BAKE_DT, frameCount } from "./templates/shared";

export interface Car {
  dir: 1 | -1;
  lane: number;
  x: Float32Array;
}

/** Nose position is spawnX plus the integral of a speed that never goes negative. */
export function bakeTraffic(width: number, duration: number, beats: number[]): Car[] {
  const frames = frameCount(duration);
  const lanes = [
    { dir: 1 as const, lane: 0.3, speed: 248 },
    { dir: -1 as const, lane: 0.66, speed: 286 },
  ];
  const cars: Car[] = [];
  lanes.forEach((spec, laneIndex) => {
    const distance = new Float32Array(frames);
    let travelled = 0;
    for (let frame = 0; frame < frames; frame++) {
      const time = frame * BAKE_DT;
      const surge = 110 * timesPulse(beats, time, 0.045, 0.2);
      travelled += (spec.speed + surge) * BAKE_DT;
      distance[frame] = travelled;
    }
    const spacing = width * 0.78;
    const count = Math.ceil((distance[frames - 1] + width + spacing) / spacing) + 1;
    for (let index = 0; index < count; index++) {
      const lead = index * spacing + laneIndex * spacing * 0.37;
      const origin = spec.dir > 0 ? -340 - lead : width + 340 + lead;
      const x = new Float32Array(frames);
      for (let frame = 0; frame < frames; frame++) x[frame] = origin + spec.dir * distance[frame];
      cars.push({ dir: spec.dir, lane: spec.lane, x });
    }
  });
  return cars;
}

/** Same car, consecutive baked frames: travel is strictly one way. */
export function auditTraffic(cars: Car[]): {
  cars: number;
  samples: number;
  reversals: number;
  minStep: number;
  interpReversals: number;
} {
  let reversals = 0;
  let samples = 0;
  let minStep = Infinity;
  for (const car of cars) {
    for (let frame = 1; frame < car.x.length; frame++) {
      const step = car.dir * (car.x[frame] - car.x[frame - 1]);
      if (!(step > 1e-3)) reversals += 1;
      if (step < minStep) minStep = step;
      samples += 1;
    }
  }
  const car = cars[0];
  let interpReversals = 0;
  if (car) {
    let previous = sampleCurve(car.x, BAKE_DT, 0.4);
    for (let time = 0.4 + 1 / 120; time < 3.2; time += 1 / 120) {
      const next = sampleCurve(car.x, BAKE_DT, time);
      if (!(car.dir * (next - previous) > 1e-4)) interpReversals += 1;
      previous = next;
    }
  }
  return { cars: cars.length, samples, reversals, minStep: Number.isFinite(minStep) ? minStep : 0, interpReversals };
}
