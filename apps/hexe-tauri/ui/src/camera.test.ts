import { expect, test } from "bun:test";
import { autoFrame, clip, defaultCam, dolly, DIST_MAX, DIST_MIN, eye, fit, FIT_MARGIN, FOV, orbit, pan, PITCH_MAX } from "./camera";

test("orbit moves yaw/pitch and clamps pitch short of the poles", () => {
  const c = orbit(defaultCam(), 100, 0);
  expect(c.yaw).toBeLessThan(defaultCam().yaw);
  expect(orbit(defaultCam(), 0, 1e6).pitch).toBe(PITCH_MAX);
  expect(orbit(defaultCam(), 0, -1e6).pitch).toBe(-PITCH_MAX);
});

test("eye sits at orbit distance from the target", () => {
  const c = { target: [1, 2, 3] as [number, number, number], yaw: 0.7, pitch: -0.3, dist: 5 };
  const e = eye(c);
  expect(Math.hypot(e[0] - 1, e[1] - 2, e[2] - 3)).toBeCloseTo(5, 9);
});

test("dolly is multiplicative and clamped", () => {
  expect(dolly(defaultCam(), 1e6).dist).toBe(DIST_MAX);
  expect(dolly(defaultCam(), -1e6).dist).toBe(DIST_MIN);
  expect(dolly(defaultCam(), 0).dist).toBe(6);
});

test("pan moves the target in the view plane only (distance and direction kept)", () => {
  const c = defaultCam();
  const p = pan(c, 50, -20, 800);
  const d = [p.target[0] - c.target[0], p.target[1] - c.target[1], p.target[2] - c.target[2]];
  const e = eye(c), f = [c.target[0] - e[0], c.target[1] - e[1], c.target[2] - e[2]];
  expect(d[0] * f[0] + d[1] * f[1] + d[2] * f[2]).toBeCloseTo(0, 9);
  expect(Math.hypot(...d)).toBeCloseTo(Math.hypot(50, 20) * ((2 * 6 * Math.tan(FOV / 2)) / 800), 9);
});

test("fit frames the bounding sphere with the 1.15 margin; clip planes scale with dist", () => {
  const b = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number], center: [0, 1, 0] as [number, number, number], radius: Math.sqrt(3) };
  const c = fit(defaultCam(), b);
  expect(c.target).toEqual([0, 1, 0]);
  expect(c.dist).toBeCloseTo((Math.sqrt(3) / Math.sin(FOV / 2)) * FIT_MARGIN, 9);
  const k = clip(c);
  expect(k.far / k.near).toBeCloseTo(1e5, 3);
  expect(k.near).toBeLessThan(c.dist);
});

test("autoFrame: timeline recooks under one display key keep the user's camera; zoom-in reaches a small dist", () => {
  const b = (s: number) => ({ min: [s - 1, -1, -s - 1] as [number, number, number], max: [s + 1, 1, -s + 1] as [number, number, number], center: [s, 0, -s] as [number, number, number], radius: 10 + s });
  let r = autoFrame(defaultCam(), null, "7", b(0)); // first display: framed
  expect(r.framed).toBe("7");
  expect(r.cam.dist).toBeCloseTo(fit(defaultCam(), b(0)).dist, 9);
  let cam = dolly(orbit(r.cam, 40, 25), -8); // user orbit + zoom in
  expect(cam.dist).toBeLessThan(0.01);
  const user = structuredClone(cam);
  for (let f = 1; f <= 120; f++) { r = autoFrame(cam, r.framed, "7", b(f * 0.37)); cam = r.cam; } // playback: bounds move per frame
  expect(cam).toEqual(user);
  cam = dolly(cam, -100);
  expect(cam.dist).toBe(DIST_MIN); // zoom-in floor is the constant, not per-frame bounds
  expect(autoFrame(cam, r.framed, "9", b(1)).cam.target).toEqual([1, 0, -1]); // display-node change reframes
  expect(autoFrame(cam, "7", "7", null).cam).toBe(cam); // no geometry: untouched
});
