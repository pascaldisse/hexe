// Houdini-style orbit camera maths (pure, no three.js: unit-tested). State = target + yaw/pitch + orbit distance.
// Mapping (egui parity): LMB / alt+LMB orbit · MMB or shift+LMB pan · RMB drag dolly · scroll dolly.
import type { Bounds } from "./geom";

export interface Cam { target: [number, number, number]; yaw: number; pitch: number; dist: number }
export const FOV = 0.9; // vertical, radians
export const FIT_MARGIN = 1.15;
export const PITCH_MAX = (89 * Math.PI) / 180;
export const DIST_MIN = 1e-3, DIST_MAX = 1e6;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export const defaultCam = (): Cam => ({ target: [0, 0, 0], yaw: 0.6, pitch: 0.45, dist: 6 });

/** Camera position. */
export function eye(c: Cam): [number, number, number] {
  const cp = Math.cos(c.pitch);
  return [c.target[0] + c.dist * cp * Math.sin(c.yaw), c.target[1] + c.dist * Math.sin(c.pitch), c.target[2] + c.dist * cp * Math.cos(c.yaw)];
}
/** Orbit by a pixel drag; pitch is clamped short of the poles. */
export function orbit(c: Cam, dx: number, dy: number, speed = 0.006): Cam {
  return { ...c, yaw: c.yaw - dx * speed, pitch: clamp(c.pitch + dy * speed, -PITCH_MAX, PITCH_MAX) };
}
/** Pan the target in the view plane by a pixel drag (scene follows the cursor). */
export function pan(c: Cam, dx: number, dy: number, viewH: number): Cam {
  const k = (2 * c.dist * Math.tan(FOV / 2)) / Math.max(1, viewH);
  const [sy, cy, sp, cp] = [Math.sin(c.yaw), Math.cos(c.yaw), Math.sin(c.pitch), Math.cos(c.pitch)];
  const right = [cy, 0, -sy], up = [-sp * sy, cp, -sp * cy];
  const t = c.target.map((v, i) => v - right[i] * dx * k + up[i] * dy * k) as [number, number, number];
  return { ...c, target: t };
}
/** Multiplicative dolly (amount > 0 = away). Distance stays in [DIST_MIN, DIST_MAX]. */
export function dolly(c: Cam, amount: number): Cam {
  return { ...c, dist: clamp(c.dist * Math.exp(amount), DIST_MIN, DIST_MAX) };
}
/** RMB drag -> dolly amount (right/up = in). */
export const dragDolly = (dx: number, dy: number): number => (dy - dx) * 0.005;
/** Wheel -> dolly amount (normalises line/page delta modes). */
export const wheelDolly = (deltaY: number, mode = 0): number => deltaY * (mode === 1 ? 0.05 : mode === 2 ? 0.5 : 0.0015);
/** Frame a bounding sphere, keeping the view direction. */
export function fit(c: Cam, b: Bounds): Cam {
  const r = Math.max(b.radius, 1e-3);
  return { ...c, target: [...b.center], dist: clamp((r / Math.sin(FOV / 2)) * FIT_MARGIN, DIST_MIN, DIST_MAX) };
}
/** Auto-frame rule on a fresh cook: frame only when the display identity `key` (scene + display node) differs from the
 *  one last framed and there is geometry. A timeline frame change / param tweak recooks under the same key, so the user's
 *  orbit / zoom survive playback. Returns the camera and the key now framed. */
export function autoFrame(c: Cam, framed: string | null, key: string, b: Bounds | null): { cam: Cam; framed: string | null } {
  return b && key !== framed ? { cam: fit(c, b), framed: key } : { cam: c, framed };
}
/** near / far are multiples of the orbit distance. */
export const clip = (c: Cam): { near: number; far: number } => ({ near: c.dist * 0.01, far: c.dist * 1000 });
