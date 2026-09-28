/**
 * Deterministic, visually distinct color per symbol class name.
 *
 * The same class always renders in the same color — across wings, jobs and
 * sessions — so a detection box, its legend chip and the class distribution
 * list always agree. Hues are spread with the golden angle so any number of
 * classes stays distinguishable (no fixed palette to run out of).
 */
export function colorForClass(name: string): string {
  const s = (name || "").trim().toUpperCase();
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  const hue = Math.round((h * 137.508) % 360);
  const sat = 68 + (h % 3) * 8; // 68 / 76 / 84 %
  const light = 36 + ((h >> 3) % 3) * 7; // 36 / 43 / 50 %
  return `hsl(${hue} ${sat}% ${light}%)`;
}
