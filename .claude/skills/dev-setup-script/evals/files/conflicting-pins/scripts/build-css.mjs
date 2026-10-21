// Concatenates assets/*.css into static/app.css. Uses import.meta.dirname (Node >= 20.11).
import { readdirSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dirname, "..");
const src = join(root, "assets");
const css = readdirSync(src).filter((f) => f.endsWith(".css")).sort()
  .map((f) => `/* ${f} */\n` + readFileSync(join(src, f), "utf8")).join("\n");
mkdirSync(join(root, "static"), { recursive: true });
writeFileSync(join(root, "static", "app.css"), css);
console.log("wrote static/app.css");
