/**
 * Runs the Next CLI with the FAT32 readlink shim preloaded.
 *
 *   node scripts/next-with-shim.mjs dev
 *   node scripts/next-with-shim.mjs build
 *   node scripts/next-with-shim.mjs start
 *
 * WHY A LAUNCHER RATHER THAN AN ENV VAR IN package.json. The shim has to be in
 * place before ANY module loads, which means `node --require`; and it has to
 * reach Next's build workers too, which means NODE_OPTIONS rather than a flag
 * on one command. Spelling NODE_OPTIONS inline in an npm script differs between
 * cmd, PowerShell and sh, so it is set here in Node where there is one spelling.
 *
 * On NTFS the shim's own probe fails and it patches nothing, so this launcher is
 * safe to use everywhere and there is no second code path to keep working.
 */

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(here, "..");
const shim = path.join(web, "fat32-readlink.cjs");

// Forward slashes: --require takes the path as given, and a Windows backslash
// inside NODE_OPTIONS is read as an escape by Node's own parser.
const preload = `--require "${shim.split(path.sep).join("/")}"`;
const existing = process.env.NODE_OPTIONS ?? "";

const nextBin = path.join(
  web,
  "node_modules",
  "next",
  "dist",
  "bin",
  "next"
);

const child = spawn(process.execPath, [nextBin, ...process.argv.slice(2)], {
  cwd: web,
  stdio: "inherit",
  env: {
    ...process.env,
    NODE_OPTIONS: existing ? `${existing} ${preload}` : preload,
  },
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 0);
});
