/**
 * FAT32 READLINK SHIM.
 *
 * Loaded via `node --require` BEFORE any other module, because webpack's
 * graceful-fs captures fs.readlink at its own require time - patching later
 * (from next.config.mjs, say) leaves that captured reference untouched and the
 * build still dies. scripts/next-with-shim.mjs sets the flag.
 *
 * THE BUG, MEASURED RATHER THAN GUESSED AT. E: is FAT32. On FAT32, Node's
 * readlink on a PLAIN FILE fails with EISDIR; on NTFS the same call on the same
 * kind of file fails with EINVAL:
 *
 *   node -e "require('fs').readlinkSync('E:/PremierLeagueML/web/app/page.tsx')"
 *       -> EISDIR                                            (FAT32)
 *   node -e "require('fs').readlinkSync('C:/Windows/notepad.exe')"
 *       -> EINVAL                                            (NTFS)
 *
 * webpack resolves every module through readlink to find its real path, and
 * treats EINVAL as "not a symlink, carry on". EISDIR is not in that set, so it
 * propagates and the build dies naming a file that is plainly a regular file:
 *
 *   Error: EISDIR: illegal operation on a directory, readlink
 *   'E:\PremierLeagueML\web\app\page.tsx'
 *
 * WHY THIS IS EXACT AND NOT A WORKAROUND. FAT32 HAS NO SYMLINKS AT ALL, so on
 * this drive every path is "not a symlink" and EINVAL is the correct answer to
 * every readlink. The shim rewrites the error code for that one syscall, and
 * only once the filesystem has already refused: it invents no success, and an
 * EACCES or ENOENT passes through untouched. On NTFS the probe fails and
 * nothing is patched at all.
 *
 * Recorded in PROJECT_GOTCHAS.md with the other environment failures, because
 * the symptom points at a corrupt node_modules and the cause is the drive.
 */

"use strict";

const fs = require("node:fs");
const os = require("node:os");

const EINVAL = -(os.constants.errno.EINVAL || 22);

/** Does readlink on a known-regular file return EISDIR here? */
function affected() {
  try {
    fs.readlinkSync(__filename);        // this file: a regular file, never a link
    return false;
  } catch (error) {
    return Boolean(error) && error.code === "EISDIR";
  }
}

function asNotALink(error) {
  if (!error || error.code !== "EISDIR" || error.syscall !== "readlink") {
    return error;
  }
  const replacement = new Error(
    "EINVAL: invalid argument, readlink '" + error.path + "'"
  );
  replacement.code = "EINVAL";
  replacement.errno = EINVAL;
  replacement.syscall = "readlink";
  replacement.path = error.path;
  return replacement;
}

if (affected() && !fs.readlinkSync.__fat32Shim) {
  const readlinkSync = fs.readlinkSync;
  const readlink = fs.readlink;
  const promisedReadlink = fs.promises.readlink;

  fs.readlinkSync = function patchedReadlinkSync() {
    try {
      return readlinkSync.apply(this, arguments);
    } catch (error) {
      throw asNotALink(error);
    }
  };
  fs.readlinkSync.__fat32Shim = true;

  fs.readlink = function patchedReadlink() {
    const args = Array.prototype.slice.call(arguments);
    const callback = args[args.length - 1];
    if (typeof callback !== "function") return readlink.apply(this, args);
    args[args.length - 1] = function (error) {
      const rest = Array.prototype.slice.call(arguments, 1);
      callback.apply(null, [error ? asNotALink(error) : error].concat(rest));
    };
    return readlink.apply(this, args);
  };
  fs.readlink.__fat32Shim = true;

  fs.promises.readlink = function patchedPromisedReadlink() {
    return promisedReadlink.apply(this, arguments).catch(function (error) {
      throw asNotALink(error);
    });
  };

  if (process.env.FAT32_SHIM_QUIET !== "1") {
    console.log("[premierleague-ml] FAT32 readlink shim active");
  }
}
