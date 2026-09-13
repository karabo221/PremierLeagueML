import fs from "node:fs";
import { fileURLToPath } from "node:url";

/**
 * Is this build running on a filesystem where readlink on a regular file
 * returns EISDIR? That is FAT32, and it breaks two separate things in webpack.
 * fat32-readlink.cjs explains the measurement; this is the same probe.
 */
function onFat32() {
  try {
    fs.readlinkSync(fileURLToPath(import.meta.url));
    return false;
  } catch (error) {
    return error?.code === "EISDIR";
  }
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Crests are small pre-sized PNGs served straight from public/, so the
  // optimiser has nothing to do and would only add a runtime dependency.
  images: { unoptimized: true },

  webpack: (config) => {
    // Same FAT32 cause as fat32-readlink.cjs, different symptom: webpack's
    // persistent pack-file cache cannot snapshot resolve dependencies on this
    // filesystem and logs "Unable to snapshot resolve dependencies" on every
    // build. The cache is already not working, so this only stops it claiming
    // otherwise. On NTFS the probe fails and the on-disk cache is kept.
    if (onFat32()) {
      config.cache = { type: "memory" };
    }

    // E: IS FAT32, AND THAT BREAKS THE DEFAULT BUILD.
    //
    // webpack resolves every module through fs.readlink to find its real path.
    // FAT32 has no symlinks at all, and on this drive the call returns EISDIR
    // rather than EINVAL, which webpack does not treat as "not a link":
    //
    //   Error: EISDIR: illegal operation on a directory, readlink
    //   'E:\PremierLeagueML\web\node_modules\next\dist\pages\_app.js'
    //
    // Nothing in this project is symlinked, so switching the resolution off is
    // exact rather than a workaround - there is no link for it to miss. Recorded
    // in PROJECT_GOTCHAS.md alongside the other environment failures, because
    // the error names a file that is plainly a regular file and reads like a
    // corrupt install.
    config.resolve.symlinks = false;
    return config;
  },
};

export default nextConfig;
