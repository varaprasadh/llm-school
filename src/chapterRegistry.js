import { lazy } from "react";

/**
 * Auto-discover every chapter component in `src/chapters/*.jsx`.
 *
 * Each file must default-export a React component and be named exactly after
 * its slug in `data/chapters.js` (e.g. `self-attention.jsx`). Adding a chapter
 * is therefore just: (1) add an entry to data/chapters.js, (2) drop in the file.
 */
const modules = import.meta.glob("./chapters/*.jsx");

export const chapterComponents = Object.fromEntries(
  Object.entries(modules).map(([path, loader]) => {
    const slug = path.split("/").pop().replace(/\.jsx$/, "");
    return [slug, lazy(loader)];
  })
);
