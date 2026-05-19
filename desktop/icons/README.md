# App icons

Tauri expects these files to exist before `cargo tauri build` succeeds:

```
desktop/icons/
  32x32.png
  128x128.png
  128x128@2x.png
  icon.icns      (macOS)
  icon.ico       (Windows, optional for Mac-only)
```

## Generate from a single source

Once you have a 1024×1024 PNG for the brand mark, run from the project root:

```bash
cd desktop
cargo tauri icon ../path/to/icon-source-1024.png
```

This generates every size + the `.icns` and `.ico` automatically.

For the initial scaffold + `cargo tauri dev`, icons aren't strictly required —
the dev window uses a default placeholder. They're only needed for `cargo
tauri build` (.dmg generation).

## Brand-mark direction

Match the Apple-quiet voice in [`project-domain-and-brand`](../../README.md):

- Wordmark only — no abstract logo
- Black on white (light mode), white on black (dark mode)
- Simple geometric letterforms; SF Pro or similar
- The whole mark should read at 16×16 — that's the favicon/tray bound
