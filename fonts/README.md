# Fonts

Bundled so that rendering is reproducible on any machine — the same seed must produce the same pixels,
which cannot hold if templates fall back to whatever the host system happens to have installed.

**Only redistributable licenses.** All four are SIL Open Font License 1.1; the license texts are in
`LICENSES/`. Do not add a font whose license forbids redistribution — this repository becomes public.

| File | Family | License | Use |
|---|---|---|---|
| `NotoSans[wdth,wght].ttf` | Noto Sans (variable) | OFL 1.1 | Default proportional face: bank confirmations, invoices, statements. Full UA/PL/DE/ES coverage. |
| `NotoSansMono[wdth,wght].ttf` | Noto Sans Mono (variable) | OFL 1.1 | Thermal receipts (58/80 mm), fiscal blocks, POS slips. Full UA/PL/DE/ES coverage. |
| `Roboto[wdth,wght].ttf` | Roboto (variable) | OFL 1.1 | Android-style mobile banking app screenshots. |
| `RobotoMono[wght].ttf` | Roboto Mono (variable) | OFL 1.1 | Monospace variant of the above. |

Source: [github.com/google/fonts](https://github.com/google/fonts), `main`, downloaded 2026-07-26.

## Glyph coverage caveat

Roboto and Roboto Mono **do not contain ₴ (hryvnia)**. Verified with `fontTools` against a probe string of
Ukrainian, Polish, German and Spanish letters plus currency signs — Noto Sans and Noto Sans Mono cover
everything; the two Roboto faces miss only ₴.

So always declare Noto as the fallback in template CSS, and Chromium will substitute per glyph:

```css
font-family: "Roboto", "Noto Sans", sans-serif;
```

Re-run the check after adding a font:

```bash
uv run --with fonttools python - <<'EOF'
from fontTools.ttLib import TTFont
import glob
probe = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ" + "ąćęłńóśźż" + "äöüß" + "áéíóúñ¿¡" + "₴€zł"
for f in sorted(glob.glob("fonts/*.ttf")):
    cmap = TTFont(f, lazy=True).getBestCmap()
    missing = sorted({c for c in probe if ord(c) not in cmap})
    print(f"{f:42} missing: {''.join(missing) or '-'}")
EOF
```
