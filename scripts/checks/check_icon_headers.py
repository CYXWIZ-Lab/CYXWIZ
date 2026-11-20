"""Every icon macro draws the glyph it names.

Checks each `#define ICON_XX_NAME "<utf-8 bytes>"  // U+XXXX <glyph name> ...`
in the Engine's icon headers against the font file the header is for:

- the bytes are valid UTF-8 for one codepoint, the codepoint in the comment;
- the font maps that codepoint to the glyph the comment names;
- when the macro's own name is a glyph of the font (ICON_FA_CIRCLE_NODES ->
  circle-nodes), the comment names that glyph;
- Font Awesome codepoints the Engine draws as icons are in U+E000-U+F8FF, the
  range application.cpp merges into the interface fonts (ASCII macros such as
  ICON_FA_HASHTAG draw the text font's own character and are allowed).

Needs fontTools (pip install fonttools); without it the check exits 77 (ctest
reports it as skipped).

    python check_icon_headers.py [<repository root>]
"""
import pathlib
import re
import sys

HEADERS = [  # header, font, macro prefix
    ('cyxwiz-engine/src/gui/icons.h', 'fa-solid-900.ttf', 'ICON_FA_'),
    ('cyxwiz-engine/src/gui/IconsTabler.h', 'tabler-icons.ttf', 'ICON_TI_'),
    ('cyxwiz-engine/src/gui/IconsRemix.h', 'remixicon.ttf', 'ICON_RI_'),
    ('cyxwiz-engine/src/gui/IconsLucide.h', 'lucide.ttf', 'ICON_LU_'),
    ('cyxwiz-engine/src/gui/IconsIconoir.h', 'iconoir.ttf', 'ICON_IO_'),
    ('cyxwiz-engine/src/gui/IconsPhosphor.h', 'phosphor.ttf', 'ICON_PH_'),
]
FONTS = 'cyxwiz-engine/resources/fonts'
RAW_ICON_FOLDERS = ['cyxwiz-engine/src', 'plugins', 'cyxwiz-server-node/src']
DEFINE = re.compile(r'#define\s+(ICON_[A-Z]+_\w+)\s+"([^"]*)"\s*(.*)')
COMMENT = re.compile(r'//\s*U\+([0-9A-Fa-f]{4,6})\s+([a-z0-9][a-z0-9-]*)')


def literal_bytes(text):
    out = bytearray()
    i = 0
    while i < len(text):
        if text.startswith('\\x', i):
            out.append(int(text[i + 2:i + 4], 16))
            i += 4
        else:
            out += text[i].encode('utf-8')
            i += 1
    return bytes(out)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        print('fontTools is not installed (pip install fonttools): icon headers not checked')
        return 77

    failures = []
    total = 0
    for header, font, prefix in HEADERS:
        cmap = TTFont(str(root / FONTS / font)).getBestCmap()
        glyphs = set(cmap.values())
        for number, line in enumerate((root / header).read_text(encoding='utf-8').splitlines(), 1):
            m = DEFINE.match(line)
            if not m:
                continue
            total += 1
            macro, literal, comment = m.groups()
            where = '%s:%d %s' % (header, number, macro)
            try:
                text = literal_bytes(literal).decode('utf-8')
            except UnicodeDecodeError:
                failures.append(where + ': not valid UTF-8')
                continue
            if len(text) != 1:
                failures.append(where + ': not one character')
                continue
            cp = ord(text)
            c = COMMENT.search(comment)
            if not c:
                failures.append(where + ': the comment does not say "U+XXXX <glyph name>"')
                continue
            named_cp, named_glyph = int(c.group(1), 16), c.group(2)
            if named_cp != cp:
                failures.append(where + ': bytes are U+%04X, the comment says U+%04X' % (cp, named_cp))
            actual = cmap.get(cp)
            if actual is None:
                failures.append(where + ': %s has no glyph at U+%04X' % (font, cp))
            elif actual != named_glyph:
                failures.append(where + ': %s draws %s at U+%04X, the comment says %s' % (font, actual, cp, named_glyph))
            own = macro[len(prefix):].lower().replace('_', '-')
            if own in glyphs and actual is not None and actual != own:
                failures.append(where + ': %s has a glyph named %s, the macro draws %s' % (font, own, actual))
            if prefix == 'ICON_FA_' and cp >= 0x80 and not 0xE000 <= cp <= 0xF8FF:
                failures.append(where + ': U+%04X is outside U+E000-U+F8FF, the range merged into the fonts' % cp)

    # Icons are named through the headers above, never typed as bytes elsewhere:
    # a typed byte string is checked by nothing (the Nodes panel's example
    # categories drew five wrong icons that way).
    headers = {str((root / h).resolve()) for h, _, _ in HEADERS}
    raw_icon = re.compile(r'"(?:\\x(?:e[e-f]|ef)\\x[0-9a-fA-F]{2}\\x[0-9a-fA-F]{2})+"')
    for folder in RAW_ICON_FOLDERS:
        for path in sorted((root / folder).rglob('*')):
            if path.suffix not in ('.cpp', '.h', '.hpp') or str(path.resolve()) in headers:
                continue
            if path.name.startswith('Icons') or path.name == 'icons.h':
                continue  # icon headers: generated ones (IconsFontAwesome6.h), the server node's own
            for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if raw_icon.search(line):
                    failures.append('%s:%d: an icon typed as bytes; use a macro from the icon headers'
                                    % (path.relative_to(root).as_posix(), number))

    for f in failures:
        print(f)
    print('%d icon macros checked, %d problems' % (total, len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
