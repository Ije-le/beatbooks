"""Convert a Beatbook Markdown file into an interactive HTML flipbook.

Sections flow as continuous content pages with headers — no separate chapter
title pages. Includes a cover, clickable table of contents, chapter-jump pills,
page-flip animation, keyboard navigation, and swipe support.

Usage:
    uv run python make_flipbook.py
    uv run python make_flipbook.py --input beatbook_college_park_v7.md --output beatbook_flipbook.html
"""

import argparse
import re
import threading
import webbrowser
import http.server
import socketserver
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

def parse_sections(md: str) -> tuple[str, str, list[dict]]:
    """Split markdown into (title, preamble, sections) based on ## headers."""
    lines = md.split("\n")
    title = ""
    preamble_lines = []
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("# ") and not title:
            title = line.lstrip("# ").strip()
        elif re.match(r"^#{1,2} ", line):
            body_start = i
            break
        else:
            preamble_lines.append(line)

    preamble = "\n".join(preamble_lines).strip()

    sections = []
    current_title = None
    current_lines = []

    for line in lines[body_start:]:
        if re.match(r"^#{1,2} ", line):
            if current_title is not None:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip()
                })
            current_title = re.sub(r"^#{1,2} ", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip()
        })

    return title, preamble, sections


def split_content_into_pages(content: str, max_chars: int = 2400) -> list[str]:
    """Split a long content block into pages at paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", content)
    pages = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            pages.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        pages.append("\n\n".join(current))

    return pages if pages else [""]


def md_to_html(text: str) -> str:
    """Convert a subset of markdown to HTML."""
    # Lines that are ONLY bold (e.g. **Name — Title**) become styled entry headers
    text = re.sub(
        r"^\*\*([^*]+)\*\*\s*$",
        r'<p class="entry-header"><strong>\1</strong></p>',
        text, flags=re.MULTILINE
    )
    # Remaining bold inline
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Blockquote
    text = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", text, flags=re.MULTILINE)
    # Horizontal rule
    text = re.sub(r"^---+$", r"<hr>", text, flags=re.MULTILINE)
    # Numbered list items
    text = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    # Bullet list items
    text = re.sub(r"^[-•] (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    # Wrap consecutive <li> in <ul>
    text = re.sub(r"(<li>.*?</li>\n?)+", lambda m: "<ul>" + m.group(0) + "</ul>", text, flags=re.DOTALL)
    # ### subheaders
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    # ## section headers (within content — used for sub-sections if any)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    # Paragraphs: wrap non-tag lines
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<"):
            result.append(stripped)
        else:
            result.append(f"<p>{stripped}</p>")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --page-width: 680px;
    --page-height: 880px;
    --bg: #1a1a2e;
    --paper: #fdf6e3;
    --paper-shadow: #c8b89a;
    --ink: #2c2c2c;
    --accent: #8b1a1a;
    --toc-bg: #f5efe0;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    min-height: 100vh;
    font-family: 'Georgia', serif;
    padding: 30px 20px 60px;
    user-select: none;
  }}

  h1.book-title {{
    color: #e8d5b0;
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 20px;
    text-align: center;
    opacity: 0.8;
  }}

  /* Book container */
  .book-wrapper {{
    perspective: 2000px;
    width: var(--page-width);
    height: var(--page-height);
    position: relative;
  }}

  .book {{
    width: 100%;
    height: 100%;
    position: relative;
    transform-style: preserve-3d;
  }}

  /* Pages */
  .page {{
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0; left: 0;
    background: var(--paper);
    border-radius: 2px 8px 8px 2px;
    box-shadow: 4px 4px 20px rgba(0,0,0,0.5), inset -3px 0 6px rgba(0,0,0,0.08);
    display: none;
    overflow: hidden;
    backface-visibility: hidden;
  }}

  .page.active {{ display: block; }}

  .page.flip-out {{
    display: block;
    animation: flipOut 0.42s ease-in forwards;
    transform-origin: left center;
    z-index: 10;
  }}
  .page.flip-in {{
    display: block;
    animation: flipIn 0.42s ease-out forwards;
    transform-origin: left center;
    z-index: 10;
  }}
  .page.flip-back-out {{
    display: block;
    animation: flipBackOut 0.42s ease-in forwards;
    transform-origin: left center;
    z-index: 10;
  }}
  .page.flip-back-in {{
    display: block;
    animation: flipBackIn 0.42s ease-out forwards;
    transform-origin: left center;
    z-index: 10;
  }}

  @keyframes flipOut {{
    0%   {{ transform: rotateY(0deg) scaleX(1); opacity: 1; }}
    50%  {{ transform: rotateY(-50deg) scaleX(0.8); opacity: 0.7; }}
    100% {{ transform: rotateY(-95deg) scaleX(0.1); opacity: 0.2; }}
  }}
  @keyframes flipIn {{
    0%   {{ transform: rotateY(95deg) scaleX(0.1); opacity: 0.2; }}
    50%  {{ transform: rotateY(50deg) scaleX(0.8); opacity: 0.7; }}
    100% {{ transform: rotateY(0deg) scaleX(1); opacity: 1; }}
  }}
  @keyframes flipBackOut {{
    0%   {{ transform: rotateY(0deg) scaleX(1); opacity: 1; }}
    50%  {{ transform: rotateY(50deg) scaleX(0.8); opacity: 0.7; }}
    100% {{ transform: rotateY(95deg) scaleX(0.1); opacity: 0.2; }}
  }}
  @keyframes flipBackIn {{
    0%   {{ transform: rotateY(-95deg) scaleX(0.1); opacity: 0.2; }}
    50%  {{ transform: rotateY(-50deg) scaleX(0.8); opacity: 0.7; }}
    100% {{ transform: rotateY(0deg) scaleX(1); opacity: 1; }}
  }}

  /* Left spine shadow */
  .page::before {{
    content: '';
    position: absolute;
    left: 0; top: 0;
    width: 18px; height: 100%;
    background: linear-gradient(to right, rgba(0,0,0,0.12), transparent);
    pointer-events: none;
    z-index: 1;
  }}

  /* Page content */
  .page-inner {{
    padding: 48px 52px 40px 52px;
    height: 100%;
    overflow-y: auto;
    color: var(--ink);
    line-height: 1.75;
    font-size: 0.93rem;
    scrollbar-width: thin;
    scrollbar-color: var(--paper-shadow) transparent;
  }}

  .page-inner::-webkit-scrollbar {{ width: 4px; }}
  .page-inner::-webkit-scrollbar-thumb {{ background: var(--paper-shadow); border-radius: 2px; }}

  .page-inner p {{ margin-bottom: 1em; }}
  .page-inner h2 {{
    font-size: 1.3rem;
    color: var(--accent);
    margin: 0 0 1em 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8d5b0;
    font-weight: bold;
    letter-spacing: 0.01em;
  }}
  .page-inner h3 {{
    font-size: 1.05rem;
    color: #5a3010;
    margin: 1.2em 0 0.5em;
    font-weight: bold;
  }}
  .page-inner p.entry-header {{
    margin-top: 1.4em;
    margin-bottom: 0.3em;
    color: var(--accent);
    font-size: 1rem;
    border-bottom: 1px solid #e8d5b0;
    padding-bottom: 2px;
  }}
  .page-inner strong {{ color: #1a1a1a; }}
  .page-inner ul {{ margin: 0.5em 0 1em 1.4em; }}
  .page-inner li {{ margin-bottom: 0.4em; }}
  .page-inner blockquote {{
    border-left: 3px solid var(--accent);
    padding: 0.3em 0 0.3em 1em;
    margin: 1em 0;
    color: #555;
    font-style: italic;
  }}
  .page-inner hr {{ border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }}

  /* Cover page */
  .page-cover .page-inner {{
    background: linear-gradient(160deg, #2c1810 0%, #4a2515 50%, #2c1810 100%);
    color: #fdf6e3;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 60px 50px;
  }}

  .cover-title {{
    font-size: 1.9rem;
    font-weight: bold;
    color: #e8d5b0;
    line-height: 1.3;
    margin-bottom: 24px;
  }}
  .cover-subtitle {{
    font-size: 0.85rem;
    color: #c4a882;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 20px;
  }}
  .cover-tagline {{
    font-size: 0.88rem;
    color: #d4c4a8;
    font-style: italic;
    line-height: 1.6;
    max-width: 420px;
    margin-bottom: 32px;
  }}
  .cover-meta {{
    font-size: 0.8rem;
    color: #ddd0b8;
    line-height: 2;
  }}
  .cover-meta strong {{
    color: #f0e4cc;
  }}
  .page-cover .page-inner blockquote {{
    color: #cfc0a0;
    border-left-color: #8b5e3c;
    font-style: italic;
  }}

  /* Back cover */
  .page-back-cover .page-inner {{
    background: linear-gradient(160deg, #2c1810 0%, #4a2515 50%, #2c1810 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 60px 50px;
  }}
  .back-cover-label {{
    font-size: 0.75rem;
    color: #a08060;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 28px;
  }}
  .back-cover-title {{
    font-size: 1.6rem;
    font-weight: bold;
    color: #e8d5b0;
    line-height: 1.35;
    margin-bottom: 10px;
  }}
  .back-cover-place {{
    font-size: 0.9rem;
    color: #c4a882;
    letter-spacing: 0.08em;
    margin-bottom: 32px;
  }}
  .back-cover-divider {{
    width: 50px;
    height: 2px;
    background: #8b5e3c;
    margin: 0 auto 32px;
  }}
  .back-cover-tagline {{
    font-size: 0.85rem;
    color: #b8a888;
    line-height: 1.7;
    max-width: 400px;
    font-style: italic;
  }}
  .back-cover-date {{
    margin-top: 40px;
    font-size: 0.75rem;
    color: #7a6040;
    letter-spacing: 0.1em;
  }}
  .cover-divider {{
    width: 60px;
    height: 2px;
    background: #8b5e3c;
    margin: 24px auto;
  }}

  /* TOC page */
  .page-toc .page-inner {{
    padding: 48px 52px;
    background: var(--toc-bg);
  }}
  .toc-title {{
    font-size: 1.2rem;
    font-weight: bold;
    color: var(--accent);
    margin-bottom: 24px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  .toc-item {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 6px 0;
    border-bottom: 1px dotted #c8b89a;
    cursor: pointer;
    transition: color 0.2s;
  }}
  .toc-item:hover {{ color: var(--accent); }}
  .toc-item-title {{ font-size: 0.9rem; }}
  .toc-item-page {{ font-size: 0.8rem; color: #888; }}

  /* Page number footer */
  .page-footer {{
    position: absolute;
    bottom: 18px;
    left: 0; right: 0;
    text-align: center;
    font-size: 0.75rem;
    color: #aaa;
    font-style: italic;
    pointer-events: none;
  }}

  /* Navigation arrows */
  .nav-btn {{
    position: fixed;
    top: 60%;
    background: rgba(58, 42, 26, 0.85);
    color: #e8d5b0;
    border: none;
    width: 44px;
    height: 80px;
    font-size: 1.4rem;
    font-family: Georgia, serif;
    cursor: pointer;
    border-radius: 6px;
    transition: background 0.2s, transform 0.1s;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    backdrop-filter: blur(4px);
  }}
  #prevBtn {{ left: 10px; }}
  #nextBtn {{ right: 10px; }}
  .nav-btn:hover {{ background: rgba(92, 56, 32, 0.95); }}
  .nav-btn:active {{ transform: scale(0.95); }}
  .nav-btn:disabled {{ opacity: 0.2; cursor: default; }}

  .nav {{
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 16px;
  }}
  .page-indicator {{
    color: #a08060;
    font-size: 0.85rem;
    font-family: Georgia, serif;
    min-width: 100px;
    text-align: center;
  }}

  /* Section jump pills */
  .chapter-nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
    max-width: var(--page-width);
    justify-content: center;
  }}
  .chapter-pill {{
    background: #2e2014;
    color: #c4a882;
    border: 1px solid #5c3820;
    padding: 4px 14px;
    font-size: 0.72rem;
    border-radius: 20px;
    cursor: pointer;
    font-family: Georgia, serif;
    transition: background 0.2s;
    white-space: nowrap;
  }}
  .chapter-pill:hover {{ background: #5c3820; color: #e8d5b0; }}
  .chapter-pill.active {{ background: #8b1a1a; color: #fdf6e3; border-color: #8b1a1a; }}

  @media (max-width: 740px) {{
    :root {{
      --page-width: 95vw;
      --page-height: 85vh;
    }}
    .cover-title {{ font-size: 1.4rem; }}
  }}
</style>
</head>
<body>

<h1 class="book-title">{title}</h1>

<div class="book-wrapper">
  <div class="book" id="book">
    {pages_html}
  </div>
</div>

<button class="nav-btn" id="prevBtn" onclick="flipPage(-1)">&#8592;</button>
<button class="nav-btn" id="nextBtn" onclick="flipPage(1)">&#8594;</button>

<div class="nav">
  <span class="page-indicator" id="pageIndicator">Page 1 of {total_pages}</span>
</div>

<div class="chapter-nav" id="chapterNav">
  {chapter_pills}
</div>

<script>
  const TOTAL = {total_pages};
  let current = 0;
  let animating = false;
  const tocMap = {toc_map};

  function showPage(index, direction) {{
    if (animating) return;
    const pages = document.querySelectorAll('.page');
    if (index < 0 || index >= TOTAL) return;

    animating = true;
    const outPage = pages[current];
    const inPage  = pages[index];

    const outClass = direction > 0 ? 'flip-out'      : 'flip-back-out';
    const inClass  = direction > 0 ? 'flip-in'       : 'flip-back-in';

    outPage.classList.add(outClass);

    setTimeout(() => {{
      outPage.classList.remove('active', outClass);
      inPage.classList.add(inClass, 'active');
      setTimeout(() => {{
        inPage.classList.remove(inClass);
        animating = false;
      }}, 430);
    }}, 420);

    current = index;
    updateUI();
  }}

  function flipPage(dir) {{ showPage(current + dir, dir); }}

  function goToSection(pageIndex) {{
    const dir = pageIndex >= current ? 1 : -1;
    showPage(pageIndex, dir);
  }}

  function updateUI() {{
    document.getElementById('pageIndicator').textContent =
      'Page ' + (current + 1) + ' of ' + TOTAL;
    document.getElementById('prevBtn').disabled = current === 0;
    document.getElementById('nextBtn').disabled = current === TOTAL - 1;

    // Highlight active section pill
    const starts = Object.values(tocMap).sort((a, b) => a - b);
    let active = null;
    for (let i = starts.length - 1; i >= 0; i--) {{
      if (current >= starts[i]) {{ active = starts[i]; break; }}
    }}
    document.querySelectorAll('.chapter-pill').forEach(pill => {{
      pill.classList.toggle('active', parseInt(pill.dataset.page) === active);
    }});
  }}

  // Keyboard navigation
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') flipPage(1);
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   flipPage(-1);
  }});

  // Swipe navigation — attached to whole book
  let touchStartX = 0;
  let touchStartY = 0;
  const bookEl = document.getElementById('book');

  bookEl.addEventListener('touchstart', e => {{
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  }}, {{ passive: true }});

  bookEl.addEventListener('touchmove', e => {{
    const dx = e.touches[0].clientX - touchStartX;
    const dy = e.touches[0].clientY - touchStartY;
    if (Math.abs(dx) > Math.abs(dy)) e.preventDefault();
  }}, {{ passive: false }});

  bookEl.addEventListener('touchend', e => {{
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;
    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 40) {{
      flipPage(dx < 0 ? 1 : -1);
    }}
  }}, {{ passive: true }});

  // Init
  updateUI();
</script>
</body>
</html>
"""


def build_page(content_html: str, page_num: int, extra_class: str = "") -> str:
    cls = ("page " + extra_class).strip()
    active = " active" if page_num == 0 else ""
    footer = f'<div class="page-footer">{page_num + 1}</div>'
    return (
        f'<div class="{cls}{active}" id="page-{page_num}">\n'
        f'  <div class="page-inner">{content_html}</div>\n'
        f'  {footer}\n'
        f'</div>\n'
    )


def build_cover(title: str, preamble: str) -> str:
    meta_html = md_to_html(preamble) if preamble else ""
    tagline = "A reporter\u2019s guide to the people, places, and persistent fights shaping one of Maryland\u2019s most dynamic university cities."
    html = (
        f'<div class="cover-title">{title}</div>'
        f'<div class="cover-divider"></div>'
        f'<div class="cover-subtitle">College Park, Maryland</div>'
        f'<div class="cover-tagline">{tagline}</div>'
        f'<div class="cover-meta">{meta_html}</div>'
    )
    return (
        f'<div class="page page-cover active" id="page-0">'
        f'<div class="page-inner">{html}</div>'
        f'</div>\n'
    )


def build_toc(sections: list[dict], page_map: dict) -> str:
    items = ""
    for sec in sections:
        pg = page_map.get(sec["title"], 0)
        items += (
            f'<div class="toc-item" onclick="goToSection({pg})">'
            f'<span class="toc-item-title">{sec["title"]}</span>'
            f'<span class="toc-item-page">{pg + 1}</span>'
            f'</div>\n'
        )
    return f'<div class="toc-title">Contents</div>\n{items}'


def make_flipbook(input_path: Path, output_path: Path):
    md = input_path.read_text(encoding="utf-8")
    title, preamble, sections = parse_sections(md)

    pages_html = ""
    page_index = 0
    toc_map = {}
    section_pills = []

    # Page 0: Cover
    pages_html += build_cover(title, preamble)
    page_index += 1

    # Page 1: TOC (placeholder — filled in after all pages are built)
    toc_placeholder = "__TOC__"
    pages_html += (
        f'<div class="page page-toc" id="page-{page_index}">'
        f'<div class="page-inner">{toc_placeholder}</div>'
        f'<div class="page-footer">{page_index + 1}</div>'
        f'</div>\n'
    )
    toc_page_index = page_index
    page_index += 1

    # Content pages — no chapter title pages, sections flow with h2 headers
    for section in sections:
        sec_title = section["title"]
        toc_map[sec_title] = page_index
        section_pills.append((sec_title, page_index))

        content_pages = split_content_into_pages(section["content"])

        for i, content_chunk in enumerate(content_pages):
            chunk_html = md_to_html(content_chunk)
            # Prepend section title as h2 on the first page of each section
            if i == 0:
                chunk_html = f'<h2>{sec_title}</h2>\n' + chunk_html
            pages_html += build_page(chunk_html, page_index)
            page_index += 1

    # Fill in TOC now that page numbers are known
    toc_html = build_toc(sections, toc_map)
    pages_html = pages_html.replace(toc_placeholder, toc_html)

    # Build section jump pills
    pills_html = (
        f'<button class="chapter-pill" data-page="0" onclick="goToSection(0)">Cover</button>\n'
        f'<button class="chapter-pill" data-page="{toc_page_index}" onclick="goToSection({toc_page_index})">Contents</button>\n'
    )
    for sec_title, pg in section_pills:
        short = sec_title[:28] + "…" if len(sec_title) > 28 else sec_title
        pills_html += (
            f'<button class="chapter-pill" data-page="{pg}" '
            f'onclick="goToSection({pg})">{short}</button>\n'
        )

    # Back cover (last page)
    import datetime
    month_year = datetime.date.today().strftime("%B %Y")
    back_cover_html = (
        f'<div class="page page-back-cover" id="page-{page_index}">\n'
        f'  <div class="page-inner">\n'
        f'    <div class="back-cover-date">Generated {month_year}</div>\n'
        f'  </div>\n'
        f'  <div class="page-footer">{page_index + 1}</div>\n'
        f'</div>\n'
    )
    pages_html += back_cover_html
    page_index += 1
    total_pages = page_index

    toc_js = "{" + ", ".join(f'"{k}": {v}' for k, v in toc_map.items()) + "}"

    html = HTML_TEMPLATE.format(
        title=title,
        pages_html=pages_html,
        total_pages=total_pages,
        chapter_pills=pills_html,
        toc_map=toc_js,
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Flipbook written to {output_path} ({total_pages} pages)")


def main():
    parser = argparse.ArgumentParser(description="Convert Beatbook Markdown to an HTML flipbook.")
    parser.add_argument("--input",  default="beatbook_college_park_v7.md", type=Path)
    parser.add_argument("--output", default="beatbook_flipbook.html",       type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.")
        return

    make_flipbook(args.input, args.output)

    output_dir = str(args.output.parent.resolve())
    filename   = args.output.name
    os.chdir(output_dir)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # silence logs

    port = 8080
    while port < 8100:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            break
        except OSError:
            port += 1
    else:
        print("Error: no available port between 8080 and 8099.")
        return

    url = f"http://localhost:{port}/{filename}"
    print(f"Serving at {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    with httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
