"""Convert a Beatbook Markdown file into an interactive HTML flipbook.

Each major section becomes a chapter with a title page followed by
content pages. Long sections are split into sub-pages automatically.
Includes a clickable table of contents and page-flip animation.

Usage:
    uv run python make_flipbook.py
    uv run python make_flipbook.py --input beatbook_college_park_v5.md --output beatbook_flipbook.html
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

def parse_sections(md: str) -> list[dict]:
    """Split markdown into sections based on ## headers."""
    # Extract document title and preamble
    lines = md.split("\n")
    title = ""
    preamble_lines = []
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("# ") and not title:
            title = line.lstrip("# ").strip()
        elif line.startswith("## ") or line.startswith("# "):
            body_start = i
            break
        else:
            preamble_lines.append(line)

    preamble = "\n".join(preamble_lines).strip()

    # Split into sections by ## or # headers
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


def split_content_into_pages(content: str, max_chars: int = 2200) -> list[str]:
    """Split a long content block into pages, breaking at paragraph boundaries."""
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
    # Lines that are ONLY bold text (e.g. **Name, Title**) become styled entry headers
    text = re.sub(r"^\*\*([^*]+)\*\*\s*$", r"<p class=\"entry-header\"><strong>\1</strong></p>", text, flags=re.MULTILINE)
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
    # ### headers
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    # ## headers
    text = re.sub(r"^## (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
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
    --chapter-bg: #2c1810;
    --chapter-text: #fdf6e3;
    --toc-bg: #f5efe0;
    --spine: #6b3a2a;
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

  .page.active {{
    display: block;
  }}

  .page.flip-out {{
    display: block;
    animation: flipOut 0.45s ease-in forwards;
    transform-origin: left center;
    z-index: 10;
  }}

  .page.flip-in {{
    display: block;
    animation: flipIn 0.45s ease-out forwards;
    transform-origin: left center;
    z-index: 10;
  }}

  .page.flip-back-out {{
    display: block;
    animation: flipBackOut 0.45s ease-in forwards;
    transform-origin: left center;
    z-index: 10;
  }}

  .page.flip-back-in {{
    display: block;
    animation: flipBackIn 0.45s ease-out forwards;
    transform-origin: left center;
    z-index: 10;
  }}

  @keyframes flipOut {{
    0%   {{ transform: rotateY(0deg); opacity: 1; }}
    100% {{ transform: rotateY(-100deg); opacity: 0.3; }}
  }}
  @keyframes flipIn {{
    0%   {{ transform: rotateY(100deg); opacity: 0.3; }}
    100% {{ transform: rotateY(0deg); opacity: 1; }}
  }}
  @keyframes flipBackOut {{
    0%   {{ transform: rotateY(0deg); opacity: 1; }}
    100% {{ transform: rotateY(100deg); opacity: 0.3; }}
  }}
  @keyframes flipBackIn {{
    0%   {{ transform: rotateY(-100deg); opacity: 0.3; }}
    100% {{ transform: rotateY(0deg); opacity: 1; }}
  }}

  /* Page spine shadow */
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
  .page-inner p.entry-header {{
    margin-top: 1.4em;
    margin-bottom: 0.3em;
    color: var(--accent);
    font-size: 1rem;
    border-bottom: 1px solid #e8d5b0;
    padding-bottom: 2px;
  }}
  .page-inner h3 {{ font-size: 1.1rem; color: var(--accent); margin: 1.2em 0 0.5em; }}
  .page-inner h4 {{ font-size: 1rem; color: var(--accent); margin: 1em 0 0.4em; }}
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
    color: var(--chapter-text);
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
    letter-spacing: 0.02em;
  }}

  .cover-subtitle {{
    font-size: 0.85rem;
    color: #c4a882;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 40px;
  }}

  .cover-meta {{
    font-size: 0.8rem;
    color: #a08060;
    line-height: 2;
  }}

  .cover-divider {{
    width: 60px;
    height: 2px;
    background: #8b5e3c;
    margin: 24px auto;
  }}

  /* Chapter title page */
  .page-chapter .page-inner {{
    background: linear-gradient(170deg, #3a2010 0%, #5c3020 100%);
    color: var(--chapter-text);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding: 60px 52px;
  }}

  .chapter-number {{
    font-size: 0.75rem;
    color: #a08060;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }}

  .chapter-title {{
    font-size: 1.7rem;
    color: #e8d5b0;
    line-height: 1.25;
    font-weight: bold;
    max-width: 520px;
  }}

  .chapter-rule {{
    width: 50px;
    height: 2px;
    background: #8b5e3c;
    margin-top: 28px;
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

  /* Navigation */
  .nav {{
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 16px;
  }}

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

  .page-indicator {{
    color: #a08060;
    font-size: 0.85rem;
    font-family: Georgia, serif;
    min-width: 100px;
    text-align: center;
  }}

  /* Chapter nav pills */
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
    const inPage = pages[index];

    const outClass = direction > 0 ? 'flip-out' : 'flip-back-out';
    const inClass = direction > 0 ? 'flip-in' : 'flip-back-in';

    outPage.classList.add(outClass);

    setTimeout(() => {{
      outPage.classList.remove('active', outClass);
      inPage.classList.add(inClass, 'active');
      setTimeout(() => {{
        inPage.classList.remove(inClass);
        animating = false;
      }}, 460);
    }}, 440);

    current = index;
    updateUI();
  }}

  function flipPage(dir) {{
    showPage(current + dir, dir);
  }}

  function goToChapter(pageIndex) {{
    const dir = pageIndex > current ? 1 : -1;
    showPage(pageIndex, dir);
  }}

  function updateUI() {{
    document.getElementById('pageIndicator').textContent = 'Page ' + (current + 1) + ' of ' + TOTAL;
    document.getElementById('prevBtn').disabled = current === 0;
    document.getElementById('nextBtn').disabled = current === TOTAL - 1;

    // Update active chapter pill
    document.querySelectorAll('.chapter-pill').forEach(pill => {{
      pill.classList.remove('active');
    }});
    // Find which chapter we're in
    const chapterStarts = Object.values(tocMap).sort((a,b) => a-b);
    let activeChapter = null;
    for (let i = chapterStarts.length - 1; i >= 0; i--) {{
      if (current >= chapterStarts[i]) {{
        activeChapter = chapterStarts[i];
        break;
      }}
    }}
    if (activeChapter !== null) {{
      document.querySelectorAll('.chapter-pill').forEach(pill => {{
        if (parseInt(pill.dataset.page) === activeChapter) {{
          pill.classList.add('active');
        }}
      }});
    }}
  }}

  // Keyboard navigation
  document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') flipPage(1);
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') flipPage(-1);
  }});

  // Swipe navigation
  let touchStartX = 0;
  let touchStartY = 0;
  const bookEl = document.getElementById('book');

  bookEl.addEventListener('touchstart', e => {{
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  }}, {{ passive: true }});

  bookEl.addEventListener('touchmove', e => {{
    // Prevent vertical scroll hijack only when clearly swiping horizontally
    const dx = e.touches[0].clientX - touchStartX;
    const dy = e.touches[0].clientY - touchStartY;
    if (Math.abs(dx) > Math.abs(dy)) e.preventDefault();
  }}, {{ passive: false }});

  bookEl.addEventListener('touchend', e => {{
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;
    if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 40) {{
      if (dx < 0) flipPage(1);
      else flipPage(-1);
    }}
  }}, {{ passive: true }});

  // Init
  updateUI();
</script>
</body>
</html>
"""


def build_page(content_html: str, page_type: str, page_num: int, total: int) -> str:
    footer = f'<div class="page-footer">{page_num + 1}</div>'
    return f'''<div class="page {page_type}" id="page-{page_num}">
  <div class="page-inner">{content_html}</div>
  {footer}
</div>'''


def build_cover(title: str, preamble: str) -> str:
    meta_html = md_to_html(preamble) if preamble else ""
    html = f'''
    <div class="cover-title">{title}</div>
    <div class="cover-divider"></div>
    <div class="cover-subtitle">College Park, Maryland</div>
    <div class="cover-meta">{meta_html}</div>
    '''
    return f'<div class="page page-cover active" id="page-0"><div class="page-inner">{html}</div></div>'


def build_toc(sections: list[dict], page_map: dict) -> str:
    items = ""
    for sec in sections:
        pg = page_map.get(sec["title"], 0) + 1
        items += f'''<div class="toc-item" onclick="goToChapter({page_map.get(sec['title'], 0)})">
      <span class="toc-item-title">{sec['title']}</span>
      <span class="toc-item-page">{pg}</span>
    </div>\n'''
    return f'<div class="toc-title">Contents</div>\n{items}'


def make_flipbook(input_path: Path, output_path: Path):
    md = input_path.read_text(encoding="utf-8")
    title, preamble, sections = parse_sections(md)

    pages_html = ""
    page_index = 0
    toc_map = {}  # section title -> page index
    chapter_pills_data = []  # (title, page_index)

    # Page 0: Cover
    pages_html += build_cover(title, preamble)
    page_index += 1

    # Page 1: TOC placeholder (filled after we know all page indices)
    toc_page_index = page_index
    pages_html += f'<div class="page page-toc" id="page-{page_index}"><div class="page-inner">__TOC__</div><div class="page-footer">{page_index + 1}</div></div>\n'
    page_index += 1

    # Build content pages
    for i, section in enumerate(sections):
        sec_title = section["title"]
        toc_map[sec_title] = page_index
        chapter_pills_data.append((sec_title, page_index))

        # Chapter title page
        chapter_html = f'''
        <div class="chapter-number">Chapter {i + 1}</div>
        <div class="chapter-title">{sec_title}</div>
        <div class="chapter-rule"></div>
        '''
        pages_html += f'<div class="page page-chapter" id="page-{page_index}"><div class="page-inner">{chapter_html}</div><div class="page-footer">{page_index + 1}</div></div>\n'
        page_index += 1

        # Content pages
        content_pages = split_content_into_pages(section["content"])
        for content in content_pages:
            content_html = md_to_html(content)
            pages_html += build_page(content_html, "", page_index, 0)
            pages_html += "\n"
            page_index += 1

    total_pages = page_index

    # Build TOC html now that we have page numbers
    toc_html = build_toc(sections, toc_map)
    pages_html = pages_html.replace("__TOC__", toc_html)

    # Build chapter pills
    pills_html = '<button class="chapter-pill" data-page="0" onclick="goToChapter(0)">Cover</button>\n'
    pills_html += f'<button class="chapter-pill" data-page="{toc_page_index}" onclick="goToChapter({toc_page_index})">Contents</button>\n'
    for sec_title, pg in chapter_pills_data:
        short = sec_title[:28] + "…" if len(sec_title) > 28 else sec_title
        pills_html += f'<button class="chapter-pill" data-page="{pg}" onclick="goToChapter({pg})">{short}</button>\n'

    # TOC map for JS
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
    parser.add_argument("--input", default="beatbook_college_park_v5.md", type=Path)
    parser.add_argument("--output", default="beatbook_flipbook.html", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.")
        return

    make_flipbook(args.input, args.output)

    # Serve the file and open in browser
    output_dir = str(args.output.parent.resolve())
    filename = args.output.name

    os.chdir(output_dir)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None  # silence server logs

    # Find an available port starting from 8080
    port = 8080
    while port < 8100:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            break
        except OSError:
            port += 1
    else:
        print("Error: could not find an available port between 8080 and 8099.")
        return

    url = f"http://localhost:{port}/{filename}"
    print(f"Serving at {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    with httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
