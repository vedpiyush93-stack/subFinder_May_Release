"""Visual identity for the subFinder Space.

The palette is the paper's, and it carries meaning rather than brand: deep teal
for anything the tool is standing behind, amber where the call is real but close,
rose for anything it is telling you not to trust. Those three appear nowhere else,
so a colour on this page is always a claim about the result beside it.

The page is a console, not a document -- read at a glance, not top to bottom -- so
the grounds are graded (a tinted page, white cards) to make the panels read as
objects, and every number that shares a column is set in tabular mono.
"""
import gradio as gr

INK    = "#101a1d"
DEEP   = "#0e5c6b"
TEAL   = "#17635f"
MINT   = "#e4efee"
ROSE   = "#a8434b"
SAND   = "#b8791f"
PAPER  = "#eaf0f0"      # page ground -- deeper than the cards, so cards read as objects
CARD   = "#ffffff"
RULE   = "#d5e0e1"
MUTED  = "#57666c"
FAINT  = "#85989d"

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&"
         "family=Inter:wght@400;500;600;700&"
         "family=JetBrains+Mono:wght@400;500;600&display=swap")

CSS = f"""
@import url('{FONTS}');

:root, .dark {{
  --sf-ink:{INK}; --sf-deep:{DEEP}; --sf-teal:{TEAL}; --sf-mint:{MINT};
  --sf-rose:{ROSE}; --sf-sand:{SAND}; --sf-paper:{PAPER}; --sf-card:{CARD};
  --sf-rule:{RULE}; --sf-muted:{MUTED}; --sf-faint:{FAINT};
  --color-accent:{TEAL}; --color-accent-soft:{MINT};
  --body-text-color:{INK};
}}

/* ---------------------------------------------------------------- shell
   The container had a max-width and no auto margin, so on any screen wider
   than the measure it pinned to the left and left the rest of the window
   empty. Centre it, and widen it -- the results table wants the room. */
.gradio-container {{
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 26px 24px 60px !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: transparent !important;
  color: var(--sf-ink) !important;
}}
body, gradio-app {{
  background:
    radial-gradient(1100px 460px at 50% -180px, #dce9e9 0%, transparent 70%),
    var(--sf-paper) !important;
  min-height: 100vh;
}}
footer {{ display:none !important; }}
.gradio-container .prose :is(h1,h2,h3,h4) {{ font-family:'Fraunces', Georgia, serif; }}

/* Gradio paints a white ground on every block; that is what made the page read
   as one flat sheet. Let the rows be transparent and only the real panels white. */
.gradio-container .form,
.gradio-container .block,
.gradio-container .gap {{ background: transparent !important; }}

/* ---------------------------------------------------------------- masthead */
#sf-hero {{
  background:
    radial-gradient(760px 300px at 88% -30%, rgba(120,220,205,.20), transparent 62%),
    linear-gradient(135deg, {DEEP} 0%, #10504f 52%, #0b3630 100%);
  border-radius: 20px; padding: 38px 42px 34px; margin-bottom: 22px;
  position: relative; overflow: hidden;
  box-shadow: 0 18px 40px -26px rgba(10,50,52,.85);
}}
#sf-hero::before {{
  content:""; position:absolute; inset:0; opacity:.16; pointer-events:none;
  background-image:
    linear-gradient(rgba(255,255,255,.30) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.30) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: radial-gradient(660px 300px at 82% 0%, #000, transparent 70%);
  -webkit-mask-image: radial-gradient(660px 300px at 82% 0%, #000, transparent 70%);
}}
#sf-hero > * {{ position:relative; }}
#sf-hero h1 {{
  font-family:'Fraunces', Georgia, serif; font-weight:600;
  font-size: 2.7rem; color:#fff; margin:0 0 8px; letter-spacing:-.022em;
}}
#sf-hero .sf-sub {{
  color:#c5e0dc; font-size:1.045rem; max-width:62ch; line-height:1.58; margin:0;
}}
#sf-hero .sf-chips {{ margin-top:22px; display:flex; gap:9px; flex-wrap:wrap; }}
#sf-hero .sf-chip {{
  background: rgba(255,255,255,.10); color:#cfe6e3; border:1px solid rgba(255,255,255,.17);
  padding:6px 13px; border-radius:999px; font-size:.785rem; font-weight:500;
  letter-spacing:.015em; backdrop-filter: blur(2px);
}}
#sf-hero .sf-chip b {{
  color:#fff; font-weight:600;
  font-family:'JetBrains Mono', monospace; font-variant-numeric:tabular-nums;
}}

/* ---------------------------------------------------------------- panels */
.sf-card {{
  background: var(--sf-card); border:1px solid var(--sf-rule); border-radius:16px;
  padding:22px 24px; box-shadow:0 1px 2px rgba(16,40,44,.05), 0 12px 26px -22px rgba(16,52,56,.5);
}}
.sf-label {{
  font-size:.7rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase;
  color:var(--sf-teal); margin:0 0 10px;
}}
/* the two console columns */
#sf-console {{ align-items: stretch; }}
#sf-console > div {{ min-width: 0; }}
#sf-inputs, #sf-dials {{
  background: var(--sf-card); border:1px solid var(--sf-rule); border-radius:16px;
  padding:18px 20px 20px; box-shadow:0 1px 2px rgba(16,40,44,.05), 0 12px 26px -22px rgba(16,52,56,.5);
}}
#sf-dials .sf-label:first-child {{ margin-top:2px; }}

/* ---------------------------------------------------------------- gradio parts */
.tab-button {{
  font-weight:600 !important; letter-spacing:.005em; flex:0 0 auto !important;
  padding:10px 18px !important;
}}
.tab-button.active {{ color:{TEAL} !important; border-bottom-color:{TEAL} !important; }}
.gradio-container textarea, .gradio-container input[type=text], .gradio-container input[type=number] {{
  font-family:'JetBrains Mono', monospace !important; font-size:.83rem !important;
  line-height:1.65 !important; background:#f7fafa !important;
  border-radius:10px !important; color:var(--sf-ink) !important;
}}
.gradio-container textarea:focus {{ box-shadow:0 0 0 3px {MINT} !important; border-color:{TEAL} !important; }}
button.primary {{
  background:{DEEP} !important; border:none !important; color:#fff !important;
  font-weight:600 !important; letter-spacing:.03em !important; border-radius:11px !important;
  box-shadow:0 10px 20px -12px rgba(14,92,107,.9) !important;
  transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}}
button.primary:hover {{
  background:{TEAL} !important; transform:translateY(-1px);
  box-shadow:0 14px 26px -12px rgba(14,92,107,.95) !important;
}}
.sf-dl button, button.sf-dl {{
  background:#fff !important; color:{DEEP} !important;
  border:1px solid var(--sf-rule) !important; border-radius:9px !important;
  font-size:.79rem !important; font-weight:600 !important; box-shadow:none !important;
}}
.sf-dl button:hover, button.sf-dl:hover {{
  border-color:{TEAL} !important; background:{MINT} !important;
}}
.sf-dl[disabled] button, button.sf-dl[disabled] {{ opacity:.45 !important; }}
/* the upload dropzone: it defaults to a very tall empty box */
#sf-upload .file-preview-holder, #sf-upload [data-testid="block-label"] + div {{ min-height:0; }}
#sf-upload .wrap {{ min-height:118px !important; }}
.label-wrap {{ font-weight:600 !important; color:{DEEP} !important; }}
.gradio-container .block-label, .gradio-container label span {{
  font-size:.8rem !important; color:var(--sf-muted) !important;
}}
.gradio-container [data-testid="block-info"], .gradio-container .info {{
  font-size:.78rem !important; line-height:1.55 !important; color:var(--sf-muted) !important;
}}
input[type=range] {{ accent-color:{TEAL}; }}
/* Gradio's markdown renders a size larger than the rest of the console.
   Scoped to a direct child so it cannot reach the hand-written HTML below:
   gr.HTML wraps its payload in .prose as well, but always inside a div. */
.gradio-container .prose > p:not([class*="sf-"]),
.gradio-container .prose > ul > li {{
  font-size:.865rem !important; line-height:1.6 !important; color:var(--sf-muted) !important;
}}
.gradio-container .prose code {{
  font-family:'JetBrains Mono',monospace; font-size:.78rem; background:#f1f5f5;
  border:1px solid #e4ebeb; color:{DEEP}; padding:1px 6px; border-radius:5px;
}}
.gradio-container .prose strong {{ color:var(--sf-ink); }}

/* ---------------------------------------------------------------- headline result */
.sf-verdict {{
  border-radius:14px; padding:20px 24px 18px; margin-bottom:14px;
  border:1px solid var(--sf-rule); background:linear-gradient(180deg,#fbfdfd,#f5f9f9);
  display:flex; align-items:flex-start; gap:30px; flex-wrap:wrap;
}}
.sf-verdict.ok    {{ border-left:5px solid {TEAL}; }}
.sf-verdict.warn  {{ border-left:5px solid {SAND}; }}
.sf-verdict.stop  {{ border-left:5px solid {ROSE}; }}
.sf-verdict .sf-name {{
  font-family:'Fraunces', Georgia, serif; font-size:1.86rem; font-weight:600;
  color:var(--sf-ink); line-height:1.12; letter-spacing:-.015em;
}}
.sf-verdict .sf-num {{
  font-family:'JetBrains Mono', monospace; font-size:1.8rem; font-weight:500;
  color:{DEEP}; line-height:1.15; font-variant-numeric:tabular-nums;
}}
.sf-verdict .sf-k {{
  font-size:.68rem; text-transform:uppercase; letter-spacing:.13em;
  color:var(--sf-muted); font-weight:700; margin-bottom:4px;
}}
.sf-verdict .sf-note {{
  flex-basis:100%; margin-top:2px; padding-top:14px;
  border-top:1px solid var(--sf-rule); font-size:.865rem; color:var(--sf-muted);
  max-width:82ch; line-height:1.6;
}}
/* the state of a call, said in words as well as in colour */
.sf-state {{
  display:inline-flex; align-items:center; gap:7px; padding:5px 12px 5px 10px;
  border-radius:999px; font-size:.74rem; font-weight:600; letter-spacing:.02em;
  border:1px solid transparent; white-space:nowrap;
}}
.sf-state::before {{ content:""; width:7px; height:7px; border-radius:50%; background:currentColor; }}
.sf-state.ok   {{ color:{TEAL}; background:{MINT};   border-color:#bcd9d5; }}
.sf-state.warn {{ color:{SAND}; background:#fbf1de; border-color:#eeddb8; }}
.sf-state.stop {{ color:{ROSE}; background:#f9ebec; border-color:#eccfd1; }}
.sf-verdict .sf-spacer {{ flex:1 1 auto; }}

/* ---------------------------------------------------------------- results table */
.sf-tbl {{ width:100%; border-collapse:collapse; font-size:.865rem; }}
.sf-tbl th {{
  text-align:left; font-size:.665rem; letter-spacing:.11em; text-transform:uppercase;
  color:var(--sf-muted); font-weight:700; padding:10px 12px;
  border-bottom:1.5px solid var(--sf-rule); white-space:nowrap;
}}
.sf-tbl td {{ padding:9px 12px; border-bottom:1px solid #eef3f3; vertical-align:middle; }}
.sf-tbl tbody tr:hover td {{ background:#f6faf9; }}
.sf-tbl tr.win td {{ background:{MINT}; }}
.sf-tbl tr.win:hover td {{ background:#dcebe9; }}
.sf-tbl tr.win td.sub {{ font-weight:600; box-shadow:inset 3px 0 0 {TEAL}; }}
/* top-ranked, but the tool is not standing behind it */
.sf-tbl tr.winoff td {{ background:#f7f3f3; }}
.sf-tbl tr.winoff:hover td {{ background:#f1ebeb; }}
.sf-tbl tr.winoff td.sub {{ font-weight:600; box-shadow:inset 3px 0 0 {ROSE}; }}
.sf-tbl tr.dim td {{ color:#9aa8ad; }}
.sf-tbl td.n {{ font-family:'JetBrains Mono', monospace; font-variant-numeric:tabular-nums; }}
.sf-tbl td.sub {{ font-weight:500; color:var(--sf-ink); }}
.sf-bar {{
  display:block; height:6px; border-radius:3px; min-width:2px;
  background:linear-gradient(90deg,{TEAL},{DEEP});
}}
.sf-bar.lo {{ background:#c8d7d8; }}
.sf-track {{ display:block; height:6px; border-radius:3px; background:#edf2f2; width:100%; }}
.sf-pill {{
  display:inline-block; padding:2px 10px; border-radius:999px; font-size:.715rem; font-weight:600;
}}
.sf-pill.yes {{ background:{MINT}; color:{TEAL}; border:1px solid #bcd9d5; }}
.sf-pill.no  {{ background:#f2f4f4; color:#93a1a5; border:1px solid #e6eaea; }}
.sf-gene {{
  display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.775rem;
  padding:2px 8px; border-radius:6px; background:#f1f5f5; color:#41565b;
  margin:1px 3px 1px 0; border:1px solid #e6ecec;
}}
.sf-gene.lit {{ background:{MINT}; color:{TEAL}; font-weight:600; border-color:#bcd9d5; }}
.sf-empty {{ color:#a3b0b4; font-style:italic; font-size:.82rem; }}

.sf-tailrow td {{ border-bottom:none; padding:4px 12px 0; }}
.sf-more {{
  background:none; border:none; cursor:pointer; padding:8px 2px;
  font-family:'Inter',sans-serif; font-size:.79rem; font-weight:600; color:{DEEP};
  letter-spacing:.01em;
}}
.sf-more::before {{ content:"+"; margin-right:7px; font-weight:600; opacity:.6; }}
.sf-more[data-open="1"]::before {{ content:"\2212"; }}
.sf-more:hover {{ color:{TEAL}; text-decoration:underline; }}

.sf-scroll {{ overflow-x:auto; }}
.sf-foot {{ font-size:.79rem; color:var(--sf-muted); margin-top:14px; line-height:1.65; }}
.sf-foot code {{ font-family:'JetBrains Mono', monospace; font-size:.755rem;
  background:#eaf1f1; padding:1px 5px; border-radius:4px; }}

/* ---------------------------------------------------------------- summary strip */
.sf-strip {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:12px;
  margin-bottom:18px;
}}
.sf-stat {{
  background:var(--sf-card); border:1px solid var(--sf-rule); border-radius:13px;
  padding:15px 17px 14px; box-shadow:0 10px 22px -22px rgba(16,52,56,.6);
}}
.sf-stat-k {{ font-size:.665rem; text-transform:uppercase; letter-spacing:.12em;
  color:var(--sf-muted); font-weight:700; }}
.sf-stat-v {{ font-family:'JetBrains Mono',monospace; font-size:1.72rem; font-weight:500;
  color:{DEEP}; line-height:1.12; margin-top:6px; font-variant-numeric:tabular-nums; }}
.sf-stat-s {{ font-size:.745rem; color:var(--sf-muted); margin-top:3px; line-height:1.4; }}

/* ---------------------------------------------------------------- sortable table */
.sf-tablebar {{ display:flex; justify-content:space-between; align-items:center;
  gap:14px; flex-wrap:wrap; margin-bottom:12px; }}
.sf-tools {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
.sf-search {{
  border:1px solid var(--sf-rule); border-radius:9px; padding:7px 12px; font-size:.83rem;
  font-family:'Inter',sans-serif; min-width:200px; background:#f7fafa; color:var(--sf-ink);
}}
.sf-search:focus {{ outline:none; box-shadow:0 0 0 3px {MINT}; border-color:{TEAL}; }}
.sf-check {{ font-size:.81rem; color:var(--sf-muted); display:flex; align-items:center;
  gap:7px; cursor:pointer; }}
.sf-check input {{ accent-color:{TEAL}; }}
.sf-count {{ font-size:.755rem; color:var(--sf-muted); font-family:'JetBrains Mono',monospace;
  font-variant-numeric:tabular-nums; }}
.sf-tallscroll {{ max-height:580px; overflow-y:auto; border:1px solid var(--sf-rule);
  border-radius:11px; }}
.sf-sortable th {{
  cursor:pointer; user-select:none; position:sticky; top:0; background:#fff; z-index:2;
  padding-right:22px !important;
}}
.sf-sortable th:hover {{ color:{TEAL}; }}
.sf-sortable th::after {{ content:"↕"; opacity:.28; margin-left:5px; font-size:.85em; }}
.sf-sortable th.asc::after  {{ content:"↑"; opacity:1; color:{TEAL}; }}
.sf-sortable th.desc::after {{ content:"↓"; opacity:1; color:{TEAL}; }}
.sf-tbl td.mono {{ font-family:'JetBrains Mono',monospace; font-size:.785rem; }}

.sf-fmt-row {{ margin-bottom:15px; }}
.sf-fmt-row b {{ font-size:.85rem; color:var(--sf-ink); }}
.sf-fmt-row span {{ display:block; font-size:.79rem; color:var(--sf-muted);
  line-height:1.55; margin:3px 0 6px; }}
.sf-fmt-row pre {{
  background:#f4f8f8; border:1px solid var(--sf-rule); border-radius:8px;
  padding:9px 11px; overflow-x:auto; font-family:'JetBrains Mono',monospace;
  font-size:.735rem; color:#3d5257; line-height:1.55; margin:0;
}}
.sf-fmt-row code {{ font-family:'JetBrains Mono',monospace; font-size:.74rem;
  background:#eef3f3; padding:1px 5px; border-radius:4px; }}
.sf-fmt-note {{ font-size:.78rem; color:var(--sf-teal); background:{MINT};
  border:1px solid #bcd9d5; border-radius:9px; padding:9px 12px; margin:0; line-height:1.55; }}
.sf-cheat {{ margin-top:20px; padding-top:18px; border-top:1px solid var(--sf-rule); }}
.sf-cheat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(228px,1fr));
  gap:13px 22px; }}
.sf-cheat-grid > div {{ line-height:1.5; }}
.sf-cheat-grid code {{
  font-family:'JetBrains Mono',monospace; font-size:.745rem; background:#f1f5f5;
  border:1px solid #e4ebeb; color:{DEEP}; padding:1px 6px; border-radius:5px;
  margin:0 3px 3px 0; display:inline-block;
}}
.sf-cheat-grid span {{ display:block; font-size:.78rem; color:var(--sf-muted);
  line-height:1.55; margin-top:3px; }}
.sf-cheat-grid span code {{ background:#eef3f3; }}
.sf-cheat-warn {{
  margin:15px 0 0; font-size:.78rem; color:{ROSE}; line-height:1.55;
  background:#faeff0; border:1px solid #eed3d5; border-radius:9px; padding:9px 12px;
}}
.sf-hint {{ font-size:.775rem; color:var(--sf-muted); line-height:1.55; margin:10px 0 0; }}
.sf-status {{ margin-top:14px !important; }}
.gradio-container .sf-status .prose > p {{
  font-size:.85rem !important; color:var(--sf-ink) !important; line-height:1.55 !important;
  background:{MINT}; border:1px solid #bcd9d5; border-left:4px solid {TEAL};
  border-radius:10px; padding:11px 14px; margin:0 !important;
}}

/* ---------------------------------------------------------------- waiting state */
.sf-idle {{
  background:var(--sf-card); border:1px solid var(--sf-rule); border-radius:16px;
  padding:26px 28px; box-shadow:0 1px 2px rgba(16,40,44,.05);
}}
.sf-idle h3 {{
  font-family:'Fraunces', Georgia, serif; font-size:1.12rem; font-weight:600;
  color:var(--sf-ink); margin:0 0 4px;
}}
.sf-idle > p {{ font-size:.87rem; color:var(--sf-muted); margin:0 0 20px; max-width:74ch;
  line-height:1.6; }}
.sf-idle-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; }}
.sf-idle-cell {{
  border:1px solid var(--sf-rule); border-radius:12px; padding:14px 16px; background:#fbfdfd;
}}
.sf-idle-cell.ok   {{ border-left:4px solid {TEAL}; }}
.sf-idle-cell.warn {{ border-left:4px solid {SAND}; }}
.sf-idle-cell.stop {{ border-left:4px solid {ROSE}; }}
.sf-idle-cell b {{ display:block; font-size:.82rem; color:var(--sf-ink); margin-bottom:3px; }}
.sf-idle-cell span {{ font-size:.795rem; color:var(--sf-muted); line-height:1.55; }}

/* ---------------------------------------------------------------- instructions */
.sf-help h4 {{ margin:18px 0 6px; font-size:.95rem; color:var(--sf-ink); font-weight:600;
  font-family:'Fraunces', Georgia, serif; }}
.sf-help p, .sf-help li {{ font-size:.87rem; color:var(--sf-muted); line-height:1.65; }}
.sf-help code {{ font-family:'JetBrains Mono',monospace; font-size:.79rem;
  background:#eaf1f1; padding:1px 6px; border-radius:4px; color:#3d5257; }}
.sf-help pre {{ background:#f4f8f8; border:1px solid var(--sf-rule); border-radius:9px;
  padding:12px 14px; overflow-x:auto; font-family:'JetBrains Mono',monospace;
  font-size:.78rem; color:#3d5257; line-height:1.55; }}
.sf-help table {{ border-collapse:collapse; font-size:.845rem; margin-top:6px; }}
.sf-help th, .sf-help td {{ text-align:left; padding:7px 14px 7px 0;
  border-bottom:1px solid #eaf0f0; vertical-align:top; }}
.sf-help th {{ color:var(--sf-ink); font-weight:600; }}
.sf-help td:first-child {{ font-family:'JetBrains Mono',monospace; font-size:.79rem;
  color:{TEAL}; white-space:nowrap; }}

@media (max-width: 880px) {{
  #sf-hero {{ padding:28px 24px 26px; }}
  #sf-hero h1 {{ font-size:2.1rem; }}
}}
"""

# The palette is meaning-bearing -- teal says trust this, rose says do not -- and
# those readings only hold on a light ground, so the page commits to one theme
# rather than inverting into a dark one that would recolour every verdict.
FORCE_LIGHT = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.href);
  }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill=PAPER,
    body_background_fill_dark=PAPER,
    block_background_fill=CARD,
    block_border_color=RULE,
    block_radius="14px",
    block_shadow="none",
    button_primary_background_fill=DEEP,
    button_primary_background_fill_hover=TEAL,
    button_primary_text_color="#ffffff",
    button_large_radius="11px",
    button_small_radius="9px",
    input_border_color=RULE,
    input_background_fill="#f7fafa",
    color_accent=TEAL,
    color_accent_soft=MINT,
)
