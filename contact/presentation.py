"""Render the supplied results table for the repository and a static web page."""
import argparse
import html
from pathlib import Path
import re

CELL = re.compile(r"(?P<bold>\\textbf\{)?(?P<value>--|-?(?:\d+\.\d+|\.\d+))"
                  r"(?(bold)\})(?:\$\^\{\\(?P<marker>dagger|ddagger)\}\$)?")
METHODS = {"UTMOSv2", "VAP", "DualTurn", "Talking Turns", "UniSRM", "TRACE",
           "TimingStats", "Ours", "w/o A--V", "w/o future joint silence",
           "w/o future voice activity", "w/o timing outputs"}


def parse_table(text):
    panels, panel, group, name = [], None, None, None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("%"):
            continue
        if line.startswith(r"\begin{tabular*}"):
            panel = {"kind": "hh" if not panels else "hai", "rows": []}
            panels.append(panel)
            group, name = None, None
        if panel is None:
            continue
        if line.startswith(r"\multicolumn") and r"\strut\textit{\textbf{" in line:
            group = line.split(r"\strut\textit{\textbf{", 1)[1].rsplit("}}}", 1)[0]
        if line in METHODS or line.startswith(r"Inter-rater $\rho$"):
            name = line
        elif name and line.startswith("& "):
            if not line.endswith(r"\\"):
                raise ValueError("Expected one complete numeric table row")
            tokens = line[2:-2].strip().split(" & ")
            expected = 20 if panel["kind"] == "hh" else 18
            if len(tokens) != expected or group is None:
                raise ValueError(f"Unexpected row shape: {name}")
            cells = []
            for token in tokens:
                match = CELL.fullmatch(token)
                if match is None:
                    raise ValueError(f"Unsupported cell: {token}")
                cells.append(dict(value=match["value"], bold=bool(match["bold"]),
                                  marker=match["marker"]))
            panel["rows"].append(dict(name=name, group=group, cells=cells))
            name = None
    if len(panels) != 2 or any(not p["rows"] for p in panels):
        raise ValueError("Expected both H–H and H–AI panels")
    return panels


def headers(kind):
    if kind == "hh":
        rows = [
            '<tr><th rowspan="4" scope="col">Scorer</th><th colspan="8">Discrimination</th><th colspan="12">MOS correlation</th></tr>',
            '<tr><th colspan="4">Turn-taking</th><th colspan="4">Affective</th><th colspan="6">Turn-taking</th><th colspan="6">Affective</th></tr>',
            '<tr>' + ''.join(f'<th colspan="{n}">{s}</th>' for n in (2, 2, 3, 3) for s in ("Dev", "Test")) + '</tr>',
            '<tr>' + ''.join(f'<th scope="col">{s}</th>' for s in ["Acc.", "C-I"] * 4 + ["P", "S", "C"] * 4) + '</tr>',
        ]
    else:
        rows = [
            '<tr><th rowspan="3" scope="col">Scorer</th>' + ''.join(f'<th colspan="6">{s}</th>' for s in ("Turn-taking", "Affective", "Overall")) + '</tr>',
            '<tr>' + ''.join(f'<th colspan="3">{s}</th>' for s in ["Dev", "Test"] * 3) + '</tr>',
            '<tr>' + ''.join(f'<th scope="col">{s}</th>' for s in ["P", "S", "C"] * 6) + '</tr>',
        ]
    return '\n'.join(rows)


def table_html(panel):
    title = "H–H evaluation" if panel["kind"] == "hh" else "H–AI evaluation"
    lines = [f'<table aria-label="{title}">', f'<caption>{title}</caption>',
             '<thead>', headers(panel["kind"]), '</thead>', '<tbody>']
    group = None
    for row in panel["rows"]:
        if row["group"] != group:
            group = row["group"]
            lines.append(f'<tr class="group"><th colspan="{len(row["cells"]) + 1}">{html.escape(group)}</th></tr>')
        name = row["name"].replace("A--V", "A–V").replace(r"$\rho$", "ρ").replace("P--S", "P–S")
        cells = []
        for cell in row["cells"]:
            value = "—" if cell["value"] == "--" else html.escape(cell["value"])
            if cell["bold"]:
                value = f'<strong>{value}</strong>'
            if cell["marker"]:
                marker = "†" if cell["marker"] == "dagger" else "‡"
                value += f'<sup>{marker}</sup>'
            cells.append('<td>' + value + '</td>')
        lines.append(f'<tr><th scope="row">{html.escape(name)}</th>' + ''.join(cells) + '</tr>')
    return '\n'.join([*lines, '</tbody>', '</table>'])


LEGEND = """<p>Acc.: paired accuracy (%); C-I: C-index; P/S/C: Spearman correlation with Participant,
Supervisor and Combined ratings. <sup>†</sup> marks Ours significantly above the highest-scoring
baseline at that endpoint. <sup>‡</sup> marks an ablation significantly below Ours in the targeted
component comparisons. Tests use 20,000 paired session-bootstrap resamples and unadjusted two-sided p &lt; .05.</p>
<p>Main H–H rows use the two-participant mean for P. In the uninstructed-participant block,
P is the uninstructed participant's rating for unilateral manipulations and the two-participant mean
for natural conditions. Competitive Floor Conflict is excluded from H–H MOS in this block
(10 dev / 20 test recordings): both participants receive instructions, so neither is uninstructed.
The main results retain Competitive Floor Conflict. The same eligible recordings are used for
P/S/C: turn-taking 50 dev / 100 test; affective 32 dev / 68 test. Discrimination retains all recordings.
Dashes in the sensitivity block's discrimination columns indicate unchanged results.
Unchanged H–AI results are not repeated. Reference rows retain the main rating protocol.</p>
<p>Bold follows the supplied table, with maxima shown separately within the H–H sensitivity block.
Component ablations remove scores from the same checkpoint.</p>"""

CSS = """body{margin:0;background:#f8fafc;color:#162330;font:15px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:32px 24px}h1{font-size:30px;margin:0 0 8px}
a{color:#075ba6}nav{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0 24px}
.table-wrap{overflow:auto;background:white;border:1px solid #dbe3eb;border-radius:8px;margin:20px 0 30px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
caption{text-align:left;font-size:19px;font-weight:700;padding:16px}
td,th{padding:8px 9px;border-bottom:1px solid #e1e7ee;white-space:nowrap;text-align:right}
thead th{background:#eaf1f7;text-align:center}.group th{background:#e7f1ed;text-align:left}
tbody th[scope=row]{text-align:left;position:sticky;left:0;background:#fff;min-width:170px}
tbody tr:hover td,tbody tr:hover th[scope=row]{background:#f4f8fc}sup{font-size:10px;margin-left:1px}
.notes{max-width:1000px;color:#33485a}.downloads{padding:12px 16px;background:#edf3f9;border-radius:6px}
@media(max-width:600px){main{padding:20px 12px}h1{font-size:25px}td,th{padding:7px 8px}}
"""


def render_document(panels):
    tables = '\n'.join('<div class="table-wrap">' + table_html(p) + '</div>' for p in panels)
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>CONTACT — Evaluation results</title><style>' + CSS + '</style></head><body><main>'
            '<h1>CONTACT evaluation results</h1><p>Naturalness evaluation on the development and test sets.</p>'
            '<nav><a href="../README.md">Repository</a><a href="results.tex" download>LaTeX table</a>'
            '<a href="../data/ratings.csv" download>Participant ratings</a>'
            '<a href="../data/scores.csv" download>Model scores</a></nav>'
            + tables + '<section class="notes" aria-label="Reading the results">' + LEGEND + '</section>'
            '<p class="downloads">The two participant ratings and assignment roles are included in ratings.csv. '
            'Questionnaire response order is not an audio-channel or speaker identity.</p>'
            '</main></body></html>\n')


def readme_section(panels):
    table_blocks = []
    for panel in panels:
        title = "H–H results" if panel["kind"] == "hh" else "H–AI results"
        table_blocks.append(f'<details open>\n<summary>{title}</summary>\n\n' + table_html(panel) + '\n\n</details>')
    return ('<!-- CONTACT_RESULTS_START -->\n## Results\n\n'
            'The table below includes the uninstructed-participant analysis. '
            '[LaTeX source](docs/results.tex) · [Standalone HTML](docs/results.html)\n\n'
            + '\n\n'.join(table_blocks) + '\n\n' + LEGEND + '\n<!-- CONTACT_RESULTS_END -->')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", type=Path, default=Path("docs/results.tex"))
    parser.add_argument("--html", type=Path, default=Path("docs/results.html"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()
    panels = parse_table(args.latex.read_text())
    args.html.write_text(render_document(panels))
    readme = args.readme.read_text()
    block = readme_section(panels)
    start, end = '<!-- CONTACT_RESULTS_START -->', '<!-- CONTACT_RESULTS_END -->'
    if start in readme:
        if readme.count(start) != 1 or readme.count(end) != 1:
            raise ValueError("Ambiguous results section")
        a, b = readme.index(start), readme.index(end) + len(end)
        readme = readme[:a] + block + readme[b:]
    else:
        marker = '## Tests\n'
        if readme.count(marker) != 1:
            raise ValueError("Cannot locate the README insertion point")
        readme = readme.replace(marker, block + '\n\n' + marker)
    args.readme.write_text(readme)
    print(f'Rendered {sum(len(p["rows"]) for p in panels)} rows to {args.html} and {args.readme}')


if __name__ == '__main__':
    main()
