from pathlib import Path
from html.parser import HTMLParser

import pytest

from contact.presentation import parse_table, table_html, render_document

ROOT = Path(__file__).resolve().parents[1]


def test_supplied_table_roundtrip_and_significance():
    panels = parse_table((ROOT / 'docs/results.tex').read_text())
    assert [len(p['rows']) for p in panels] == [16, 17]
    assert all(len(r['cells']) == (20 if p['kind'] == 'hh' else 18) for p in panels for r in p['rows'])
    for p in panels:
        own = [r for r in p['rows'] if r['name'] == 'Ours']
        assert len(own) == 2
        if p['kind'] == 'hai':
            assert own[0]['cells'][15]['value'] == '.41'
            assert own[0]['cells'][15]['marker'] == 'dagger'
            assert own[0]['cells'][17]['marker'] == 'dagger'
            assert [c['value'] for c in own[0]['cells']] == [c['value'] for c in own[1]['cells']]
            assert [c['marker'] for c in own[0]['cells']] == [c['marker'] for c in own[1]['cells']]
    html = render_document(panels)
    assert html == (ROOT / 'docs/results.html').read_text()
    assert 'instructed' in html and 'NaN' not in html
    assert html.count('<td>') == 626
    assert html.count('<table ') == html.count('</table>') == 2
    for p in panels:
        assert table_html(p) in (ROOT / 'README.md').read_text()


def test_visible_cells_match_source():
    class Cells(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current = None
            self.values = []
        def handle_starttag(self, tag, attrs):
            if tag == 'td':
                self.current = ''
        def handle_data(self, data):
            if self.current is not None:
                self.current += data
        def handle_endtag(self, tag):
            if tag == 'td':
                self.values.append(self.current)
                self.current = None
    panels = parse_table((ROOT / 'docs/results.tex').read_text())
    parser = Cells()
    parser.feed(render_document(panels))
    expected = []
    for p in panels:
        for row in p['rows']:
            for cell in row['cells']:
                value = '—' if cell['value'] == '--' else cell['value']
                expected.append(value + {'dagger': '†', 'ddagger': '‡', None: ''}[cell['marker']])
    assert parser.values == expected


def test_reject_unexpected_markup():
    text = (ROOT / 'docs/results.tex').read_text()
    with pytest.raises(ValueError, match='Unsupported cell'):
        parse_table(text.replace('& 44.0 &', '& <script> &', 1))
