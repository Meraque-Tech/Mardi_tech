"""Tests for collapsible web UI panel markup."""

from html.parser import HTMLParser
from pathlib import Path
import unittest


INDEX_HTML = Path(__file__).parent / "static" / "index.html"
EXPECTED_PANELS = {"dataset", "training", "gpu", "advanced", "logs", "results", "testing"}


class CollapsibleMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.panels = set()
        self.toggles = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        panel_key = attributes.get("data-panel-key")
        if panel_key:
            self.panels.add(panel_key)
        toggle_key = attributes.get("data-panel-toggle")
        if toggle_key:
            self.toggles[toggle_key] = attributes


class CollapsibleSectionTests(unittest.TestCase):
    def test_each_requested_panel_has_an_accessible_toggle(self):
        parser = CollapsibleMarkupParser()
        parser.feed(INDEX_HTML.read_text(encoding="utf-8"))

        self.assertEqual(parser.panels, EXPECTED_PANELS)
        self.assertEqual(set(parser.toggles), EXPECTED_PANELS)
        for attributes in parser.toggles.values():
            self.assertIn(attributes.get("aria-expanded"), {"true", "false"})
            self.assertTrue(attributes.get("aria-label"))


if __name__ == "__main__":
    unittest.main()
