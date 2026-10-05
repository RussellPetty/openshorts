"""Caption presets: the API, the MCP tool and the dashboard agree on them."""
import asyncio
import re

from subtitles import AUTO_CAPTION_STYLE, CAPTION_PRESETS, line_budget


def test_default_preset_is_what_clips_ship_with():
    d = CAPTION_PRESETS["default"]
    for k in ("font_name", "font_size", "highlight_color", "border_width",
              "effect", "uppercase", "max_chars", "max_duration"):
        assert d[k] == AUTO_CAPTION_STYLE[k], k


def test_line_budget_shrinks_with_size_and_wide_fonts():
    assert line_budget("Anton", 44) == 16
    assert line_budget("Anton", 70) < line_budget("Anton", 44) < line_budget("Anton", 34)
    assert line_budget("Montserrat ExtraBold", 44) < line_budget("Anton", 44)


class _FakeResp:
    status_code = 200

    def json(self):
        return {"ok": True}


class _FakeClient:
    def __init__(self):
        self.body = None

    async def post(self, path, json):
        assert path == "/api/subtitle"
        self.body = json
        return _FakeResp()


def _call(**args):
    from mcp_server import _tool_add_subtitles
    client = _FakeClient()
    asyncio.run(_tool_add_subtitles(client, {"job_id": "j", "clip_index": 0, **args}))
    return client.body


