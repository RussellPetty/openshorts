"""Sentence-aware boundary snapping, the false-ending rule, the quiet-point
search windows and the word-timed transcript the detail pass reads."""
import pytest

from clip_selection import (
    ends_complete,
    is_false_ending_word,
    is_terminal,
    normalize_words,
    quiet_search_window,
    sentence_end_at,
    sentence_start_at,
    snap_clip_bounds,
    timed_transcript,
)


def _sentences(*sentences, start=0.0, word_len=0.3, gap=0.1, sentence_gap=0.5):
    """Word list from sentences; each word ``word_len`` long, ``gap`` apart,
    ``sentence_gap`` between sentences. Returns normalized [{'w','s','e'}]."""
    words, t = [], start
    for sentence in sentences:
        toks = sentence.split()
        for k, tok in enumerate(toks):
            words.append({"word": " " + tok, "start": round(t, 3), "end": round(t + word_len, 3)})
            t += word_len + (sentence_gap if k == len(toks) - 1 else gap)
    return normalize_words(words)


def _filler_sentence(n_words, end_word="done."):
    return " ".join(["word"] * (n_words - 1) + [end_word])


class TestWordRules:
    def test_terminal_punctuation(self):
        assert is_terminal(" done.")
        assert is_terminal(' "really?"')
        assert is_terminal("wow!")
        assert not is_terminal(" and")
        assert not is_terminal(" well,")

    @pytest.mark.parametrize("w", [" and", " but", " the", " a", " to", " of", " because",
                                   " so", " that", " with", " And"])
    def test_false_ending_words(self, w):
        assert is_false_ending_word(w)

    def test_punctuated_false_ending_word_is_trusted(self):
        assert not is_false_ending_word(" so.")

    def test_content_words_are_not_false_endings(self):
        assert not is_false_ending_word(" money")
        assert not is_false_ending_word(" at")  # "what we're good at"

    def test_sentence_end_by_pause_and_capital(self):
        ws = normalize_words([
            {"w": " we", "s": 0.0, "e": 0.2}, {"w": " won", "s": 0.3, "e": 0.6},
            {"w": " Next", "s": 0.9, "e": 1.1}, {"w": " we", "s": 1.15, "e": 1.3},
            {"w": " ate", "s": 1.35, "e": 1.5}, {"w": " cake", "s": 2.6, "e": 2.9},
        ])
        assert sentence_end_at(ws, 1)        # 0.3s breath + capitalized "Next"
        assert not sentence_end_at(ws, 2)    # mid-sentence
        assert sentence_end_at(ws, 4)        # 1.1s pause
        assert sentence_end_at(ws, 5)        # end of source
        assert sentence_start_at(ws, 0) and sentence_start_at(ws, 2)

    def test_capital_I_is_not_a_new_sentence(self):
        ws = normalize_words([{"w": " then", "s": 0, "e": 0.2}, {"w": " I", "s": 0.5, "e": 0.6},
                              {"w": " left", "s": 0.7, "e": 0.9}])
        assert not sentence_end_at(ws, 0)


class TestEndsComplete:
    def test_terminal_ok(self):
        ws = _sentences("this is done.", "Then more")
        assert ends_complete(ws, 2)[0]

    def test_hard_midclause_word_fails(self):
        ws = _sentences("we went there and", "then left.")
        ok, reason = ends_complete(ws, 3)
        assert not ok and "false ending" in reason

    def test_same_speaker_continues_fails(self):
        ws = normalize_words([{"w": " we", "s": 0, "e": 0.2}, {"w": " bought", "s": 0.3, "e": 0.5},
                              {"w": " houses", "s": 0.6, "e": 0.9}, {"w": " in", "s": 1.0, "e": 1.1},
                              {"w": " Texas.", "s": 1.2, "e": 1.5}])
        ok, reason = ends_complete(ws, 2)
        assert not ok and "continues" in reason


class TestSnapClipBounds:
    def _talk(self):
        # 12 sentences of 10 words: each sentence = 10*0.3 + 9*0.1 = 3.9s, +0.5 gap = 4.4s
        return _sentences(*[_filler_sentence(10, f"end{i}.") for i in range(12)])

    def test_start_moves_to_sentence_start_and_end_to_sentence_end(self):
        ws = self._talk()
        # proposal starts mid-sentence 2 (8.8 + ~1.2) and ends mid-sentence 6
        r = snap_clip_bounds(10.0, 28.0, ws, 60.0, min_duration=15, max_duration=60)
        assert r is not None
        starts = {w["s"] for w in ws}
        assert r["word_start"] in starts
        first = next(i for i, w in enumerate(ws) if w["s"] == r["word_start"])
        assert sentence_start_at(ws, first)
        last = next(i for i, w in enumerate(ws) if w["e"] == r["word_end"])
        assert is_terminal(ws[last]["w"])
        assert r["start"] <= r["word_start"] and r["end"] >= r["word_end"]
        assert 15 - 0.5 <= r["word_end"] - r["word_start"] <= 60
        assert r["start_kind"] == "sentence" and r["end_kind"].startswith("sentence")

    def test_never_cuts_mid_word(self):
        ws = self._talk()
        r = snap_clip_bounds(10.05, 30.05, ws, 60.0, min_duration=15, max_duration=60)
        for w in ws:
            assert not (w["s"] < r["start"] < w["e"])
            assert not (w["s"] < r["end"] < w["e"])

    def test_extends_to_sentence_end_instead_of_false_ending(self):
        # One long sentence whose words all continue: only the final "end." closes it.
        ws = _sentences("we " * 50 + "end.", "Then more words here.")
        r = snap_clip_bounds(0.0, 12.0, ws, 60.0, min_duration=5, max_duration=30,
                             end_back=1.0, end_fwd=1.0)
        assert r is not None
        assert r["last_word"] == "end."
        assert r["end_kind"] == "sentence-extended"

    def test_drops_when_no_sentence_end_fits(self):
        # A 40s run-on sentence with no pause or punctuation, max 30s: nothing fits.
        ws = _sentences("and " * 100 + "end.")
        assert snap_clip_bounds(0.0, 20.0, ws, 60.0, min_duration=10, max_duration=30) is None

    def test_lenient_keeps_a_word_boundary_and_avoids_false_endings(self):
        ws = _sentences("we went and " * 40 + "end.")
        r = snap_clip_bounds(0.0, 20.0, ws, 60.0, min_duration=10, max_duration=30, lenient=True)
        assert r is not None
        assert not is_false_ending_word(r["last_word"])
        assert r["word_end"] - r["word_start"] <= 30

    def test_bounds_validated_in_range(self):
        ws = self._talk()
        total = ws[-1]["e"]
        r = snap_clip_bounds(total - 20, total + 5, ws, total, min_duration=10, max_duration=60)
        assert r is not None
        assert 0 <= r["start"] < r["end"] <= total

    def test_steps_over_leading_filler(self):
        ws = _sentences("um so this is the claim that matters a lot today.",
                        *[_filler_sentence(10, f"e{i}.") for i in range(6)])
        r = snap_clip_bounds(0.0, 20.0, ws, 60.0, min_duration=10, max_duration=60)
        assert r["first_word"] == "so"

    def test_no_words_returns_clamped_original(self):
        r = snap_clip_bounds(5.0, 25.0, [], 20.0)
        assert (r["start"], r["end"]) == (5.0, 20.0)

    def test_garbage_input_is_dropped(self):
        assert snap_clip_bounds("x", 10, self._talk(), 60.0) is None


class TestQuietSearchWindow:
    def test_in_point_window(self):
        assert quiet_search_window(10.0, "in") == (9.9, 10.05)

    def test_out_point_window(self):
        assert quiet_search_window(10.0, "out") == (9.94, 10.2)

    def test_never_crosses_neighbouring_words(self):
        assert quiet_search_window(10.0, "in", prev_end=9.97) == (9.97, 10.05)
        assert quiet_search_window(10.0, "out", next_start=10.08) == (9.94, 10.08)

    def test_clamped_to_source(self):
        assert quiet_search_window(0.03, "in") == (0.0, 0.08)
        assert quiet_search_window(10.0, "out", video_duration=10.1) == (9.94, 10.1)

    def test_no_room(self):
        assert quiet_search_window(10.0, "out", next_start=9.95) is None

    def test_bad_kind(self):
        with pytest.raises(ValueError):
            quiet_search_window(1.0, "middle")


class TestTimedTranscript:
    def test_one_line_per_sentence_with_times(self):
        ws = _sentences("Hello there friend.", "Second one here.")
        out = timed_transcript(ws, 0, 10).splitlines()
        assert out[0] == "0.0-1.1 Hello there friend."
        assert out[1].startswith("1.6-") and out[1].endswith("Second one here.")

    def test_long_sentence_continues_with_tilde(self):
        ws = _sentences("word " * 20 + "end.")
        lines = timed_transcript(ws, 0, 100, max_words_per_line=8).splitlines()
        assert not lines[0].startswith("~")
        assert all(line.startswith("~") for line in lines[1:])

    def test_window_slice_marks_mid_sentence_start(self):
        ws = _sentences("one two three four five six.")
        lines = timed_transcript(ws, 1.0, 10).splitlines()
        assert lines[0].startswith("~")

    def test_compact(self):
        ws = _sentences(*["this is a normal sentence of words."] * 30)
        text = timed_transcript(ws, 0, 1000)
        plain = " ".join(w["w"].strip() for w in ws)
        # timing overhead stays well under the text itself
        assert len(text) < len(plain) * 1.4
