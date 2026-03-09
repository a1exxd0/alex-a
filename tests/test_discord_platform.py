from assistant.platforms.discord_platform import _chunk


class TestChunk:
    def test_short_text_returned_as_single_chunk(self):
        text = "Hello world"
        assert _chunk(text) == [text]

    def test_text_exactly_at_limit_is_not_split(self):
        text = "a" * 2000
        assert _chunk(text) == [text]

    def test_text_one_over_limit_is_split(self):
        text = "a" * 2001
        chunks = _chunk(text)
        assert len(chunks) == 2

    def test_splits_at_newline_when_possible(self):
        text = "a" * 1999 + "\n" + "b" * 100
        chunks = _chunk(text)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 1999
        assert chunks[1] == "b" * 100

    def test_splits_at_size_boundary_when_no_newline(self):
        text = "a" * 3000  # no newlines
        chunks = _chunk(text)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2000
        assert len(chunks[1]) == 1000

    def test_empty_string_returns_single_empty_chunk(self):
        assert _chunk("") == [""]

    def test_custom_size(self):
        chunks = _chunk("abcdef", size=3)
        assert chunks == ["abc", "def"]

    def test_custom_size_with_newline_preference(self):
        # "ab\ncd" with size=4: first 4 chars = "ab\nc", newline at index 2
        # splits before the newline
        chunks = _chunk("ab\ncd", size=4)
        assert chunks == ["ab", "cd"]

    def test_total_length_preserved(self):
        text = "x" * 5000
        chunks = _chunk(text)
        assert sum(len(c) for c in chunks) == 5000

    def test_no_chunk_exceeds_limit(self):
        text = "word " * 1000  # 5000 chars with spaces (potential split points)
        chunks = _chunk(text)
        for chunk in chunks:
            assert len(chunk) <= 2000

    def test_multiple_chunks_for_very_long_text(self):
        text = "a" * 6500
        chunks = _chunk(text)
        assert len(chunks) == 4  # 2000 + 2000 + 2000 + 500

    def test_trailing_newline_stripped_from_next_chunk(self):
        # After splitting, the leading newline on the remainder is lstripped
        text = "a" * 1999 + "\n" + "b" * 50
        chunks = _chunk(text)
        assert not chunks[1].startswith("\n")
