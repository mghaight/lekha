"""Service for editing and saving segment text."""

from __future__ import annotations

from ..project import ProjectStore, Segment


class SegmentEditor:
    """Handles text editing, saving, and synchronization of segments."""

    def __init__(
        self,
        orders: dict[str, list[str]],
        segments_by_id: dict[str, Segment],
        parents: dict[str, str],
        edits: dict[str, str],
        store: ProjectStore,
    ):
        self.orders: dict[str, list[str]] = orders
        self.segments_by_id: dict[str, Segment] = segments_by_id
        self.parents: dict[str, str] = parents
        self.edits: dict[str, str] = edits
        self.store: ProjectStore = store

    def save(self, segment_id: str, view: str, text: str) -> None:
        """
        Save edited text for a segment.

        Args:
            segment_id: ID of segment to save
            view: Current view mode ("line" or "word")
            text: New text content
        """
        if view == "line":
            self._save_line(segment_id, text)
        else:
            self._save_word(segment_id, text)
        self._persist()

    def _save_line(self, line_id: str, text: str) -> None:
        """
        Save line text and update child words if they exist.

        Args:
            line_id: ID of line segment
            text: New line text
        """
        line_segment = self.get_segment(line_id)
        tokens = text.split()
        word_ids = line_segment.word_ids
        if word_ids:
            if not tokens:
                tokens = ["" for _ in word_ids]
            if len(tokens) < len(word_ids):
                tokens.extend([""] * (len(word_ids) - len(tokens)))
            elif len(tokens) > len(word_ids):
                leading = tokens[: len(word_ids) - 1]
                trailing = " ".join(tokens[len(word_ids) - 1 :])
                tokens = leading + [trailing]
            for word_id, token in zip(word_ids, tokens):
                self.edits[word_id] = token
        self.edits[line_id] = text

    def _save_word(self, word_id: str, text: str) -> None:
        """
        Save word text and update parent line.

        Args:
            word_id: ID of word segment
            text: New word text
        """
        word_segment = self.get_segment(word_id)
        self.edits[word_id] = text
        parent_line_id = self.parents.get(word_id)
        if parent_line_id:
            parent_segment = self.get_segment(parent_line_id)
            tokens: list[str] = []
            for child_id in parent_segment.word_ids:
                if child_id == word_id:
                    tokens.append(text)
                else:
                    tokens.append(self.edits.get(child_id, self.get_segment(child_id).consensus_text))
            line_text = " ".join(tokens).strip()
            self.edits[parent_line_id] = line_text
        else:
            # ensure master text still consistent
            self._recalculate_line_from_word(word_segment.line_index, word_segment.page_index)

    def _recalculate_line_from_word(self, line_index: int, page_index: int) -> None:
        """
        Recalculate line text from constituent words.

        Args:
            line_index: Line index within page
            page_index: Page index
        """
        line_id = f"p{page_index:03d}_l{line_index:04d}"
        if line_id not in self.segments_by_id:
            return
        line_segment = self.segments_by_id[line_id]
        tokens = [
            self.edits.get(word_id, self.segments_by_id[word_id].consensus_text) for word_id in line_segment.word_ids
        ]
        self.edits[line_id] = " ".join(tokens).strip()

    def get_text(self, segment_id: str) -> str:
        """
        Get current text for a segment (edited or consensus).

        Args:
            segment_id: ID of segment

        Returns:
            Current text content
        """
        if segment_id in self.edits:
            return self.edits[segment_id]
        segment = self.get_segment(segment_id)
        if segment.view == "line" and segment.word_ids:
            tokens = [self.get_text(word_id) for word_id in segment.word_ids]
            text = " ".join(tokens).strip()
            return text if text else segment.consensus_text
        return segment.consensus_text

    def get_segment(self, segment_id: str) -> Segment:
        """
        Get segment by ID.

        Args:
            segment_id: ID of segment

        Returns:
            Segment object

        Raises:
            KeyError: If segment ID not found
        """
        return self.segments_by_id[segment_id]

    def compose_master_text(self) -> str:
        """
        Compose full master text from all line segments.

        Returns:
            Complete manuscript text
        """
        lines: list[str] = []
        for line_id in self.orders["line"]:
            lines.append(self.get_text(line_id))
        return "\n".join(lines)

    def _persist(self) -> None:
        """Write edits and master text to storage."""
        self.store.write_edits(self.edits)
        master_text = self.compose_master_text()
        self.store.write_master(master_text)
