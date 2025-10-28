"""Service for navigating through segments and switching views."""

from __future__ import annotations

from ..project import ProjectStore, Segment


class SegmentNavigator:
    """Handles segment navigation and view switching logic."""

    def __init__(
        self,
        orders: dict[str, list[str]],
        segments_by_id: dict[str, Segment],
        parents: dict[str, str],
        edits: dict[str, str],
        state: dict[str, str],
        store: ProjectStore,
    ):
        self.orders: dict[str, list[str]] = orders
        self.segments_by_id: dict[str, Segment] = segments_by_id
        self.parents: dict[str, str] = parents
        self.edits: dict[str, str] = edits
        self.state: dict[str, str] = state
        self.store: ProjectStore = store

    def navigate(self, view: str, current_id: str, action: str) -> str:
        """
        Navigate to next/prev segment or next issue.

        Args:
            view: Current view mode ("line" or "word")
            current_id: Current segment ID
            action: Navigation action ("prev", "next", or "next_issue")

        Returns:
            ID of the target segment
        """
        order = self.orders.get(view, [])
        if not order:
            return current_id
        try:
            index = order.index(current_id)
        except ValueError:
            index = 0
        if action == "prev":
            index = max(index - 1, 0)
        elif action == "next":
            index = min(index + 1, len(order) - 1)
        elif action == "next_issue":
            next_issue_id = self._next_issue(view, index)
            return next_issue_id or current_id
        return order[index]

    def _next_issue(self, view: str, start_index: int) -> str | None:
        """
        Find the next segment with an unresolved conflict.

        Args:
            view: View mode to search in
            start_index: Index to start searching from

        Returns:
            Segment ID with next conflict, or None if none found
        """
        order = self.orders.get(view, [])
        length = len(order)
        if not length:
            return None
        begin = max(start_index + 1, 0)
        for idx in range(begin, length):
            seg_id = order[idx]
            segment = self.segments_by_id[seg_id]
            if segment.has_conflict and seg_id not in self.edits:
                return seg_id
        return None

    def switch_view(self, current_segment: str, target_view: str) -> str:
        """
        Switch between line and word views, maintaining position when possible.

        Args:
            current_segment: Current segment ID
            target_view: Target view mode ("line" or "word")

        Returns:
            ID of segment to display in target view
        """
        segment = self.segments_by_id.get(current_segment)
        if target_view == "line":
            if segment and segment.view == "line":
                target = current_segment
            elif segment and segment.view == "word":
                target = self.parents.get(current_segment, "")
            else:
                target = ""
            if not target and self.orders["line"]:
                target = self.orders["line"][0]
        else:
            if segment and segment.view == "word":
                target = current_segment
            elif segment and segment.view == "line" and segment.word_ids:
                target = segment.word_ids[0]
            else:
                target = ""
            if not target and self.orders["word"]:
                target = self.orders["word"][0]
        if not target:
            target = current_segment
        self.persist_state(target_view, target)
        return target

    def persist_state(self, view: str, segment_id: str) -> None:
        """
        Save current view and segment to persistent state.

        Args:
            view: Current view mode
            segment_id: Current segment ID
        """
        # Mutate state dict in place so shared references stay in sync
        self.state.clear()
        self.state["view"] = view
        self.state["segment_id"] = segment_id
        self.store.write_state(self.state)

    def navigation_status(self, segment_id: str, view: str) -> dict[str, bool]:
        """
        Get navigation capabilities for current segment.

        Args:
            segment_id: Current segment ID
            view: Current view mode

        Returns:
            Dictionary with can_prev, can_next, has_next_issue flags
        """
        if view not in {"line", "word"}:
            return {"can_prev": False, "can_next": False, "has_next_issue": False}
        order = self.orders.get(view, [])
        if not order:
            return {"can_prev": False, "can_next": False, "has_next_issue": False}
        try:
            index = order.index(segment_id)
        except ValueError:
            return {"can_prev": False, "can_next": False, "has_next_issue": False}
        can_prev = index > 0
        can_next = index < len(order) - 1
        has_next_issue = self._next_issue(view, index) is not None
        return {"can_prev": can_prev, "can_next": can_next, "has_next_issue": has_next_issue}
