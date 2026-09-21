"""
Pick/Place event state machine.

State flow:

PLACED
   ↓
PICKING
   ↓
PICKED
   ↓
PERSON_LEAVES
   ↓
PLACING
   ↓
PLACED

The machine receives cleaned per-frame signals from pipeline.py.
Transitions are debounced using consecutive supporting frames.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(Enum):
    PLACED = "PLACED"
    PICKING = "PICKING"
    PICKED = "PICKED"
    PERSON_LEAVES = "PERSON_LEAVES"
    PLACING = "PLACING"


@dataclass
class Signals:
    wrist_in_roi: bool
    backpack_in_roi: bool
    person_visible: bool
    frame_idx: int


@dataclass
class Event:
    frame_idx: int
    from_state: State
    to_state: State


class PickPlaceStateMachine:

    def __init__(self, confirm_frames: int = 5, leave_frames: int = 15):

        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")

        if leave_frames < 1:
            raise ValueError("leave_frames must be at least 1")

        self.confirm_frames = confirm_frames
        self.leave_frames = leave_frames

        # Assignment starts with the backpack placed.
        self.state = State.PLACED

        # Used for debouncing state transitions.
        self._pending_state: Optional[State] = None
        self._pending_count = 0

        # Used to confirm that the person has actually left.
        self._person_absent_count = 0

        # Stores confirmed state transitions.
        self.events: list[Event] = []

    def _try_commit(
        self,
        candidate: State,
        frame_idx: int,
        min_frames: int
    ):
        """
        Commit a candidate state only after it has been
        continuously supported for min_frames.
        """

        # Candidate is already the current state.
        if candidate == self.state:
            self._pending_state = None
            self._pending_count = 0
            return

        # Same candidate as previous frame.
        if candidate == self._pending_state:
            self._pending_count += 1

        # New candidate transition.
        else:
            self._pending_state = candidate
            self._pending_count = 1

        # Candidate has been stable long enough.
        if self._pending_count >= min_frames:
            self._commit(candidate, frame_idx)

    def _commit(self, new_state: State, frame_idx: int):
        """
        Confirm and record a state transition.
        """

        self.events.append(
            Event(
                frame_idx,
                self.state,
                new_state
            )
        )

        self.state = new_state

        # Reset transition debounce information.
        self._pending_state = None
        self._pending_count = 0

        # Reset absence counter after any normal transition.
        self._person_absent_count = 0

    def step(self, s: Signals) -> State:
        """
        Process one frame of signals and return the current state.
        """

        # ---------------------------------------------------------
        # PLACED -> PICKING
        # ---------------------------------------------------------
        if self.state == State.PLACED:

            # Person's wrist interacts with the backpack
            # while the backpack is still inside the ROI.
            if s.wrist_in_roi and s.backpack_in_roi:

                self._try_commit(
                    State.PICKING,
                    s.frame_idx,
                    self.confirm_frames
                )

            else:
                self._pending_state = None
                self._pending_count = 0

        # ---------------------------------------------------------
        # PICKING -> PICKED
        # ---------------------------------------------------------
        elif self.state == State.PICKING:

            # Wrist and backpack have both moved outside ROI.
            if not s.wrist_in_roi and not s.backpack_in_roi:

                self._try_commit(
                    State.PICKED,
                    s.frame_idx,
                    self.confirm_frames
                )

            # False interaction:
            # wrist leaves but backpack stays in place.
            elif not s.wrist_in_roi and s.backpack_in_roi:

                self._try_commit(
                    State.PLACED,
                    s.frame_idx,
                    self.confirm_frames
                )

            else:
                self._pending_state = None
                self._pending_count = 0

        # ---------------------------------------------------------
        # PICKED -> PERSON_LEAVES
        # ---------------------------------------------------------
        elif self.state == State.PICKED:

            if not s.person_visible:

                self._person_absent_count += 1

                if self._person_absent_count >= self.leave_frames:

                    self._commit(
                        State.PERSON_LEAVES,
                        s.frame_idx
                    )

            else:
                # Person is visible again.
                self._person_absent_count = 0

        # ---------------------------------------------------------
        # PERSON_LEAVES -> PLACING
        # ---------------------------------------------------------
        elif self.state == State.PERSON_LEAVES:

            # Person returns and interacts with the ROI
            # while the backpack is inside the ROI.
            if (
                s.person_visible
                and s.wrist_in_roi
                and s.backpack_in_roi
            ):

                self._try_commit(
                    State.PLACING,
                    s.frame_idx,
                    self.confirm_frames
                )

            else:
                self._pending_state = None
                self._pending_count = 0

        # ---------------------------------------------------------
        # PLACING -> PLACED
        # ---------------------------------------------------------
        elif self.state == State.PLACING:

            # Wrist leaves but backpack remains inside ROI.
            if not s.wrist_in_roi and s.backpack_in_roi:

                self._try_commit(
                    State.PLACED,
                    s.frame_idx,
                    self.confirm_frames
                )

            else:
                self._pending_state = None
                self._pending_count = 0

        return self.state