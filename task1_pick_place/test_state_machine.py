"""
Sanity-check the pick/place state machine using synthetic signal sequences.

Tests include:

- complete pick -> leave -> return -> place cycle
- brief wrist flicker
- temporary person loss
- false pickup attempt
- noisy pickup transition
- temporary wrist/backpack detection loss

Run:

    python test_state_machine.py
"""

from state_machine import PickPlaceStateMachine, Signals, State


def run(machine, seq):
    """
    seq:
        list of (wrist_in_roi, backpack_in_roi, person_visible)
    """

    for i, (wrist, backpack, person) in enumerate(seq):

        machine.step(
            Signals(
                wrist_in_roi=wrist,
                backpack_in_roi=backpack,
                person_visible=person,
                frame_idx=i
            )
        )

    return machine.state


def test_full_happy_path():

    m = PickPlaceStateMachine(
        confirm_frames=3,
        leave_frames=5
    )

    seq = []

    # Backpack initially placed inside ROI.
    seq += [(False, True, True)] * 5

    # Wrist enters ROI.
    seq += [(True, True, True)] * 3

    # Wrist and backpack leave ROI.
    seq += [(False, False, True)] * 3

    # Person leaves camera.
    seq += [(False, False, False)] * 6

    # Person returns and interacts with backpack.
    seq += [(True, True, True)] * 3

    # Person removes wrist while backpack remains placed.
    seq += [(False, True, True)] * 3

    final = run(m, seq)

    assert final == State.PLACED

    path = [event.to_state for event in m.events]

    expected = [
        State.PICKING,
        State.PICKED,
        State.PERSON_LEAVES,
        State.PLACING,
        State.PLACED
    ]

    assert path == expected, path

    print(
        "test_full_happy_path OK ->",
        [state.value for state in path]
    )


def test_noisy_wrist_does_not_flip_state():

    m = PickPlaceStateMachine(
        confirm_frames=5,
        leave_frames=5
    )

    seq = []

    # Initial placed state.
    seq += [(False, True, True)] * 5

    # Wrist enters ROI for only one frame.
    seq += [(True, True, True)]

    # Wrist immediately leaves.
    seq += [(False, True, True)] * 5

    final = run(m, seq)

    assert final == State.PLACED
    assert len(m.events) == 0

    print(
        "test_noisy_wrist_does_not_flip_state OK"
        " -> no false transition"
    )


def test_temporary_person_loss_does_not_trigger_leave():

    m = PickPlaceStateMachine(
        confirm_frames=3,
        leave_frames=10
    )

    seq = []

    # Initial placed state.
    seq += [(False, True, True)] * 3

    # Pickup interaction.
    seq += [(True, True, True)] * 3

    # Backpack is picked.
    seq += [(False, False, True)] * 3

    # Person temporarily disappears.
    # Only 4 frames, below leave threshold of 10.
    seq += [(False, False, False)] * 4

    # Person comes back.
    seq += [(False, False, True)] * 3

    final = run(m, seq)

    assert final == State.PICKED

    print(
        "test_temporary_person_loss_does_not_trigger_leave OK"
        " -> stayed PICKED"
    )


def test_false_start_reverts_to_placed():

    m = PickPlaceStateMachine(
        confirm_frames=3,
        leave_frames=5
    )

    seq = []

    # Initial placed state.
    seq += [(False, True, True)] * 3

    # Wrist enters ROI -> PICKING.
    seq += [(True, True, True)] * 3

    # Wrist leaves but backpack remains.
    seq += [(False, True, True)] * 3

    final = run(m, seq)

    assert final == State.PLACED

    print(
        "test_false_start_reverts_to_placed OK"
    )


def test_noisy_pickup_does_not_commit():

    m = PickPlaceStateMachine(
        confirm_frames=3,
        leave_frames=5
    )

    seq = []

    # Initial placed.
    seq += [(False, True, True)] * 3

    # Enter PICKING.
    seq += [(True, True, True)] * 3

    # Backpack appears outside for only one frame.
    seq += [(False, False, True)]

    # Detection returns to inside.
    seq += [(True, True, True)] * 3

    final = run(m, seq)

    # The single noisy outside frame must not commit PICKED.
    assert final == State.PICKING

    print(
        "test_noisy_pickup_does_not_commit OK"
        " -> stayed PICKING"
    )


def test_temporary_backpack_loss_does_not_create_false_pick():

    m = PickPlaceStateMachine(
        confirm_frames=3,
        leave_frames=5
    )

    seq = []

    # Initial placed.
    seq += [(False, True, True)] * 3

    # Start interaction.
    seq += [(True, True, True)] * 3

    # Backpack detection temporarily missing.
    # Wrist also outside because the cleaned pipeline signal
    # should not immediately cause a confirmed transition.
    seq += [(True, False, True)]

    # Backpack detected again.
    seq += [(True, True, True)] * 3

    final = run(m, seq)

    assert final == State.PICKING

    print(
        "test_temporary_backpack_loss_does_not_create_false_pick OK"
        " -> stayed PICKING"
    )


if __name__ == "__main__":

    test_full_happy_path()

    test_noisy_wrist_does_not_flip_state()

    test_temporary_person_loss_does_not_trigger_leave()

    test_false_start_reverts_to_placed()

    test_noisy_pickup_does_not_commit()

    test_temporary_backpack_loss_does_not_create_false_pick()

    print("\nAll state machine tests passed.")