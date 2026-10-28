# Tasks

## PandaPickPlace-v0 (v0.1)

| Item | Value |
|---|---|
| Robot | Franka Emika Panda + Franka hand, mujoco_menagerie `franka_emika_panda` @ f3b6013 |
| Object | 4 cm cube, 50 g, friction 1.0, condim 4 |
| Goal | 6 cm square centred at (0.55, 0.15) m, drawn on the table, no walls |
| Initial cube pose | uniform in the nominal box (see docs/results.md); shifted box for the robustness eval |
| Control | 20 Hz; 6-DoF delta TCP pose (≤ 2 cm, ≤ 0.1 rad per step) + gripper, DLS IK onto joint position actuators |
| Physics | 500 Hz, implicitfast integrator, elliptic cones, impratio 10 |
| Observations | front + wrist RGB 96×96, 15-D state |
| Success | cube centre inside the square, cube on the table, finger gap ≥ 5 cm, cube speed < 1 cm/s, all held for 10 consecutive steps (0.5 s) |
| Episode limit | 250 steps (12.5 s); the episode ends at success |
| Scripted expert | `sdp/expert/waypoint.py`: above cube → descend → close → lift → above target → lower → open → retreat |

## Deferred (not in v0.1)

- **PandaPegInsert-v0**: 2 cm peg into a 2.2 cm chamfered hole, success = peg tip ≥ 2 cm below the hole surface.
- **So101PickPlace-v0**: the same task on `robotstudio_so101` (5 DoF; 4-DoF delta + gripper action).
