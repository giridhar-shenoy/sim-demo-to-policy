# What transfers to hardware, and what does not

> **DRAFT — for Giri to edit.** Written by the session that built v0.1; first person is
> Giri's voice. Every factual claim about this repo points at code or a results file;
> the hardware paragraphs are claims for Giri to confirm or cut.

This repo trains an imitation policy entirely in simulation. Nothing here has run on a
robot, and the CPU-sized ACT policy does not yet solve the task even in simulation
(0/100; see the README's "Where it fails"). This page is my list of what I would expect to carry over to a real arm, what
would break first, and why, so that the success rates in the README are read as
simulation numbers and nothing more.

## Observation space

The policy sees two 96×96 RGB images (a fixed camera across the table and a camera on
the hand) and a 15-D state: TCP position, TCP orientation as a quaternion, finger gap
and seven joint angles. On a real Panda the state half transfers almost directly: joint
angles and the finger width come from the robot, and the TCP pose is forward kinematics
of the same joints.

The image half does not. The sim cameras have no noise, no motion blur, no exposure
changes, no lens distortion and perfectly flat lighting (shadows are switched off). To
keep the env fast, the observation cameras also draw the arm's collision meshes instead
of its visual meshes (`OBS_GEOM_GROUPS` in `sdp/envs/base_mujoco_env.py`). At 96 px the
two look the same, but it is one more way the training images are cleaner than any
real frame. The cube is also a saturated red block on a plain checkered table. A real
deployment would need, at minimum, a calibrated wrist camera, colour and lighting
randomisation in sim, and a few real demonstrations mixed into training.

## Action space

The action is a 6-DoF delta TCP pose plus a gripper command at 20 Hz. The env integrates
it into a target pose, clamps it to a workspace box and an orientation cone, and solves
damped least-squares IK onto the Panda's joint position actuators with gravity
compensation (`MujocoArmEnv._integrate_target` and `_solve_ik`). On a real arm I would
not send those IK solutions straight to the joints at 20 Hz. The policy output would
become a Cartesian setpoint, interpolated and tracked by a controller at the robot's
own rate (1 kHz on a Panda), with velocity and workspace limits enforced below the
policy. That layer is what I built on production 6-axis arms: Jacobian-based TCP
jogging and gravity-compensated hand guiding, both of which take exactly this kind of
delta-pose command. In this repo it is replaced by IK plus MuJoCo's position actuators,
which track far better than any real joint servo. The sim never shows the policy the
lag, overshoot or joint-limit behaviour of a real controller.

## Latency

In sim, the observation is rendered and the action applied within the same control
step, so latency is zero apart from the policy's own compute. The harness logs that
compute per episode (`policy_ms_mean` and `policy_ms_max` in `results/eval_act_*.csv`).
Most steps just pop a queued action. Every 10th step runs the network, so the maximum
is the number that matters. On hardware, add camera
exposure and transport (tens of milliseconds), and expect the action chunk to be
executed open-loop for 0.5 s. That is fine for a static cube and is the first thing
that would fail for anything that moves.

## Resets and seeds

Every sim episode starts from an exact, seeded state: the arm at its home keyframe and
the cube sampled from a known box. That is what makes the eval seeds meaningful and the
seed split auditable. A real cell has no seeds. Resets are done by a person or a
scripted reset routine, the initial-state distribution drifts, and "50 trials" becomes
half a day of supervised robot time. The statistical reading of the results (Wilson
intervals over independent trials) still applies. The independence assumption is the
part that gets harder to defend.

## Contact

The grasp is a parallel-jaw pinch on a 4 cm, 50 g cube. In MuJoCo that depends on soft
contacts, elliptic friction cones and `impratio = 10`, which I set so the cube does not
creep out of the fingers (`_build_spec`). The real Panda hand has different pad
friction, a different force profile and finger compliance that MuJoCo does not model.
Pick-and-place of a light cube is forgiving here. A contact-rich task such as peg
insertion (deferred to "What's next") is where the sim-to-real gap in contact dynamics
would show up first.

## Randomisation gaps

Only the cube's initial pose (x, y, yaw) is randomised. The target square, lighting,
colours, camera poses, table texture, cube size and mass, and robot dynamics are all
fixed. The "shifted" eval condition (the same box moved 8 cm) is the only robustness
probe in v0.1, and it is a geometric shift, not a visual one.

## What I would do first on a real arm

1. Put the policy behind a real Cartesian controller with velocity limits and a
   watchdog, and replay the scripted expert's trajectories open-loop to check the
   action scaling.
2. Calibrate the wrist camera and record a small set of real teleoperated
   demonstrations with the same feature schema (`sdp/record/lerobot_writer.py`), so sim
   and real data share one dataset format.
3. Evaluate with the same harness logic: listed initial conditions, one failure label
   per trial, and intervals on every success rate.
