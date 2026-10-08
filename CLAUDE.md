# Triton Droids simulation onboarding — working notes

Onboarding repo teaching MuJoCo through three tasks: `task1` scene construction,
`task2` imitation learning (behavior cloning), `task3` reinforcement learning.
`task0` is optional XML/kinematic-tree background.

**Current work: task 2.** Steps 1–4 of the plan below are done, committed and
verified. Steps 5–7 are not started.

---

## Environment

Use the conda env, not the system Python:

```bash
/opt/anaconda3/envs/mujoco_cpu/bin/python      # mujoco 3.3.4, gymnasium 0.29.1, numpy 2.2.6, scipy 1.13.1
```

`python3` on PATH is `/opt/anaconda3/bin/python3` (3.14.6, mujoco 3.15) — a
different MuJoCo version. Fine for JSON/text munging of the notebook, wrong for
running the sim.

**Not installed locally, by design:**

| Package | Why it matters |
|---|---|
| `mplib` | No macOS wheel; source build fails. Linux x86_64 cp38–cp312 only. The expert has an IK fallback, see below. |
| `torch` | Needed for notebook Parts 2–4 (steps 6–7). |
| `moviepy` | `RecordVideo` needs it; guarded, so its absence skips recording. |

MuJoCo must be loaded with an **absolute** XML path — relative paths break mesh
resolution (`assets/descriptions/assets/descriptions/...` double-prefix error).

Rendering: `MUJOCO_GL=glfw` on macOS, `egl` on Linux/Colab (headless, no X).

---

## Scene facts (`assets/descriptions/DropCubeInBinEnv.xml`)

```
nq=16  nv=15  nu=8  nkey=1  nsite=2
observation = [qpos, qvel] = 31      action = 7 arm joint targets (rad) + gripper (0–255)
```

Actuators are **position servos**, not torque motors — an action is a desired
joint angle. They settle ~10–50 mm short of the commanded pose under gravity;
see `servo_to_pose` for the correction.

| Name | Kind | Note |
|---|---|---|
| `link0` | body | arm base at `(-0.35, 0, 0.76)`; MJCF counterpart of URDF `panda_link0`, the planner's base frame |
| `hand`, `left_finger`, `right_finger` | body | MJCF uses these, **not** the URDF's `panda_hand` / `panda_leftfinger` |
| `cube` | body | free joint `cube_free`, qposadr **9**, dofadr **9**; qpos[9:16] = `[x,y,z,qw,qx,qy,qz]` |
| `simpleWoodBin` / `bin_mount` | body | there is no body named `bin` |
| `panda_hand_tcp` | **site** | grasp point between the fingers; `data.site_xpos`, not `data.xpos` |
| `bin_center` | **site** | middle of the bin opening, `(0.05, -0.2, 0.795)` |

Geometry: table top z=**0.76**; cube half-size **0.02** (4 cm), rests at z=0.78;
bin inner half-width **0.045**, floor 0.025 below `bin_center`, rim 0.025 above.
Gripper opens to 8 cm (0.04 per finger). TCP at the `home` keyframe is
`(0.265, 0, 0.9298)` with quaternion `[0, 1, 0, 0]` — **gripper pointing straight
down**, so that quaternion is the fixed grasp orientation for every episode.

Quaternions are MuJoCo's `[w, x, y, z]`. All pose accessors are world-frame.

---

## Three fixes that make the task solvable — do not revert

The repo as shipped could not complete this task. Diagnosed and fixed in
`572e87e`; each is commented in place.

1. **Cube was 10 cm.** `size` is a *half*-size and was `0.05`. Too wide for the
   8 cm gripper and the 9 cm bin interior. Now `0.02`.

2. **The whole Panda had collision disabled.** `contype="0" conaffinity="0"` sat
   on the top-level `panda` default class in `panda/mjcf/panda_asset.xml` and
   propagated to the collision meshes and fingertip pads, so the gripper passed
   through the cube. Enabled on the `collision` class, matching upstream MuJoCo
   Menagerie, where those flags live on `visual`.

3. **Gripper servo too weak to hold the cube.** `biasprm`'s second term is the
   servo stiffness kp. At the original `kp=100` a fully-closed command against a
   4 cm cube yields only ~2 N: the pads made intermittent *one-sided* contact
   (measured 5.1 N on one pad, 0.000 N on the other) and the cube slid out on the
   lift. Raised to `kp=500` (~10 N), with `gainprm` rescaled to `0.0784313725` so
   the ctrl 0–255 → width 0–0.04 m remap is unchanged. **0/7 → 7/7 lifts.**

Things that looked like the cause but were **not**, all tested and left alone:
solver `iterations=3` (an MJX-style setting), `impratio`, elliptic friction cone,
fingertip pad friction/`condim`, and ramping the gripper closed. None were
needed once the stiffness was right.

---

## Notebook map (`task2/imitation_learning.ipynb`)

Cell indices are stable; patch it with `json` rather than by hand.

| Cell | Contents | State |
|---|---|---|
| 2, 3 | Colab bootstrap (clone + pip install), rendering backend | done |
| 7 | reference scaffold of `TrajEnv`, annotated as such | leave stubbed |
| 8 | **the real `TrajEnv`** — accessors, `is_grasped`, `terminated`, `reset_model` | done |
| 15 | mplib planner setup, guarded import | done |
| 17 | `pick_cube_solution` expert + demo runner | done |
| 21 | dataset collection loop | wired up; `EPISODES` still 10_000 |
| 23 | `TrajectoryDataset` | **stub — step 6** |
| 25 | `Actor` network | **stub — step 6** |
| 27 | `eval_policy` (provided, do not rewrite) | — |
| 28 | training loop | **stub — step 7** |
| 30 | final evaluation | provided |

`require_id()` replaces the original `_safe_body_id`: `mj_name2id` returns `-1`
for an unknown name instead of raising, and `-1` silently indexes the *last*
row. That bug made `planner.set_base_pose` receive **the cube's pose** as the
arm's base frame. Never reintroduce a name lookup that tolerates `-1`.

### The expert (cell 17)

Seven phases: approach (12 cm above cube) → descend (5 mm above centre) → close
→ lift (20 cm) → traverse (16 cm above bin) → release → settle.

Motion backend is pluggable, and this is the only place the two differ:
- `planner.plan_screw(target_pose, qpos[:9])` when mplib imports
- damped-least-squares IK on the TCP site Jacobian otherwise

`servo_to_pose` adds the measured residual back into the setpoint each round
until the TCP is within 2 mm — without it the arm misses the cube entirely.

`record()` logs `(obs, action)` with the observation read **before** stepping, so
each pair is the state the expert saw when it chose that action. It reads state
off `env.unwrapped` because `RecordVideo` may wrap the env.

Measured locally with the IK backend: **10/10 success** at `cube_xy_noise=0.02`,
~556 steps/episode, **0.28 s/episode** (1000 episodes ≈ 5 min).

---

## Remaining plan

### Step 5 — collect the dataset (blocked on the fork, below)

Drop `EPISODES` in cell 21 from 10_000 to ~20 first, confirm the pickle is
non-empty and one trajectory looks sane, then scale to a few hundred.

Cell 21 already passes `cube_xy_noise=0.02`. **Keep it.** At the default `0.0`
every episode replays the identical layout and the dataset carries no
information — 10_000 byte-identical trajectories.

### Step 6 — dataset + network (cells 23, 25)

`TrajectoryDataset` flattens all episodes into one list of pairs and converts to
`float32` once in `__init__`. `Actor` is an MLP (31 → 256 → 256) with separate
`mean` and `log_std` heads of width 8, `log_std` clamped to `[-5, 2]`.

⚠️ **Normalize the actions.** The gripper's 0–255 range is ~80× the arm's
radians, so an unnormalized Gaussian NLL is dominated by the gripper dimension
and the arm barely learns. Either scale the gripper to [0,1] and invert it in
`eval_policy`, or standardize all 8 dims by dataset mean/std.

### Step 7 — train and evaluate (cell 28)

Loss is provided. Every ~10 epochs call `eval_policy` and checkpoint on
improvement. Set `shuffle=True` in the `DataLoader` — it ships `False`.

⚠️ `eval_policy` loops `while not (terminated or truncated)` but `TrajEnv`
always returns `truncated=False`. **Wrap the eval env in `TimeLimit`** or a
failing policy loops forever.

Closing question to answer: the policy is erratic because of compounding error /
covariate shift, plus demonstrations that are not a smooth function of state.

---

## Decisions already made

- **Steps 5–7 run in Colab with mplib**, not locally with the IK fallback. The
  user chose this; local collection is faster but bypasses the course's intent.
- **Work is committed** as it completes (`572e87e`, `2cccdc1`).
- `task1/visualize.py` has unrelated uncommitted edits by the user — leave them.

## Blocking the Colab run

`ml_onboarding` is the **course's shared branch**; do not push to it. Fixes 2
and 3 above live in `assets/descriptions/panda/mjcf/`, so uploading only the
scene XML to Colab silently restores the unsolvable scene. The user must fork,
push, and set `REPO_URL` in cell 2.

Open questions for the user's first Colab run:
1. Did `pip install mplib` succeed, and on which Python version (needs ≤3.12)?
2. Planner demo success rate?
3. Is `result["position"]` from `plan_screw` shaped `(T, 7)` or `(T, 9)`? The
   code slices `[:, :7]` to tolerate both, but this is unverified.

**The mplib branch has never been executed.** Everything else here was run and
measured locally. If it misbehaves, set `planner = None` to fall back to IK —
the same episode should succeed, which isolates planner bugs from scene bugs.

---

## Verification snippets

Run one expert episode and check success:

```bash
cd /Users/diegorosales/Downloads/simulation
/opt/anaconda3/envs/mujoco_cpu/bin/python -I - <<'EOF'
import json
nb = json.load(open('task2/imitation_learning.ipynb')); ns = {}
for i in (8, 15): exec(''.join(nb['cells'][i]['source']), ns)
exec(''.join(nb['cells'][17]['source']).replace("EPISODES = 10", "EPISODES = 3"), ns)
EOF
```

Visual check of the scene: `python task1/visualize.py` (opens a viewer), or
render offscreen with `mujoco.Renderer` and the `fixed` camera.
