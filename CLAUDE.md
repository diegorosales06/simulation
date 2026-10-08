# Triton Droids simulation onboarding — working notes

Onboarding repo teaching MuJoCo through three tasks: `task1` scene construction,
`task2` imitation learning (behavior cloning), `task3` reinforcement learning.
`task0` is optional XML/kinematic-tree background.

**Current work: task 2.** All seven steps are done and run end to end locally
with the IK expert. `ONBOARDING_REPORT.html` at the repo root is the written
deliverable (self-contained, images embedded).

> **Still open:** the user intends to re-run steps 5–7 in Colab with the real
> mplib planner, which has never been executed. See "Where things stand".
> Everything local is finished; do not redo it.

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

## Four fixes that make the task work — do not revert

The repo as shipped could not complete this task, and could not score it
honestly either. Fixes 1–3 are in `572e87e`, fix 4 came later; each is
commented in place.

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

4. **The solver was loose enough to score failures as successes.** The scene
   shipped with `iterations="3" ls_iterations="5"` — an MJX-style setting for
   GPU batch training. On CPU that is too few to resolve contacts for a fast
   object: a cube knocked across the table at 4.5 m/s passes **through** the
   1 cm bin wall and lands inside, so `terminated` fires on a failed episode.
   This produced a bogus 5% policy success rate. Raised to `100/50`; the solver
   converges early, so it costs no measurable wall time (0.28 s/episode either
   way) and the expert is still 200/200.

   Found only because the metrics contradicted each other — "never lifts the
   cube" and "finishes the job 5% of the time" cannot both be true. Replaying
   that episode showed the cube peaked at 0.800 m against a 0.82 m rim.

Things that looked like the cause of the **grasp** failure but were not, all
tested and left alone: `impratio`, elliptic friction cone, fingertip pad
friction/`condim`, and ramping the gripper closed. None were needed once the
stiffness was right.

Note the subtlety on solver iterations: raising them does **not** fix grasping
(tested, no effect — fix 3 is what matters there), but it is required for
contact integrity at speed. Both things are true; do not read the grasp result
as a reason to put it back to 3.

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
| 21 | dataset collection loop | done; `EPISODES = 20`, `cube_xy_noise=0.02` |
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

Measured locally with the IK backend: **20/20 success** at `cube_xy_noise=0.02`,
~556 steps/episode, **0.28 s/episode** (1000 episodes ≈ 5 min). The mplib
backend is slower; time 20 episodes before scaling up.

---

## Remaining plan

### Steps 5–7 — done locally, still unrun with mplib

Cell 21 is ready: `EPISODES = 20` as a smoke test, `cube_xy_noise=0.02`.

**Keep both.** At `cube_xy_noise=0.0` every episode replays the identical layout
and the dataset carries no information. And the cell shipped at `EPISODES =
10_000`, which is hours of planning for a first run — raise it to a few hundred
only after 20 succeed and one trajectory has been eyeballed.

Measured locally with the IK backend: 200/200 success, 111,200 pairs at 200
episodes, ~556 steps per trajectory, 0.28 s/episode.

**Final local results** (200 episodes, 100 epochs, ~95 s training, corrected
physics): training loss -0.115 -> -37.3. Held-out copying error **0.4%** of a
typical movement (retrained on 160 episodes, tested on 40 unseen — so it
generalises, it has not memorised). Closed-loop: grips **35%**, lifts >=1 cm
**25%**, clears the bin rim **0%**, finishes **0%**. Best lift 37 mm against
the 60 mm needed.

The open-loop/closed-loop gap is the expected behavior-cloning failure, not a
pipeline bug: per-step error ~0.5 mm against a grasp clearance under 0.5 mm.
Do not "fix" this by hunting for a training bug.

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
- **Work is committed as it completes.** Do not leave a dirty tree.
- `task1/visualize.py` has unrelated uncommitted edits by the user — leave them.
- `.DS_Store` is untracked and should stay that way.

## Git layout — read before pushing anything

| Remote | Points at | Push? |
|---|---|---|
| `origin` | `triton-droids/simulation` | **No.** `ml_onboarding` is the course's shared branch, used by every other onboarding member. |
| `fork` | `diegorosales06/simulation` | Yes. This is where Colab clones from. |

Commits on `ml_onboarding`, oldest first (`b6c05b4` is the last upstream commit):

```
572e87e  scene physics fixes + env logic + expert policy   (steps 1-4)
2cccdc1  Colab setup cells actually set up Colab
3d9bb66  this file
b6726d9  EPISODES 10_000 -> 20
235c86b  REPO_URL -> the fork
```

Local and `fork/ml_onboarding` are both at `235c86b`. Local is 5 ahead of
`origin/ml_onboarding`, which is correct and should stay that way.

**After any change the Colab run depends on, push to `fork`** — Colab clones
from GitHub, so an uncommitted local edit is invisible to it.

## Where things stand

The fork is created, everything is pushed, and `REPO_URL` in cell 2 points at
it. The user is running this notebook:

```
https://colab.research.google.com/github/diegorosales06/simulation/blob/ml_onboarding/task2/imitation_learning.ipynb
```

Instructions given: run cells in order, **stop after cell 21**, because 23/25/28
are still stubs that raise `NotImplementedError`.

Verified by fetching raw files back from the fork — the fixes really are there:
`gainprm="0.0784313725" biasprm="0 -500 -50"`, `contype="1" conaffinity="1"` on
the collision class, `size="0.02 0.02 0.02"`, and the `panda_hand_tcp` site.

### Three answers to expect, and what each means

1. **Python version + did mplib install?** Needs ≤3.12; Colab is on 3.12. On
   3.13+ there is no wheel and the source build fails — then the realistic
   options are the IK backend or an older Colab runtime.
2. **Cell 17's success rate.** The go/no-go. If it is poor, have them set
   `planner = None` and re-run: IK gets 20/20 here, so IK-succeeds-mplib-fails
   isolates the planner integration from the scene.
3. **Shape of `result["position"]` from `plan_screw`** — `(T, 7)` or `(T, 9)`.
   `plan_waypoints` slices `[:, :7]` to tolerate both; this is the single
   unverified assumption in the mplib path.

**The mplib branch has never been executed.** Everything else in this file was
run and measured locally. Treat cell 17 as a real checkpoint.

### Offered and not yet answered

Writing Parts 2–4 (steps 6–7: `TrajectoryDataset`, `Actor`, training loop) ahead
of the user's Colab results, so one session covers steps 5–7 instead of three
round-trips. It cannot be run locally — no torch, no dataset — so it would be
untested either way. The user has not said yes or no.

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
