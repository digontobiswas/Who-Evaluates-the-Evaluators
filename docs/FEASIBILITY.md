# Stage 0: feasibility on PARAM Shakti

Checked on 8 Oct 2026. Every repository below was cloned and read at the commit given; nothing
was run on the cluster. Three labels are used throughout:

- **Verified**: checked directly, either in the upstream source or by running it in a Linux sandbox.
- **Documented**: stated by the authors or the cluster documentation, not checked by us.
- **Unknown**: not stated anywhere I could find. These need one of the commands in section 7.

## 1. Verdict

| Part of the study | Can it run on PARAM Shakti under the project rules? | What it needs |
|---|---|---|
| Open-loop L2 / ADE | Yes | CPU only; computed from the same logs and trajectories as NAVSIM. |
| NAVSIM v1 PDMS (navtest) | Yes, pending the probe | CPU only (the devkit runs agents on CPU). Python 3.9 from a relocatable build, wheels only. About 220 GB of data. |
| NAVSIM v2 EPDMS (navhard) | Yes, pending the probe | Same environment and CPU-only inference; 31 GB more data. Not yet exercised. |
| Extra navtrain policies (Population B) | Mostly yes | DiffusionDrive, WoTE, GoalFlow and GTRS are NAVSIM forks with no compiled operators. iPad needs mmcv-full. Each fork ships its own copy of the `navsim` package, so each needs its own checkout. |
| HUGSIM (3DGS closed loop) | **No, not as shipped** | Three CUDA extensions compiled from source, a conda-forge/pixi environment, and a wheel (open3d) that needs glibc 2.27. Fallbacks in section 4. |
| Generative world model | **Partly** | 16 GB V100s rule out most models. Vista is the only candidate that matches the cluster's constraints; none of the open models is multi-camera and action-conditioned, so a true closed loop with a three-camera policy is not available. Fallbacks in section 5. |
| Bench2Drive (CARLA) | Not assessed | Listed in the proposal for within-evaluator ordering only. Not looked at in this pass. |

The two rows in bold are on the critical path: the proposal uses the closed-loop evaluator as the
reference for the stage 2 monotonicity check. Both need a decision before the stage 1 go/no-go.

## 2. The cluster

From the public IIT Kharagpur HPC pages ([system configuration](https://hpc.iitkgp.ac.in/HPCF/systemConfig),
[job scheduling](https://hpc.iitkgp.ac.in/hpc_new/job-scheduling)). All **Documented**; section 7 confirms them.

| Fact | Value | Why it matters |
|---|---|---|
| GPU nodes | 22 nodes, 2 × NVIDIA V100 **16 GB** each, 40 cores, 192 GB RAM | 16 GB caps every generative model. 44 GPUs in total, shared with all users. |
| CPU nodes | 384 nodes, 40 cores each | NAVSIM v1/v2 scoring is CPU work, so it does not compete for GPUs. |
| OS | CentOS 7.6, which ships glibc 2.17 | Only `manylinux2014`-or-older wheels install. torch 2.0.1 and 2.4.x have such wheels; torch 2.7.1 publishes only `manylinux_2_28` and cannot be installed (**verified** on PyPI). |
| Partitions | `shared`, `medium`, `large`, `gpu` are named; GPU limits are not stated | Not used anywhere in the code. `configs/hpc/cluster.env` takes them from `sinfo`. |
| Storage | /home 40 GB soft, 50 GB hard; /scratch 2 TB soft, files older than 60 days deleted; jobs submitted from /scratch/$USER | Everything large goes under one scratch workspace. The 60-day window covers the project (39 days to deadline) but not much more. |
| NVIDIA driver, CUDA modules | **Unknown** | Per NVIDIA's compatibility table, CUDA 11.x builds need driver 450.80.02 or newer and CUDA 12.x builds need 525.60.13 or newer. Decides which world models are even installable. |

## 3. NAVSIM v1 and v2

Repository: [autonomousvision/navsim](https://github.com/autonomousvision/navsim). v1 is branch `v1.1`
(commit `3e8291b`, 5 Jun 2025); v2 is `main` (commit `0a380a9`, 27 Oct 2025).

| Item | NAVSIM v1 (PDMS, navtest) | NAVSIM v2 (EPDMS, navhard) |
|---|---|---|
| Licence | Code Apache-2.0. Data: OpenScene is tagged CC BY-NC-SA 4.0 and inherits the nuPlan dataset licence (non-commercial). | Same. |
| Python / PyTorch / CUDA | Python 3.9, torch 2.0.1, torchvision 0.15.2 (PyPI wheel bundles CUDA 11.7), pytorch-lightning 2.2.1, numpy 1.23.4, nuplan-devkit v1.2. Upstream installs with conda; nothing in it requires conda. | Identical `requirements.txt`. |
| Compiled CUDA extensions | None. **Verified.** | None. **Verified.** |
| Data | nuPlan maps (zip), `test` logs about 1 GB, `test` sensors **217 GB** in 32 camera and 32 LiDAR shards. navtest is a filter over `test`: 136 logs, 12,146 scenarios (**verified** from the devkit config). | The above plus `navhard_two_stage` synthetic views, **31 GB**. |
| GPU memory for inference | **None needed.** `AbstractAgent.compute_trajectory` runs the forward pass on CPU and never moves the model to a GPU. **Verified** in both branches. | Same. |
| Training data, only if we train | navtrain: 14 GB logs + 445 GB sensors (300 GB without history). | Same. |

What was verified in a sandbox (Linux, not CentOS 7):

1. The complete NAVSIM dependency set resolves to prebuilt wheels for CPython 3.9 on glibc 2.17.
   All 150 pinned packages were checked one by one against PyPI. Two pure-Python packages
   (`antlr4-python3-runtime`, `control`) have no wheel and build without a compiler.
2. The pinned relocatable CPython 3.9.25 has working `ssl` (OpenSSL 3.5.4), `lzma` and `venv`, and
   links against nothing newer than glibc 2.17. This is the workaround for the cluster's Python.
3. With that environment the v1.1 devkit imports, and Hydra composes both devkit entry points with
   the exact command lines our job builds.
4. `scripts/setup_env.sh` ran end to end in that sandbox.

What that does not prove: that it installs and runs on PARAM Shakti. The smoke job is the test.

Three things in the devkit that would have cost a failed job each:

- **The TransFuser backbone downloads ImageNet weights when it is constructed**
  (`timm.create_model(..., pretrained=True)`), even though the checkpoint overwrites them. On a
  compute node with no internet this fails. `fetch_assets.sh` caches the weights on the login node
  and proves the offline load works; the job sets `HF_HUB_OFFLINE=1`.
- **A missing sensor file does not stop the devkit.** The scenario is marked invalid and the average is
  taken over the rest, which would shift the PDMS silently. Our job checks every file before scoring
  and withholds the comparison if any scenario is missing or invalid.
- **The devkit sizes its worker pool from the node's core count,** not from the SLURM allocation. Our
  job passes the allocated core count explicitly.

### Baseline checkpoints

[autonomousvision/navsim_baselines](https://huggingface.co/autonomousvision/navsim_baselines), Apache-2.0,
4.06 GB in total. Three training seeds per model. File names and sizes are **verified** from the
Hugging Face API; published scores are from Table 1 of the NAVSIM paper
([arXiv:2406.15349v2](https://arxiv.org/abs/2406.15349)).

| Model | Inputs | Files | Published PDMS (navtest) |
|---|---|---|---|
| TransFuser | 3 front cameras + LiDAR + ego status | `transfuser/transfuser_seed_{0,1,2}.ckpt`, 672 MB each | 84.0 |
| Latent TransFuser (LTF) | 3 front cameras + ego status | `ltf/ltf_seed_{0,1,2}.ckpt`, 673 MB each | 83.8 |
| Ego Status MLP | ego status only | `ego_status_mlp/ego_status_mlp_seed_{0,1,2}.ckpt`, 6.5 MB each | 65.6 |

The paper reports a standard deviation of 0.56 PDMS across three TransFuser training seeds. A single
seed can sit half a point from 84.0, so the "within about 1 point" check is fairer against the
three-seed mean. Whether these checkpoints were also the ones behind the v2 navhard numbers is
**unknown**; the EPDMS reference values must be read from the v2 paper before that run, not assumed.

### Other public policies trained on navtrain (Population B candidates)

| Policy | Repo (commit date) | Licence | Weights | Published score | Notes |
|---|---|---|---|---|---|
| DiffusionDrive | [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive) (Dec 2025) | MIT | Hugging Face | PDMS 88.1 | NAVSIM v1 fork, ResNet-34, 60 M parameters, no compiled operators. The authors publish their **per-scenario score file**, so this one can be checked token by token. Best second reproduction target. |
| WoTE | [liyingyanUCAS/WoTE](https://github.com/liyingyanUCAS/WoTE) (Jun 2025) | Apache-2.0 | Google Drive | PDMS 88.3 | NAVSIM v1 fork. Authors state 22 GB GPU memory for training; inference not stated. |
| GoalFlow | [YvanYin/GoalFlow](https://github.com/YvanYin/GoalFlow) (Jul 2025) | Apache-2.0 | Google Drive | PDMS 90.3 | V2-99 backbone, needs `diffusers`. Google Drive downloads are awkward from a login node; fetch on a PC and copy. |
| GTRS-Dense / GTRS-Aug / Hydra-MDP (image only) / Diffusion Policy | [NVlabs/GTRS](https://github.com/NVlabs/GTRS) (Feb 2026) | Apache-2.0 | Hugging Face | EPDMS 41.7 / 42.1 / 37.5 / 25.6 (navhard) | NAVSIM v2 fork, V2-99 at 512×2048. Four checkpoints from one repo. |
| iPad | [Kguo-cs/iPad](https://github.com/Kguo-cs/iPad) (Jun 2026) | Apache-2.0 | Google Drive | not read | Requires `mmcv_full==1.7.2`, which contains compiled CUDA operators; usable only if a prebuilt wheel matches. |
| ReCogDrive (2B and 8B, three training stages each) | [xiaomi-research/recogdrive](https://github.com/xiaomi-research/recogdrive) (Aug 2026) | Apache-2.0 | Hugging Face | PDMS 84.1 to 90.8 | Vision-language models. GPU inference on 12k scenarios per variant is expensive on V100s. Optional. |
| AutoVLA | [ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA) (May 2026) | Academic-only licence | Hugging Face | not read | Large VLA; licence restricts use to academic research. Optional. |

GPU memory for inference is **not stated** for any of these. The ResNet-34 and V2-99 planners are small
enough that 16 GB is not a concern; the VLMs are the open question.

Counting three seeds each of TransFuser, LTF and Ego Status MLP, plus DiffusionDrive, WoTE, GoalFlow
and the four GTRS checkpoints, there are 16 public checkpoints without touching the VLMs.

## 4. HUGSIM

Repository: [hyzhou404/HUGSIM](https://github.com/hyzhou404/HUGSIM) (commit `62c690d`, 8 Nov 2025), with
the policy side in [hyzhou404/NAVSIM](https://github.com/hyzhou404/NAVSIM) (commit `ca0ca7e`, 29 May 2025).

| Item | Finding |
|---|---|
| Licence | MIT (code and the [XDimLab/HUGSIM](https://huggingface.co/datasets/XDimLab/HUGSIM) scene release). The policy client is Apache-2.0. |
| Python / PyTorch / CUDA | Python 3.11.10, torch 2.4.1 + cu118, installed with **pixi** from conda-forge. |
| Compiled CUDA extensions | **Yes, from source:** a gsplat fork (`HUGSIM_splat`), `simple-knn`, `tiny-cuda-nn`; plus `pytorch3d` and NVIDIA `apex` for data preparation. The closed-loop path itself imports the first three. **Verified** in `pixi.toml` and the imports. |
| Other blockers | `open3d` 0.18 publishes only `manylinux_2_27` wheels, so it cannot install on glibc 2.17 (**verified** on PyPI) and `closed_loop.py` imports it. The launcher calls `zsh`. |
| Data | Scenes, 3DRealCar vehicles and scenarios: **60.7 GB**. Scenes are reconstructed from nuScenes, KITTI-360, Waymo and PandaSet. Scenarios that load HD maps read the nuScenes dataset, which is already on the cluster. |
| GPU memory | **Not stated.** The simulator and the policy are designed to sit on separate GPUs, which matches the two V100s per node. |
| Policies it can host | UniAD, VAD and **LTF only**. The LTF client feeds nuScenes front cameras resized to 1920×1080, zero LiDAR, and a driving command fixed to "straight". TransFuser cannot be hosted (no LiDAR). |

Why this matters beyond installation: HUGSIM scenes are not nuPlan scenes, so the policy is run
outside its training camera geometry, and only camera-only bases can enter the closed-loop row of
the matrix. Population A would be LTF-based in that row.

Fallbacks, in the order I would try them:

1. **BridgeSim** ([VAIL-UCLA/BridgeSim](https://github.com/VAIL-UCLA/BridgeSim), Apache-2.0, CoRL 2026,
   last commit 24 Sep 2026). Closed-loop evaluation of NAVSIM policies (TransFuser, LTF, Ego MLP,
   DiffusionDrive) on converted navhard and navtest scenarios, built on MetaDrive. No compiled CUDA
   extensions; Python 3.9. It stays in the NAVSIM data domain, so the same scenarios appear in every
   row. It distributes its own checkpoint files; whether the `navsim_baselines` files load directly is
   **unknown**. Other caveats: its environment pins torch 2.7.1, which cannot install on glibc 2.17, so the pins
   would have to be relaxed to the NAVSIM set; it renders with Panda3D and needs headless OpenGL/EGL
   on the compute nodes (**unknown**); the camera images are graphics renders, not real images. It
   is a graphics simulator, so it would replace the "3DGS" family with a different one.
2. **HUGSIM off the cluster**, on a workstation or rented GPU with a normal Ubuntu. Results come back
   as per-scenario files exactly like the cluster runs. Costs money or a favour, not engineering risk.
3. **HUGSIM on the cluster with wheels built elsewhere**: build the three extensions in a
   `manylinux2014` container against CUDA 11.8 and copy the wheels over. This is still custom CUDA
   compilation, only not on the cluster, and open3d would need replacing. I would not start this
   without your explicit go-ahead; I estimate two to three days with a real chance of failure.

## 5. Action-conditioned driving world models

| Model | Repo (commit date) | Licence | Python / PyTorch / CUDA | Compiled extensions | Training data, views | GPU memory (authors) | Fit |
|---|---|---|---|---|---|---|---|
| **Vista** | [OpenDriveLab/Vista](https://github.com/OpenDriveLab/Vista) (Jul 2025) | Apache-2.0 (per its README) | Python 3.9, torch 2.0.1, CUDA 11.7 | None built from source (`xformers` ships wheels) | OpenDV + nuScenes; **front camera only**; conditions on trajectory, steering/speed, command or goal | "Minimum 32 GB" suggested; a low-VRAM mode exists | Same Python and torch as NAVSIM. You have already run it in 12 GB. Includes a reward estimator. Best fit to the cluster. |
| **ReSim** | [OpenDriveLab/ReSim](https://github.com/OpenDriveLab/ReSim) (May 2026) | Code Apache-2.0; weights under the CogVideoX licence (academic use allowed) | Python 3.10, torch 2.4.0, CUDA 12.4 | No CUDA extension; `deepspeed` has no wheel and installs from its source package (untested) | OpenDV + **NAVSIM**; front camera; conditions on a future trajectory | Not stated. CogVideoX-2B plus a T5-XXL text encoder | Best fit scientifically: trained on NAVSIM and built for non-expert actions. But only the expert-action weights are released, 16 GB is doubtful, and CUDA 12.4 needs a recent driver. |
| Epona | [Kevin-thu/Epona](https://github.com/Kevin-thu/Epona) (Jul 2025) | MIT | Python 3.10, torch ≥ 2.1 + cu121 | None | **nuPlan** and nuScenes; front camera; trajectory-controlled | "A single 4090" (24 GB) | Right data domain, but sized for 24 GB. |
| DriveArena | [PJLab-ADG/DriveArena](https://github.com/PJLab-ADG/DriveArena) (May 2025) | Apache-2.0 | mmcv-full 1.4.5, mmdet 2.20 stack | **Yes** (mmcv operators) | nuScenes and nuPlan; multi-view, conditioned on layouts with a traffic manager | Not stated | The only real multi-view closed loop, but it hosts UniAD/VAD, not NAVSIM agents, and needs compiled operators. Not feasible. |
| Cosmos-Predict2.5 | [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Code Apache-2.0; weights NVIDIA Open Model License | Not read | Not read | Driving model is 7-camera but text/image/video-conditioned; the action-conditioned model is for robots | Not stated | No action-conditioned driving model. Not feasible. |

Two structural problems, independent of installation:

1. **No open model is both multi-camera and action-conditioned in the NAVSIM camera setup.** Vista,
   ReSim and Epona generate the front camera only; TransFuser and LTF read three front cameras. So
   the generated frames cannot be fed back to these policies, and "closed-loop rollout in generated
   video" as written in the proposal is not available.
2. **A world model needs a scoring rule to act as an evaluator.** What can be built with open parts
   is single-step: take the real observation, condition the world model on the policy's planned
   trajectory, and score the imagined future (Vista's own reward estimator does this). That is a
   world-model-as-critic, and it sees a perception failure only through the trajectory it causes.
   H2 ("world models detect perception failures but miss action failures") has to be re-read against
   that design before stage 2.

Fallback order: Vista as a single-step critic on a scenario subset; ReSim if the driver and memory
allow; otherwise report the world-model row as not obtainable on this hardware and say so.

## 6. Points the proposal should absorb

- **The baseline policies are single-frame and use three cameras.** "History removal" has nothing to
  remove in TransFuser or LTF, and camera dropout can blank at most three cameras that matter, not
  four of eight. "Stale observation" does work: the logs hold frames 0.5, 1.0 and 1.5 s old.
- **BridgeSim is not in the related-work table and overlaps with the project's premise.** It is titled
  "Unveiling the OL-CL Gap in End-to-End Autonomous Driving" (arXiv 2604.10856, April 2026, CoRL 2026)
  and evaluates NAVSIM policies closed loop on navhard. The same group posted DreamStream and
  NavSafe-∞ in September 2026. They study the gap and adapt policies; they do not test evaluators
  against a known ordering, but a reviewer will ask.
- **Population C can be had without training:** three public seeds per baseline already exist. A
  training run on V100s needs navtrain (300 to 445 GB) and an unknown number of GPU-days.

## 7. Commands to run on the cluster

Both are read-only and submit or install nothing beyond one five-minute probe job.

```bash
# 1. Login node. Partitions, limits, quotas, modules, Python, outbound network.
bash scripts/hpc_probe.sh 2>&1 | tee hpc_probe_login.txt

# 2. One GPU node. Driver version, GPU memory, glibc, /tmp, network, EGL libraries.
#    Fill in the partition and gres exactly as step 1 printed them.
cd $SCRATCH
sbatch --partition=<GPU_PARTITION> --gres=<GRES> --time=00:05:00 \
       --output=wete-probe-%j.out <path to repo>/jobs/probe_node.sbatch
```

What each answer decides:

| Question | Command inside the probe | Decides |
|---|---|---|
| Partition names, time limits, gres string | `sinfo`, `scontrol show partition` | The values in `configs/hpc/cluster.env`. |
| Per-user job and GPU limits | `sacctmgr show assoc`, `sacctmgr show qos` | How many variants can run in parallel in stage 3. |
| Quota and free space | `myquota`, `lfs quota`, `df` | Whether 220 GB (v1) + 31 GB (v2) + 61 GB (HUGSIM) fit, and where the venv goes. |
| glibc version | `ldd --version` | Confirms the wheel target (expected 2.17). |
| Driver version and GPU memory | `nvidia-smi` | Which PyTorch CUDA builds work; whether ReSim is even installable. |
| Login-node access to GitHub, PyPI, Hugging Face, the nuPlan S3 bucket | `curl -sI` | Whether setup and downloads run as written or need the offline options. |
| Compute-node internet | `curl -sI` in the job | Confirms that everything must be pre-fetched. |
| `/tmp` and `/dev/shm` on a compute node | `df` | Whether Ray can start for metric caching. |
| EGL / OpenGL libraries | `find /usr/lib64` | Whether the BridgeSim fallback can render. |
| `git`, `curl`, `wget`, `unzip`, `zsh`, `screen` | `command -v` | Small script assumptions. |
