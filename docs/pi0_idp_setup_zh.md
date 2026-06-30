# pi0-IDP LIBERO 复现实验配置说明

这份文档记录当前仓库里 PyTorch pi0 + IDP 实验的完整复现流程，目标是在一台新服务器上从零完成：

1. 通过 SSH 克隆代码和 submodule。
2. 配好 `uv` 的 PyTorch 训练环境。
3. 下载或准备 LIBERO 数据、pi0 预训练 checkpoint、norm stats。
4. 跑 flow baseline / IDP 训练。
5. 通过官方 LIBERO eval 脚本评测四个 suite。

下面所有路径都假设大盘目录是 `/2024233219`。如果新服务器的大盘路径不同，把 `/2024233219` 整体替换成对应路径即可。不要把 HuggingFace cache、checkpoint、eval 视频放到 `/root`，否则镜像和系统盘会很快膨胀。

## 0. 基本约定

推荐目录结构：

```bash
/2024233219/code/openpi-idp          # git 仓库
/2024233219/cache/huggingface        # HF 数据和模型缓存
/2024233219/cache/openpi             # OpenPI 转换后的 checkpoint 等
/2024233219/cache/openpi_checkpoints # 训练 checkpoint
/2024233219/cache/openpi_eval        # eval 日志和视频
```

建议先写一组环境变量，后续所有命令都复用：

```bash
export BIGDISK=/2024233219
export OPENPI_ROOT=$BIGDISK/code/openpi-idp
export OPENPI_CACHE=$BIGDISK/cache

export HF_HOME=$OPENPI_CACHE/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export XDG_CACHE_HOME=$OPENPI_CACHE/xdg

export OPENPI_DATA_HOME=$OPENPI_CACHE/openpi
export OPENPI_CKPT_HOME=$OPENPI_CACHE/openpi_checkpoints
export OPENPI_EVAL_HOME=$OPENPI_CACHE/openpi_eval

mkdir -p "$BIGDISK/code" "$HF_HOME" "$OPENPI_DATA_HOME" "$OPENPI_CKPT_HOME" "$OPENPI_EVAL_HOME"
```

如果服务器在国内网络，建议额外加：

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126
```

## 1. SSH 克隆代码和 submodule

这台服务器只接受 SSH 形式的 GitHub 地址，所以不要用 `https://github.com/...`。

```bash
cd "$BIGDISK/code"
git clone --recurse-submodules git@github.com:YizhifengyeYZM/openpi-idp.git
cd "$OPENPI_ROOT"

git submodule sync --recursive
git submodule update --init --recursive
```

确认 submodule remote 已经是 SSH：

```bash
git config --file .gitmodules --get-regexp url
git submodule status --recursive
```

应该能看到类似：

```text
submodule.third_party/aloha.url git@github.com:Physical-Intelligence/aloha.git
submodule.third_party/libero.url git@github.com:Lifelong-Robot-Learning/LIBERO.git
```

如果有旧 clone 里残留了 HTTPS 地址，重新同步：

```bash
git submodule sync --recursive
git submodule foreach --recursive 'git remote -v'
git submodule update --init --recursive
```

## 2. 配置 uv + PyTorch 主训练环境

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

在仓库根目录安装依赖：

```bash
cd "$OPENPI_ROOT"
uv sync
```

确认 PyTorch 和 transformers 版本：

```bash
uv pip show torch transformers
```

当前我们验证过的组合是：

```text
torch: 2.7.1+cu126
transformers: 4.53.2
```

OpenPI 的 PyTorch pi0 需要对 transformers 打 patch。每次新建 `.venv` 或重新安装 transformers 后都做一次：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

验证 CUDA 和 transformers：

```bash
uv run python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
PY
```

如果这里 `cuda available: False`，先不要开始训练，检查 NVIDIA driver、CUDA runtime、`CUDA_VISIBLE_DEVICES` 和 PyTorch wheel。

## 3. HuggingFace 登录和 LIBERO 数据

训练配置 `pi0_libero` 使用的是官方已经转换好的 LeRobot 数据集：

```text
physical-intelligence/libero
```

它是四个 LIBERO suite 合并后的训练集，不只是 Spatial。包括：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

登录 HuggingFace：

```bash
uv run huggingface-cli login
```

预热下载数据集：

```bash
cd "$OPENPI_ROOT"
LEROBOT_VIDEO_BACKEND=pyav uv run python - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("physical-intelligence/libero")
print("num_episodes:", ds.num_episodes)
print("num_frames:", ds.num_frames)
print("tasks:", len(ds.meta.tasks))
PY
```

如果这个脚本因为 LeRobot 版本 API 差异报属性错误，也可以直接跳过预热，训练脚本会自动触发下载。真正重要的是 cache 目录在大盘：

```bash
echo "$HF_HOME"
echo "$HF_LEROBOT_HOME"
du -sh "$HF_HOME" || true
```

训练时建议一直带：

```bash
export LEROBOT_VIDEO_BACKEND=pyav
```

### 3.1 如果集群推荐用 wget 下载 LIBERO

`physical-intelligence/libero` 是 HuggingFace dataset repo，完整文件大约 33GB，约 1699 个文件。不要只下载前几十个 parquet；必须把 `data/` 和 `meta/` 都下全。

仓库里已经提供脚本和静态文件清单。先确认直连 HuggingFace 文件下载可用：

```bash
wget --spider https://huggingface.co/datasets/physical-intelligence/libero/resolve/main/meta/info.json
```

如果输出里有 `200 OK` 或 `Remote file exists`，直接跑：

```bash
cd "$OPENPI_ROOT"
OPENPI_CACHE="$OPENPI_CACHE" scripts/download_libero_wget.sh
```

脚本默认会自动恢复中断下载：每个文件用 `wget -c` 断点续传；某个文件失败不会导致整个脚本退出；一轮扫完整个清单后如果还有缺失文件，会等待 60 秒后继续下一轮，直到下载完整为止。

如果 `wget` 报公司网关自签证书，例如 `Self-signed certificate encountered`，先尝试找管理员要公司 CA 文件，然后：

```bash
OPENPI_CACHE="$OPENPI_CACHE" scripts/download_libero_wget.sh \
  --ca-certificate /path/to/company-ca.pem
```

如果暂时拿不到 CA 文件，只能先绕过证书校验：

```bash
OPENPI_CACHE="$OPENPI_CACHE" scripts/download_libero_wget.sh \
  --no-check-certificate \
  --jobs 8
```

`--jobs 8` 会并发下载 8 个文件；如果公司代理开始 429、502、504 或频繁断，把它降到 `--jobs 4`。如果想限制最多重试 20 轮，每轮之间等 120 秒：

```bash
OPENPI_CACHE="$OPENPI_CACHE" scripts/download_libero_wget.sh \
  --no-check-certificate \
  --jobs 8 \
  --max-passes 20 \
  --retry-sleep 120
```

脚本会下载到：

```text
$HF_LEROBOT_HOME/physical-intelligence/libero
```

也会自动检查：

```text
1693 个 parquet
meta/info.json
HF_HUB_OFFLINE=1 的 LeRobotDataset 离线加载
```

如果只想检查本地文件，不下载：

```bash
scripts/download_libero_wget.sh --check-only
```

如果必须换下载源，例如公司内部镜像的 `resolve/main` 文件下载 URL，可显式传：

```bash
scripts/download_libero_wget.sh \
  --base-url https://huggingface.co/datasets/physical-intelligence/libero/resolve/main
```

当前观察到 `HF_ENDPOINT=http://mirrors.tools.huawei.com/huggingface` 会让 HuggingFace API 请求超时。使用本脚本下载时不需要设置 `HF_ENDPOINT`；如果之前设置过，先清掉：

```bash
unset HF_ENDPOINT HF_HUB_ENDPOINT
```

## 4. 准备 pi0 base 的 PyTorch checkpoint

`pi0_libero` 官方是从 `pi0_base` 大规模预训练 checkpoint 开始微调，不是从已经 LIBERO 微调过的模型开始。JAX checkpoint 地址是：

```text
gs://openpi-assets/checkpoints/pi0_base
```

PyTorch 训练脚本需要先把它转换成 PyTorch 格式。推荐输出到大盘：

```bash
export PI0_BASE_PT=$OPENPI_DATA_HOME/pytorch/pi0_base_pytorch
mkdir -p "$(dirname "$PI0_BASE_PT")"

cd "$OPENPI_ROOT"
uv run examples/convert_jax_model_to_pytorch.py \
  --config_name pi0_libero \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi0_base \
  --output_path "$PI0_BASE_PT"
```

转换完成后应当至少有：

```bash
ls -lh "$PI0_BASE_PT"
test -f "$PI0_BASE_PT/model.safetensors"
```

如果服务器不能直接读 `gs://`，先用 `gcloud` 下载到大盘，再转换：

```bash
mkdir -p "$OPENPI_DATA_HOME/jax/pi0_base"
gcloud storage cp -r gs://openpi-assets/checkpoints/pi0_base/params "$OPENPI_DATA_HOME/jax/pi0_base/"

uv run examples/convert_jax_model_to_pytorch.py \
  --config_name pi0_libero \
  --checkpoint_dir "$OPENPI_DATA_HOME/jax/pi0_base" \
  --output_path "$PI0_BASE_PT"
```

### 4.1 如果集群推荐用 wget 下载 pi0 base

OpenPI 官方 `pi0_base` 在 GCS 上，但公开对象可以通过 `https://storage.googleapis.com/...` 直接下载。仓库里已经提供脚本和静态文件清单：

```bash
cd "$OPENPI_ROOT"
OPENPI_CACHE="$OPENPI_CACHE" scripts/download_pi0_base_wget.sh
```

这个脚本也使用同样的自动恢复机制：断点续传、失败后进入下一轮、直到文件清单完整才开始转换 PyTorch checkpoint。

如果同样遇到公司网关自签证书：

```bash
OPENPI_CACHE="$OPENPI_CACHE" scripts/download_pi0_base_wget.sh \
  --no-check-certificate \
  --jobs 4
```

脚本会完成两步：

```text
下载 JAX checkpoint 到 $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base
转换 PyTorch checkpoint 到 $OPENPI_DATA_HOME/pytorch/pi0_base_pytorch
```

如果只下载 JAX checkpoint，不转换：

```bash
scripts/download_pi0_base_wget.sh --download-only
```

如果转换中断，重新转换：

```bash
scripts/download_pi0_base_wget.sh --force-convert
```

### 4.2 下载完成后的下一步脚本

上面这些步骤完成后，不需要手打验证和 smoke test 命令，直接用总控脚本：

```bash
cd "$OPENPI_ROOT"
OPENPI_CACHE="$OPENPI_CACHE" scripts/cluster_next_step.sh \
  --no-check-certificate \
  --jobs 8
```

默认会依次执行：

```text
检查 LIBERO 本地文件和离线加载
准备 pi0_base PyTorch checkpoint
跑 10-step smoke_flow 训练
```

如果只做某一步：

```bash
scripts/cluster_next_step.sh verify --no-check-certificate --jobs 8
scripts/cluster_next_step.sh prepare-pi0 --no-check-certificate --jobs 4
scripts/cluster_next_step.sh prepare-tokenizer --no-check-certificate
scripts/cluster_next_step.sh prepare-norm-stats
scripts/cluster_next_step.sh smoke --no-check-certificate --jobs 8 --gpu 0
```

LeRobot 提示 `physical-intelligence/libero` 是 2.0 数据集、当前代码兼容 2.1，这是 warning，不需要先升级数据集。真正会阻塞训练的是缺少 `assets/pi0_libero/physical-intelligence/libero/norm_stats.json`。总控脚本会在训练前检查；如果缺失，会自动运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero
```

也可以单独跑：

```bash
scripts/cluster_next_step.sh prepare-norm-stats
```

如果 smoke test 卡在 `paligemma_tokenizer.model`，说明 tokenizer 还没有提前下载到 `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`。更新代码后直接跑：

```bash
scripts/cluster_next_step.sh prepare-tokenizer --no-check-certificate
```

之后再重跑：

```bash
scripts/cluster_next_step.sh smoke --no-check-certificate --jobs 8 --gpu 0
```

smoke test 通过后，正式 flow / IDP 也可以直接用脚本启动：

```bash
scripts/cluster_next_step.sh train-flow --gpu 0
scripts/cluster_next_step.sh train-idp --gpu 1
```

## 5. norm stats / assets

训练和推理都需要 `pi0_libero` 的 normalization stats。仓库里通常已经有：

```bash
ls assets/pi0_libero/physical-intelligence/libero
```

如果缺失，重新计算：

```bash
cd "$OPENPI_ROOT"
LEROBOT_VIDEO_BACKEND=pyav uv run scripts/compute_norm_stats.py --config-name pi0_libero
```

确认：

```bash
find assets/pi0_libero -maxdepth 4 -type f | sort
```

## 6. wandb

服务器支持 online wandb 时：

```bash
uv run wandb login
export WANDB_MODE=online
```

训练脚本会把 loss、learning rate、grad norm、IDP 额外指标、部分 sample image 写到 W&B。IDP run 会多看到：

```text
loss_idp_zero
loss_idp_tau
idp_metric_excess_mean
idp_n_eff_mean
```

## 7. 单卡训练命令

官方 `pi0_libero` PyTorch 配置默认：

```text
num_train_steps = 30000
global batch_size = 32
optimizer = AdamW
lr schedule = cosine decay
default peak lr = 2.5e-5
save_interval = 1000
log_interval = 100
```

注意 `scripts/train_pytorch.py` 里的 `--batch_size` 是全局 batch size；DDP 时脚本会按 GPU 数切分到每张卡。

### 7.1 flow baseline，全 VLA 微调

```bash
cd "$OPENPI_ROOT"

CUDA_VISIBLE_DEVICES=0 \
LEROBOT_VIDEO_BACKEND=pyav \
uv run scripts/train_pytorch.py pi0_libero \
  --exp_name flow_full_30k \
  --project_name openpi-idp \
  --pytorch_weight_path "$PI0_BASE_PT" \
  --assets_base_dir ./assets \
  --checkpoint_base_dir "$OPENPI_CKPT_HOME" \
  --batch_size 32 \
  --num_workers 8 \
  --num_train_steps 30000 \
  --log_interval 100 \
  --save_interval 1000 \
  --pytorch_loss_type flow
```

### 7.2 IDP-Geo，全 VLA 微调

```bash
cd "$OPENPI_ROOT"

CUDA_VISIBLE_DEVICES=1 \
LEROBOT_VIDEO_BACKEND=pyav \
uv run scripts/train_pytorch.py pi0_libero \
  --exp_name idp_geo_full_30k \
  --project_name openpi-idp \
  --pytorch_weight_path "$PI0_BASE_PT" \
  --assets_base_dir ./assets \
  --checkpoint_base_dir "$OPENPI_CKPT_HOME" \
  --batch_size 32 \
  --num_workers 8 \
  --num_train_steps 30000 \
  --log_interval 100 \
  --save_interval 1000 \
  --pytorch_loss_type idp_geo \
  --pytorch_idp_tau 0.1 \
  --pytorch_valid_action_dim 7
```

### 7.3 只训练 action expert / projection 的消融

这个不是官方推荐主实验，但代码里保留了开关，可以做消融：

```bash
CUDA_VISIBLE_DEVICES=0 \
LEROBOT_VIDEO_BACKEND=pyav \
uv run scripts/train_pytorch.py pi0_libero \
  --exp_name idp_geo_freeze_paligemma_30k \
  --project_name openpi-idp \
  --pytorch_weight_path "$PI0_BASE_PT" \
  --assets_base_dir ./assets \
  --checkpoint_base_dir "$OPENPI_CKPT_HOME" \
  --batch_size 32 \
  --num_workers 8 \
  --num_train_steps 30000 \
  --log_interval 100 \
  --save_interval 1000 \
  --pytorch_loss_type idp_geo \
  --pytorch_idp_tau 0.1 \
  --pytorch_valid_action_dim 7 \
  --pytorch_freeze_paligemma
```

## 8. 八卡训练命令

八卡时推荐先跑官方等价的全量微调：全局 batch size 256，也就是每卡 32。这样更接近 pi0.5 LIBERO 配置里的大 batch 习惯，也比单卡 32 更稳定。

### 8.1 八卡 flow

```bash
cd "$OPENPI_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LEROBOT_VIDEO_BACKEND=pyav \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi0_libero \
  --exp_name flow_full_30k_bs256 \
  --project_name openpi-idp \
  --pytorch_weight_path "$PI0_BASE_PT" \
  --assets_base_dir ./assets \
  --checkpoint_base_dir "$OPENPI_CKPT_HOME" \
  --batch_size 256 \
  --num_workers 32 \
  --num_train_steps 30000 \
  --log_interval 100 \
  --save_interval 1000 \
  --pytorch_loss_type flow
```

### 8.2 八卡 IDP-Geo

```bash
cd "$OPENPI_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LEROBOT_VIDEO_BACKEND=pyav \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi0_libero \
  --exp_name idp_geo_full_30k_bs256 \
  --project_name openpi-idp \
  --pytorch_weight_path "$PI0_BASE_PT" \
  --assets_base_dir ./assets \
  --checkpoint_base_dir "$OPENPI_CKPT_HOME" \
  --batch_size 256 \
  --num_workers 32 \
  --num_train_steps 30000 \
  --log_interval 100 \
  --save_interval 1000 \
  --pytorch_loss_type idp_geo \
  --pytorch_idp_tau 0.1 \
  --pytorch_valid_action_dim 7
```

如果显存不够，先把全局 batch size 改成 128；如果数据加载成为瓶颈，再调大 `--num_workers`。

## 9. tmux 跑训练

推荐每组实验一个 tmux：

```bash
tmux new -s flow_full_30k
# 粘贴 flow 训练命令
```

另一个：

```bash
tmux new -s idp_geo_full_30k
# 粘贴 IDP 训练命令
```

常用 tmux 操作：

```bash
tmux ls
tmux attach -t flow_full_30k
tmux detach-client
```

检查 checkpoint：

```bash
find "$OPENPI_CKPT_HOME/pi0_libero" -maxdepth 3 -type d | sort | tail
```

训练完成后，目标 checkpoint 一般是：

```text
$OPENPI_CKPT_HOME/pi0_libero/flow_full_30k/30000
$OPENPI_CKPT_HOME/pi0_libero/idp_geo_full_30k/30000
```

如果训练脚本最后保存的是 `29999`，优先使用最后一个实际存在的目录。

## 10. LIBERO eval 环境

官方 README 推荐 Docker，但很多服务器 Docker/MuJoCo 图形配置比较麻烦。这里给无 Docker 的方式。LIBERO eval 环境和主训练环境分开，避免老版本 robosuite / torch 依赖污染训练用的 `.venv`。

```bash
cd "$OPENPI_ROOT"

uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

export PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/third_party/libero:$OPENPI_ROOT/packages/openpi-client/src
```

如果是无显示器服务器，优先试 OSMesa：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
```

如果 OSMesa 不可用，再试：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## 11. 启动 policy server

server 用主 `uv` 环境，eval client 用 `examples/libero/.venv`。

### 11.1 flow checkpoint server

```bash
cd "$OPENPI_ROOT"
export FLOW_CKPT=$OPENPI_CKPT_HOME/pi0_libero/flow_full_30k/30000

CUDA_VISIBLE_DEVICES=0 \
OPENPI_DISABLE_PYTORCH_COMPILE=1 \
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir "$FLOW_CKPT"
```

### 11.2 IDP checkpoint server

IDP checkpoint 的 metadata 会自动告诉 server 使用 `idp_geo` sampler。为了保险，也可以显式指定：

```bash
cd "$OPENPI_ROOT"
export IDP_CKPT=$OPENPI_CKPT_HOME/pi0_libero/idp_geo_full_30k/30000

CUDA_VISIBLE_DEVICES=1 \
OPENPI_DISABLE_PYTORCH_COMPILE=1 \
OPENPI_PYTORCH_LOSS_TYPE=idp_geo \
uv run scripts/serve_policy.py --port 8001 policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir "$IDP_CKPT"
```

### 11.3 flow checkpoint 的 1-step sampler 消融

这个用于检查“同样是 flow checkpoint，如果只走一步 ODE 会怎样”：

```bash
CUDA_VISIBLE_DEVICES=0 \
OPENPI_DISABLE_PYTORCH_COMPILE=1 \
OPENPI_SAMPLE_NUM_STEPS=1 \
uv run scripts/serve_policy.py --port 8002 policy:checkpoint \
  --policy.config pi0_libero \
  --policy.dir "$FLOW_CKPT"
```

## 12. 运行 LIBERO eval

每个 eval 命令连接一个已经启动的 server。suite 名字固定为：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

### 12.1 评测 flow 的 Spatial

```bash
cd "$OPENPI_ROOT"
source examples/libero/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/third_party/libero:$OPENPI_ROOT/packages/openpi-client/src
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID

python -u examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.video-out-path "$OPENPI_EVAL_HOME/videos/flow_full_30k/libero_spatial_50trial" \
  2>&1 | tee "$OPENPI_EVAL_HOME/flow_full_30k_libero_spatial_50trial.log"
```

### 12.2 评测四个 suite

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  python -u examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port 8000 \
    --args.task-suite-name "$suite" \
    --args.num-trials-per-task 50 \
    --args.video-out-path "$OPENPI_EVAL_HOME/videos/flow_full_30k/${suite}_50trial" \
    2>&1 | tee "$OPENPI_EVAL_HOME/flow_full_30k_${suite}_50trial.log"
done
```

IDP 只要把 port 和输出目录换掉：

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  python -u examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port 8001 \
    --args.task-suite-name "$suite" \
    --args.num-trials-per-task 50 \
    --args.video-out-path "$OPENPI_EVAL_HOME/videos/idp_geo_full_30k/${suite}_50trial" \
    2>&1 | tee "$OPENPI_EVAL_HOME/idp_geo_full_30k_${suite}_50trial.log"
done
```

如果只是快速 smoke test，把 `--args.num-trials-per-task` 改成 `5`。

## 13. 汇总 eval 结果

每个 log 最后一段会有：

```text
Total success rate: ...
Total episodes: ...
```

快速收结果：

```bash
grep -R "Total success rate\\|Total episodes" "$OPENPI_EVAL_HOME"/*.log
```

更整齐一点：

```bash
python - <<'PY'
import pathlib
import re

root = pathlib.Path("/2024233219/cache/openpi_eval")
for path in sorted(root.glob("*.log")):
    text = path.read_text(errors="ignore")
    rate = re.findall(r"Total success rate: ([0-9.]+)", text)
    episodes = re.findall(r"Total episodes: ([0-9]+)", text)
    if rate:
        print(f"{path.name}: success={float(rate[-1]) * 100:.1f}% episodes={episodes[-1] if episodes else '?'}")
PY
```

## 14. 常见问题

### 14.1 submodule 仍然走 HTTPS

```bash
git config --file .gitmodules --get-regexp url
git submodule sync --recursive
git submodule update --init --recursive
```

如果某个 submodule 内部 remote 还是 HTTPS：

```bash
cd third_party/libero
git remote set-url origin git@github.com:Lifelong-Robot-Learning/LIBERO.git
cd "$OPENPI_ROOT"
git submodule update --init --recursive
```

### 14.2 transformers 报 AdaRMS / KV cache / dtype 相关错误

基本都是忘了复制 patch：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

然后重新跑：

```bash
uv run python - <<'PY'
import transformers
print(transformers.__version__)
PY
```

### 14.3 训练报缺 norm stats

确认 assets 路径：

```bash
find assets/pi0_libero -maxdepth 4 -type f | sort
```

缺的话重新算：

```bash
LEROBOT_VIDEO_BACKEND=pyav uv run scripts/compute_norm_stats.py --config-name pi0_libero
```

### 14.4 LIBERO eval 报 `No module named libero`

确认 eval venv 和 PYTHONPATH：

```bash
source examples/libero/.venv/bin/activate
uv pip install -e third_party/libero
uv pip install -e packages/openpi-client
export PYTHONPATH=$PYTHONPATH:$OPENPI_ROOT/third_party/libero:$OPENPI_ROOT/packages/openpi-client/src
```

### 14.5 MuJoCo / OpenGL / EGL 报错

无头服务器优先：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
```

如果必须 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

### 14.6 数据或 checkpoint 下载到了 `/root`

检查：

```bash
du -sh ~/.cache ~/.cache/huggingface ~/.cache/openpi 2>/dev/null || true
du -sh "$OPENPI_CACHE"/* 2>/dev/null || true
```

以后启动 shell 时固定导出：

```bash
export HF_HOME=$OPENPI_CACHE/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_LEROBOT_HOME=$HF_HOME/lerobot
export XDG_CACHE_HOME=$OPENPI_CACHE/xdg
```

### 14.7 单卡 H20 能不能跑

可以跑当前代码里的单卡 `batch_size=32` pi0 LIBERO 微调；我们之前已经按这个规模跑过 flow 和 IDP。八卡时更推荐全局 batch size 256，也就是每卡 32。

### 14.8 这个仓库里的 IDP 改动是什么

核心改动：

1. `pytorch_loss_type=flow`：原始 pi0 flow matching。
2. `pytorch_loss_type=idp_iso`：加入 one-step IDP 目标，使用各向同性近端项。
3. `pytorch_loss_type=idp_geo`：加入基于 context 的对角几何权重，只对真实 LIBERO 7 维 action 计算几何。
4. `pytorch_freeze_paligemma=True`：可选冻结 PaliGemma trunk，只训练 action expert / projection，用于消融。
5. serve 时会从 checkpoint metadata 自动恢复 `pytorch_loss_type` 等字段；也可以用 `OPENPI_PYTORCH_LOSS_TYPE` 强制覆盖。
