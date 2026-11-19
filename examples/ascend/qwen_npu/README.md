# Qwen NPU 训练入口

Qwen3.6-27B / Qwen3.8-27B 共用 Torch 2.10 + HyperParallel 环境。模型、数据和训练参数由一个 YAML 配置，启动器不再维护旧 Torch 2.7 分支。

## 目录

- `image/bin/`：配置检查、启动、信号转发和主机内存保护。
- `image/configs/`：两份256K任务模板、一份16卡拓扑、一份CP4配置。
- `image/train.sh`：镜像内启动脚本，固定解释器和运行环境。
- `train.sh`：转发到已安装的镜像入口。
- `install.sh`：部署入口和配置；不安装或升级训练依赖。
- `patches/`：固定版本HyperParallel所需的CPU offload/导出补丁。
- `tests/`：启动器配置检查测试。

不包含模型、数据、checkpoint、训练日志、wheel或历史实验目录。

## 环境与安装

已验证组合：Python3.11、torch2.10.0+cpu、torch_npu2.10.0、transformers5.8.0、LlamaFactory0.9.6.dev0、CANN9.0.0。`+cpu`并不表示不能使用NPU，设备支持由torch_npu提供。

解释器固定为 `/usr/local/share/venvs/qwen38/bin/python`。HyperParallel源码位于 `/usr/local/share/qwen38/hyper-parallel`，基础提交为 `0eeeb787429214154d135af181c57443e221b508`，应用 `patches/hyperparallel-cpu-offload.patch`。该补丁在镜像中已经应用，不要重复应用。

本目录不是从空白系统安装Ascend环境的脚本。已有匹配环境时，从仓库根目录安装当前分支的框架，再部署入口：

```bash
/usr/local/share/venvs/qwen38/bin/python -m pip install --no-deps --no-build-isolation .
bash examples/ascend/qwen_npu/install.sh
bash /usr/local/share/npu-training/train.sh --check-env
```

若单独使用本机 `/mnt/llamafactory-training/`，框架已安装，只需执行该目录的 `install.sh`。入口安装到 `/usr/local/share/npu-training/`，新节点不需要挂载源码目录。

HyperParallel声明的Transformers最低版本高于这套实测组合；当前保留5.8.0，不应直接执行无约束的全量依赖升级。

## 配置和启动

复制 `image/configs/qwen38-256k-16npu.yaml` 或 `image/configs/qwen36-256k-16npu.yaml` 为自己的任务文件，修改：

- `model_name_or_path`：本地模型目录。
- `dataset_dir` / `dataset`：数据目录及dataset_info.json中的注册名。
- `output_dir`：新的持久化输出目录。
- 训练超参数及 `launcher.nproc_per_node`。

```bash
# 配置和路径检查，不使用NPU、不创建训练输出
bash /usr/local/share/npu-training/train.sh /mnt/jobs/task.yaml --dry-run
# 额外检查实际可见NPU数量，不启动训练
bash /usr/local/share/npu-training/train.sh /mnt/jobs/task.yaml --check
# 前台启动
bash /usr/local/share/npu-training/train.sh /mnt/jobs/task.yaml
```

本机也可用 `bash /mnt/llamafactory-training/train.sh /mnt/jobs/task.yaml`。旧的 `/usr/local/share/llamafactory-training/train.sh` 仅作转发，不再包含独立实现。

平台启动命令，例如上传代码目录名为my-job：

```bash
bash /usr/local/share/npu-training/train.sh "${MA_JOB_DIR}/my-job/task.yaml"
```

不加nohup或后台`&`。模型和数据必须在作业内挂载到YAML所写位置。相对路径按任务YAML所在目录解析，可使用环境变量。

## 模板含义

模板为单节点16卡、CP4×DP4、microbatch1、梯度累积16、有效global batch64、cutoff262144、全参数SFT。默认 `npu_validation_steps: 2`，保留48步学习率计划，适合复现两步检查；这不是无限制的正式训练默认值。正式任务需调整 `max_steps`、warmup、保存策略，并将 `npu_validation_steps` 设为0。

```yaml
launcher:
  nproc_per_node: 16
  min_host_available_gib: 128
  max_host_used_fraction: 0.60
```

入口移除launcher后将其余字段传给LlamaFactory。未指定卡数时默认2卡；CP必须整除卡数，实际可见卡数须匹配。可通过 `launcher.accelerate_config` 或第二个位置参数提供单节点Accelerate配置，冲突会报错。多节点不在当前入口支持范围内。

可指定已有tokenized_path，但必须自行确认缓存对应当前模型、模板和截断设置；入口不会证明缓存内容。非空output_dir要求明确resume_from_checkpoint；换模型或数据应使用新输出目录。旧profile、内嵌dataset_info和optimizations不再接受，应从新模板迁移。

## 运行保护与输出

每5秒检查可用主机内存及cgroup占用，达到阈值后终止本任务进程组。SIGINT/SIGTERM也会转发，20秒未退出则强制终止。该保护不能保证平台绝不重启。

output_dir/runs/记录生成的任务配置、分布式配置、资源快照及退出状态；训练日志输出到平台控制台。检查命令不会启动训练或创建该目录。

## 验证边界

使用归一化精度调整前的代码。Qwen3.8完成256K两步、导出和重载，但尚未完全对齐GPU指标。Qwen3.6已完成第一步：约69.8分钟，旧记录约155.7分钟，loss绝对差约0.0000019、grad_norm相对差约0.0305%；第二步由用户中止，不声称完成两步或长期收敛验证。历史计时、编译缓存和并行方式不同，不能把速度差全部归因于单一框架。

启动器测试：

```bash
/usr/local/share/venvs/qwen38/bin/python examples/ascend/qwen_npu/tests/test_launcher.py
```

## 保存镜像

保留基础Python共享依赖、qwen38 venv、HyperParallel、npu-training及CANN。venv继承系统依赖，不能只复制venv。模型、数据、任务YAML和训练输出单独持久化；`/mnt`、`/home/ma-user/work`、`/cache`不能作为镜像内必需代码来源。保存镜像后仍需在新节点检查导入和短训练。
