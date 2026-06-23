# IGR — 本科毕设

本项目实现了基于隐式几何正则化（Implicit Geometric Regularization, IGR）的三维形状重建与形状空间学习方法，并结合 Instant-NGP 多分辨率哈希编码与 DiGS 稳定训练策略进行加速与改进。

## 目录结构

```
IGR-master/
├── code/                        # 源代码
│   ├── datasets/                # 数据集加载
│   │   └── dfaustdataset.py     #   D-Faust 数据集
│   ├── model/                   # 网络模型
│   │   ├── network.py           #   标准 MLP 隐式网络（ImplicitNet）
│   │   ├── hash_encoding.py     #   Instant-NGP 多分辨率哈希编码
│   │   └── sample.py            #   点采样器（NormalPerPoint / AdaptiveSurfaceSampler）
│   ├── preprocess/              # 数据预处理
│   │   └── dfaust.py            #   D-Faust 数据预处理脚本
│   ├── reconstruction/          # 单形状重建
│   │   ├── run.py               #   重建训练/评估入口
│   │   └── setup.conf           #   重建配置文件
│   ├── shapespace/              # 形状空间学习
│   │   ├── train.py             #   训练入口
│   │   ├── eval.py              #   评估入口
│   │   ├── interpolate.py       #   形状插值
│   │   ├── latent_optimizer.py  #   隐向量优化
│   │   └── dfaust_setup.conf    #   形状空间配置文件
│   ├── splits/                  # 数据划分（JSON）
│   │   └── dfaust/              #   D-Faust 训练/测试/插值划分
│   └── utils/                   # 工具
│       ├── general.py           #   通用工具函数
│       ├── plots.py             #   Marching Cubes 网格导出与可视化
│       └── eval_metrics.py     #   评估指标（Chamfer / F-Score / NC）
├── data/                        # 数据
│   ├── sphere_point/            #   球体点云（示例）
│   ├── cube_point/              #   立方体点云（示例）
│   ├── d-faust/                 #   D-Faust 原始数据
│   └── d-faust-processed/       #   D-Faust 预处理后数据
├── trained_models/              # 训练好的模型
│   └── dfaust_pretrained/       #   预训练形状空间模型
├── exps/                        # 实验输出
├── paper/                       # 论文文档
├── environment.yml              # Conda 环境配置
└── fixed.md                    # 修改记录与问题说明
```

## 环境配置

### 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate igr
```

### 关键依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.7.12 | |
| PyTorch | 1.10.0+cu111 | CUDA 11.1 |
| numpy | 1.21.6 | |
| scipy | 1.7.3 | KDTree 评估 |
| trimesh | 3.9.39 | 网格导出 |
| pyhocon | 0.3.63 | 配置文件解析 |
| scikit-image | 0.19.3 | Marching Cubes |

## 使用方法

### 1. 单形状重建

对单个点云（如球体、立方体）进行隐式表面重建。

```bash
cd code
python reconstruction/run.py --conf reconstruction/setup.conf --expname sphere --nepoch 10000 --gpu_index 0
```

配置文件 `reconstruction/setup.conf` 中可修改：
- `input_path`：输入点云路径
- `network.inputs.dims`：MLP 各层维度
- `plot.resolution`：Marching Cubes 分辨率

### 2. D-Faust 数据预处理

在形状空间学习前，需要先将 D-Faust 原始数据处理为点云 `.npy` 文件。

```bash
cd code
# 解压原始数据
tar -xvf 50007.tar

# 预处理（以 50007 为例）
python preprocess/dfaust.py \
    --src-path /path/to/d-faust/50007 \
    --out-path /path/to/d-faust-processed \
    --names 50007
```

参数说明：
- `--src-path`：原始 D-Faust 数据路径
- `--out-path`：输出路径
- `--names`：指定处理的人物 ID（逗号分隔）
- `--mode 0`：仅处理训练数据；`--mode 1`：仅处理测试数据

### 3. 形状空间训练

在 D-Faust 数据集上训练形状空间模型。

```bash
cd code
python shapespace/train.py \
    --expname dfaust_800epochs \
    --nepoch 800 \
    --conf dfaust_setup.conf \
    --batch_size 16 \
    --points_batch 8000 \
    --threads 4 \
    --split dfaust/my_train.json
```

### 4. 形状空间评估

在测试集上评估训练好的模型。

```bash
cd code
python shapespace/eval.py \
    --checkpoint 1200 \
    --exp-name dfaust_pretrained \
    --exps-dir trained_models \
    --split dfaust/my_test_models.json \
    -r 256
```

参数说明：
- `--checkpoint`：使用的训练轮次
- `--exp-name`：实验名称
- `--exps-dir`：模型目录
- `--split`：测试集 JSON 文件
- `-r`：Marching Cubes 分辨率（128 / 256）

### 5. 形状插值

在两个形状之间生成插值序列。

```bash
cd code
python shapespace/interpolate.py \
    --interval 10 \
    --checkpoint 1200 \
    --exp-name dfaust_pretrained \
    --exps-dir trained_models \
    --split dfaust/interpolate.json
```

## 核心技术

### 网络架构

支持两种网络：

1. **ImplicitNet**（`model/network.py`）：标准 MLP，带几何初始化
2. **HashGridImplicitNet**（`model/hash_encoding.py`）：Instant-NGP 多分辨率哈希编码 + 浅层 MLP，显著加速训练

在 `dfaust_setup.conf` 中通过 `network_class` 切换：

```hocon
# 标准 MLP
network_class = model.network.ImplicitNet

# Instant-NGP 哈希编码（推荐）
network_class = model.hash_encoding.HashGridImplicitNet
```

### 损失函数

| 损失 | 公式 | 说明 |
|------|------|------|
| Manifold Loss | `\|f(x)\|` | 表面点 SDF 值趋近于零 |
| Gradient Loss (DiGS) | `log(1 + (\|\|\nabla f\|\| - 1)²)` | 归一化梯度，替代 Eikonal |
| Normals Loss | `1 - cos(∇f, n)` | 梯度方向与法向量对齐 |
| Divergence Loss (代理) | `1 - cos(∇f_表面, ∇f_非表面)` | 梯度方向一致性，替代二阶 Laplacian |
| Latent Loss | `\|\|z\|\|²` | 隐向量正则化 |

> **散度损失代理**：原 IGR 通过二阶导数计算 Laplacian 散度，但哈希编码中的 `floor` 操作不可微，导致二阶导数为零。本项目改用梯度方向一致性作为一阶代理，兼容哈希编码且零额外计算开销。

### 采样策略

- **NormalPerPoint**：标准高斯采样（局部 + 全局）
- **AdaptiveSurfaceSampler**（DiGS）：自适应表面感知采样，在薄结构和尖锐特征处密集采样

### 混合精度训练

支持 AMP（Automatic Mixed Precision）训练，在 `dfaust_setup.conf` 中启用：

```hocon
train {
    use_amp = True
}
```

### 评估指标

| 指标 | 说明 |
|------|------|
| Chamfer Distance | 双向点云距离 |
| F-Score | 阈值内点的精确率/召回率调和均值 |
| Normal Consistency | 法向量一致性 |

## 配置说明

形状空间配置文件 `dfaust_setup.conf` 关键参数：

```hocon
train {
    latent_size = 256          # 隐向量维度
    use_amp = True             # 混合精度训练
    compute_eval_metrics = True # 训练时计算评估指标
}

network {
    inputs {
        # 哈希编码参数
        n_levels = 16              # 多分辨率层数
        n_features_per_level = 2   # 每层特征维度
        log2_hashmap_size = 19      # 哈希表大小（2^19）
        base_resolution = 16        # 基础分辨率
        per_level_scale = 2.0      # 层间缩放因子
        hidden_dim = 128           # MLP 隐藏层维度
        num_layers = 2             # MLP 层数
    }
    loss {
        lambda = 0.2           # 梯度损失权重
        normals_lambda = 0.5   # 法向量损失权重
        latent_lambda = 1e-4   # 隐向量正则化权重
        div_lambda = 1e-7      # 散度损失权重
        grad_clip_norm = 2.0   # 梯度裁剪
    }
}
```

## 引用

本项目基于以下工作：

- Gropp et al., *Implicit Geometric Regularization for Learning Shapes* (ICML 2020)
- Müller et al., *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding* (SIGGRAPH 2022)
- DiGS: *Diffusion Implicit Geometric Regularization* 的稳定训练策略
