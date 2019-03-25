# 数据说明

## MCTDH 与 ML-MCTDH 的比较

$$
\hat H = \hat T + \hat V
$$

### 模型：SHO

$$
V = \sum_{i=1}^d x_i^2 + c^2 \sum_{i=1}^{d} x_i x_{i+1}
$$

使用 Sine DVR（[-5.0, 5.0]）。

1. Case 1
    * $c = 0.5, d=4$
    * `steps=1000, ode_inter=0.01, cmf_step=10, method='RK23', fast=True, split=False`
    * MCTDH：$n_1 = 10, n_2 = 40$
    * MLMCTDH：$n_1 = 10, n_2 = 10, n_3 = 40​$，满二叉树。
    * $|a|-t$ 图像：
    ![case 1](.\figures\Figure_1.svg)
    * 数据文件：`data/SHO_1_*`
    * 测试参考：`tests/ml/cmp_2_3layers.py`
2. Case 2
    * $c = 0.5, d=4$
    * `steps=1000, ode_inter=0.01, cmf_step=10, method='RK23', fast=True, split=False`
    * MCTDH：$n_1 = 6, n_2 = 40$
    * ML–MCTDH：$n_1 = 3, n_2 = 6, n_3 = 40​$
    * $|a|-t$ 图像：
      ![case 2](.\figures\Figure_2.svg)
    * 数据文件：`data/SHO_2_*`
    * 测试参考：`tests/cmp_2_3layers.py`
3. Case 3
    * $c = 0.5, d=4$
    * `steps=1000, ode_inter=0.01, cmf_step=10, method='RK23', fast=True, split=False`
    * MCTDH：$n_1 = 6, n_2 = 40​$
    * ML–MCTDH：$n_1 = 10, n_2 = 40$，偏二叉树
    * $|a|-t$ 图像：
      ![case 3](.\figures\Figure_3.svg)
    * 数据文件：`data/SHO_3_*`
    * 测试参考：`tests/cmp_2_dmrg_style.py`
4. Case 4
    * $c = 0.5, d=8$
    * `steps=1000, ode_inter=0.01, cmf_step=10, method='RK23', fast=True, split=False`
    * MCTDH：$n_1 = 5, n_2 = 40$
    * ML–MCTDH：$n_1 = 10, n_2 = 40$，满二叉树
    * $|a|-t$ 图像：
      ![case 4](.\figures\Figure_4.svg)
    * 数据文件：`data/SHO_4_*`
    * 测试参考：`tests/cmp_2_4layers.py`

5. Case 5
    * Projector-splitting: 
        * ？？？