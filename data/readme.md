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
    * $c = 0.5, d=4​$
    * `cmf_steps=10, ode_method='RK23'`
    * `steps=1000, ode_inter=0.01, fast=True, split=False`
    * MCTDH：$n_1 = 10, n_2 = 40​$
    * ML-MCTDH：$n_1 = 10, n_2 = 10, n_3 = 40$，满二叉树。
    * $|a|-t​$ 图像：
      ![case 1](.\figures\Figure_1.svg)
    * 数据文件：`data/SHO_1_*`
    * 测试参考：`tests/ml/cmp_2_3layers.py`
2. Case 2
    * $c = 0.5, d=4​$
    * `cmf_steps=10, ode_method='RK23'`
    * `steps=1000, ode_inter=0.01, fast=True, split=False`
    * MCTDH：$n_1 = 6, n_2 = 40$
    * ML–MCTDH：$n_1 = 3, n_2 = 6, n_3 = 40$，满二叉树。
    * $|a|-t$ 图像：
      ![case 2](.\figures\Figure_2.svg)
    * 数据文件：`data/SHO_2_*`
    * 测试参考：`tests/cmp_2_3layers.py`
3. Case 3
    * $c = 0.5, d=4$
    * `cmf_steps=10, ode_method='RK23'`
    * `steps=1000, ode_inter=0.01, fast=True, split=False`
    * MCTDH：$n_1 = 6, n_2 = 40​$
    * ML-MCTDH：$n_1 = 10, n_2 = 40​$，偏二叉树
    * $|a|-t​$ 图像：
      ![case 3](.\figures\Figure_3.svg)
    * 数据文件：`data/SHO_3_*`
    * 测试参考：`tests/cmp_2_dmrg_style.py`
4. Case 4
    * $c = 0.5, d=8$
    * `cmf_steps=10, ode_method='RK23'`
    * `steps=1000, ode_inter=0.01, fast=True, split=False`
    * MCTDH：$n_1 = 5, n_2 = 40$
    * ML-MCTDH：$n_1 = n_2 = n_3 = 10, n_4 = 40$，满二叉树
    * $|a|-t$ 图像：
      ![case 4](.\figures\Figure_4.svg)
    * 数据文件：`data/SHO_4_*`
    * 测试参考：`tests/cmp_2_4layers.py`


上述四者的问题均可以通过改变多层情况下的初始态（尽量取低能态）来实现。

5. Case 5

    * Projector-splitting

        `snd_order` 关键字：`True/False` （版本 ID：`8b272a9d237bffc5dec20ecf757ece8d9341007a`）

    - $|a|-t$ 图像：

        ![case5](C:\Users\XinxianChen\Documents\minimisTN\data\figures\Figure_5.svg)

        其他参数：

        - $c = 0.5, d=2$
        - `cmf_steps=10, ode_method='RK23'`
        - `steps=300, ode_inter=0.1, fast=False, split=True`
        - MCTDH：$n_1 = 5, n_2 = 40$，随机初始态。
        - 数据文件：`data/SHO_5_*`

        几乎完全一致，但含义不同。尝试直接使用 `Tensor.linkage_visitor` 方法无法实现（在对 2L 的 SHO 模型测试（约 1 个周期）时，似乎仅仅差了一个与自由度有关的对时间的缩放系数，原因未知）。

6. Case 6

    - Projector-splitting
    - $|a|-t$ 图像：

    ![](C:\Users\XinxianChen\Documents\minimisTN\data\figures\Figure_6.svg)

    - $c = 0.5, d=2​$
    - `cmf_steps=10, ode_method='RK23'`
    - `steps=300, ode_inter=0.1, fast=False, split=True/False`
    - MCTDH：$n_1 = 5, n_2 = 40$
    - 数据文件：`data/SHO_6_*`

### 模型：SBM (无溶剂)

- ZT

    $\tau \sim 0.25~\text{fs}$

- FT (500 K)

	str_list = [
		'sbm-ft-split',
		'sbm-ft-split-case1',
	]




### 模型：SBM (含溶剂)

- ZT