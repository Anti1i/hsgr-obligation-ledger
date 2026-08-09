# DCH-HSGR �?H200 运行�?

自包含目录：代码 + 数据 + 已有 pilot 输出（jsonl）。拷�?H200 机器后按下面顺序跑�?
**优先级：先跑 S1 验收门槛**，通过后才值得投入主实验（理由�?`../DCH-HSGR_Feasibility_and_Route_v4_2026-08-09.md` 第二节）�?

---

## 0. 环境

```bash
pip install -r requirements.txt
# 模型：Qwen/Qwen2.5-7B-Instruct（主）、Qwen2.5-1.5B-Instruct（探�?RL 策略�?
# 若离线：export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

4×H200�?41GB/卡）�?`pilot.py` 的默�?batch 已按 H200 调大�?090 上需减半以上）�?

---

## 1. 数据准备（CPU，几秒）

```bash
python data_prep.py --which all --data-dir data
```

产出�?

| 文件 | 内容 | 用�?|
|---|---|---|
| `data/math_l5.jsonl` | MATH-500 �?Level 5 子集�?34 题） | 更大 oracle gap |
| `data/gsm_deep_test.jsonl` | GSM8K 标注步数 �? 的题�?56 题）�?*�?gold 中间�?* | S2 深层层级；首次可测节点级 oracle |
| `data/gsm_deep_train.jsonl` | 同上，train split�?661 题） | 势函�?RL 训练数据 |
| `data/gsm_chain_test.jsonl` | 组合�?GSM�?00 题）：B 中一个数�?A 的答案替换，gold 由重新求值得�?| 强制组合性，S1 主力 |

`gsm_chain` 的每一条都通过了恒等替换校验（用原数重新求值必须复现原 gold），并过滤了百分比、重复出现的数、非整数�?gold、量级偏�?>3× 的替换�?*仍建议人工抽�?20 �?*再正式使用�?
`gsm_deep` 直接来自 GSM8K �?`<<expr=val>>` 标注，最可靠�?

---

## 2. S1：设定验收（**先做这一�?*�?

在新数据集上�?depth-2 pilot，然后算验收门槛�?

```bash
bash run_s1.sh          # 4 卡分片跑 gsm_chain + math_l5
python s1_gate.py --dirs outputs_chain,outputs_mathl5 --gate 0.10
```

**门槛**：`gold 仅靠分解可得 > 10%`�?
- 现状对比：MATH-500 只有 **3.6%**、GSM8K **0.0%** �?层级叙事在旧设定上没有立足点�?
- 若两个新数据集都 FAIL �?触发路线 v4 止损条款：放�?层级分解"叙事，论文重定位为「候选域延迟提交 + 结构�?credit RL」�?

---

## 3. S2：深�?3 层级 + 双向消息传递重�?

V6 已证�?top-down 回传在深�?2 �?*完全�?no-op**�? 轮准确率不变，根 argmax 仅动 �?/136 题）。深�?3 让中间层同时有父和子，top-down 才有可传信息�?

```bash
bash run_s2.sh          # tree -> leaf -> mid -> root -> bp
```

�?`outputs_deep/s2_bp_report.json` �?`acc_by_round`�?
- �?r2–r4 相对 r1 有提�?�?双向传播这一 claim 成立，可写；
- 若仍完全不变 �?该组件从论文中移除（不要硬写）�?

---

## 4. G1 / G3：闭合验证链（依赖已�?outputs/，可�?S1 并行�?

```bash
bash run_g1_g3.sh
```

- **G1**：把根候选补采到 20 个，给出 SC@k �?accuracy–token 曲线。V7 已算�?DCH+probe �?**SC@7.9**、DCH+verify �?**SC@13** �?token 量，所�?SC@5 不是合法对照，必须有这条曲线�?
- **G3**：沿**真实推理轨迹**（含 25%/50%/75% 中途位置）取隐状态训探针。V3 用的是程序化合成负样本（题内 AUROC 0.842），G3 检验真实轨迹上是否也成立，并测能否在答案出现之前就判断（决�?ATLAS �?latent gating 是否可行）�?

---

## 5. 复现已完成的验证实验（可选，CPU�?

```bash
python verify_latent.py    --stage all      # V1 探针架构 / V2 跨域 / V3 推理有效�?
python verify_credit.py                     # V4 LOO vs 反事�?vs 频率
python verify_structure.py --stage all      # V5 conformal K / V6 双向传播 / V7 预算
python verify_headroom.py                   # V8 剩余空间分解
```

V1–V3 需�?`outputs*/hidden_feats.pt`（约 100MB，未打包）。需要时先重跑：

```bash
python phase06_hidden_probe.py --stage extract \
    --dirs outputs_gsm_train,outputs_math_train,outputs,outputs_gsm_test
```

---

## 6. 主实验（S1 通过后）

| ID | 内容 | 资源 | 时长估计 |
|---|---|---|---|
| E1 | 7B 全奖励矩阵（6 条件 × 3 seed），value-class LOO / GRPO / 局�?coverage / SetPO 风格 / 温度基线 / SFT | 2 卡一条，并行 2 �?| ~3.5 �?|
| E2 | 14B 头条（最优条�?vs GRPO vs SFT�?| 4 �?ZeRO-3 | ~3 �?|
| E3 | latent �?scaling（按域选层 + ranking loss + 数据扩到 gsm_deep_train�?| 1 �?| 1�? �?|
| E4 | conformal 节点�?K + OOD（深�?分支外推�?| 1 �?| 2 �?|

奖励实现要点（已�?V4 验证）：�?**value-class LOO credit**，不要用 raw-candidate LOO�?2% 退化为 0），也不要只奖励局�?gold�?

---

## 目录

```
    全部脚本（pilot / S1 / S2 / G1 / G3 / V1-V8 / 探针 / scorer�?
data/    原始与派生数据集
outputs*/            已有 pilot 输出（jsonl�?pt 特征未打包）
run_s1.sh run_s2.sh run_g1_g3.sh
```
