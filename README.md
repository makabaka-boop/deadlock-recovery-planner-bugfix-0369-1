# Deadlock Simulator Backend

模拟作业—单实例资源系统中的授予/完成/死锁解除过程（至多 12 个作业、15 个资源）。

## 语义

- 每个资源至多由一个作业持有；每个作业可持有多个资源，同一时刻最多等待一个资源。
- 模拟器反复执行直至不能前进：
  1. **完成**：所有不再等待的作业完成并释放所持资源（按作业 ID 升序）；
  2. **授予**：每个有空闲等待者的资源授予**作业 ID 最小**的等待者（按资源 ID 升序应用）。
- 若仍有作业卡住（死锁），枚举可中止作业子集并独立重放，选出**总中止代价最小**
  且能使剩余作业全部完成的集合；代价并列时取**排序后中止 ID 序列字典序最小**者。
- 受保护（不可中止）作业导致无解时，返回 `unresolvable` 并明确报告，
  绝不强行释放受保护作业的资源。

## 文件

- `deadlock_simulator.py` — 核心：`build_state`（校验）、`simulate`（授予/完成回放）、
  `find_min_abort_set`（子集枚举 + 重放）、`solve`（完整求解）。
- `backend_server.py` — 标准库 HTTP 后端：`POST /simulate`、`GET /health`。
- `test_deadlock_simulator.py` — unittest 测试（无需第三方依赖）。

## 运行

```bash
python3 backend_server.py 8000        # 启动后端
python3 -m unittest test_deadlock_simulator -v   # 运行测试
```

## API

`POST /simulate`，请求体：

```json
{
  "jobs": [{"id": 1, "holding": [1], "waiting_for": 2, "abortable": true, "abort_cost": 5}],
  "resources": [{"id": 1, "holder": 1}, {"id": 2, "holder": null}]
}
```

响应 `status` 为 `completed` / `resolved_with_aborts` / `unresolvable`，
`events` 为 `grant` / `complete` / `abort` 事件的完整有序回放；
非法输入返回 400 及错误说明。


## 检查点恢复模式
原 JSON 增加 mode: "checkpoint" 与 checkpoints 数组（id、job、keep、cost）。
检查点只保留 keep 中的已持有资源，其余资源需在作业继续前重新取得；原等待需求也保留。
受保护作业不能中止，但可使用自身显式声明的检查点。每作业最多选一个检查点；
恢复要让所有原作业正常完成，先最小化回退总成本，再按排序的检查点 ID 列表裁决。
没有合法组合时不返回部分恢复计划，原中止恢复模式保持兼容。


回放中仍按原授予规则裁决：完成阶段先于授予阶段，空闲资源按资源 ID、等待者按作业 ID 排序。
回退作业先继续原等待请求，再按资源 ID 递增重新取得回退释放的资源；一次最多等待一个资源，
所有需求得到满足才可正常完成。回退后的授予与释放均须在回放中体现，未回退作业的需求不变。

检查点模式响应 `status` 为 `completed` / `recovered` / `unresolvable`；`events` 在
`grant` / `complete` 之外包含 `rollback` 事件（job、checkpoint、released、retained）；
`cost` 为回退总成本，`checkpoints` 为排序后的检查点 ID 列表。
