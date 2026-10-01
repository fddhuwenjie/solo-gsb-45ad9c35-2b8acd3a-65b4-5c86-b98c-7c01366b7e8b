"""无中断改线演练（hitless rewire）。

操作员提交：
  - 当前配置 config（设备 / 插孔对 / 常态模式 / 当前路由列表）；
  - 目标路由列表 target.routes（设备与插孔对不变，只改路由）；
  - 必须始终保持通声的路由序号 protected（从 0 起；同序号路由在现状与
    目标中的源 / 目标端口必须完全一致）。

系统先按当前配置重建已安装跳线（solver.plan_all），再对目标路由列表做一次
完整规划（目标配置非法或不可解时整体失败，不产生任何部分方案），然后在
「全常态 / 半常态 / 直通 + 插孔占用」规则下，用 A* 搜索从现状跳线集合到
目标跳线集合的逐步拆线 / 换接 / 补线序列：

  - 一个宏操作 = 拆线（拔两端，2 次拆装）、补线（插两端，2 次拆装）或
    换接（兼容跳线的一端拔下改插，1 拔 1 插，2 次拆装）；
  - 每个原子插拔后都建立快照（含改接中悬空的单个插头），保护路由必须在
    每个快照上仍有 src → dst 有向通路，否则该操作顺序不可用；
  - A* 的可采纳启发式按「旧边 / 新边之间共享插孔的最大二分匹配」估算最少
    拆装次数，因此结果优先复用兼容跳线、拆装次数全局最少；
  - 拆装次数并列时，优先不产生临时跳线（最终目标之外、中途用完即拆的
    物理跳线）的序列；
  - 搜索不到可行序列时返回结构化冲突（REWIRE_BLOCKED）与阻塞原因：
    在「只允许使用当前空闲插孔做临时跳线」的乐观图上找出保护路由必经、
    而目标又必须断开的边（第一道无法跨越的失声点）。
"""

from heapq import heappop, heappush

from solver import FULL, THROUGH, MODE_LABELS, ConfigError, _pid, parse_config, plan_all

NODE_CAP = 400_000                       # A* 展开节点上限（超出返回 SEARCH_LIMIT）


class RewireError(ConfigError):
    """无中断改线演练请求非法 / 无解。"""

    def __init__(self, message, code="REWIRE_INVALID", details=None, status=400):
        super().__init__(message, code=code, details=details)
        self.status = status


# ---- 环境与快照 -----------------------------------------------------------


class _Env:
    """静态物理环境：设备 / 插孔对 / 端口-插孔映射。"""

    def __init__(self, devices, pairs, port_jack):
        self.devices = devices
        self.pairs = pairs
        self.port_jack = port_jack
        self.jack_port = {j: p for p, j in port_jack.items()}
        self.pair_of = {}
        for p in pairs:
            self.pair_of[f"{p['id']}T"] = p
            self.pair_of[f"{p['id']}B"] = p
        self.normal_from = {}
        for p in pairs:
            if p["mode"] != THROUGH and p["top"] and p["bottom"]:
                self.normal_from[_pid(p["top"])] = (_pid(p["bottom"]), p)
        self.t_jacks = sorted(j for j, p in self.jack_port.items()
                              if j.endswith("T"))
        self.b_jacks = sorted(j for j, p in self.jack_port.items()
                              if j.endswith("B"))
        self.port_dir = {}
        for name, d in devices.items():
            for p in d["inputs"]:
                self.port_dir[f"{name}.{p}"] = (name, "in")
            for p in d["outputs"]:
                self.port_dir[f"{name}.{p}"] = (name, "out")


class _Snap:
    """某一时刻的面板快照：已完成跳线 edges + 改接中悬空插头 hang_t/hang_b。"""

    def __init__(self, env, edges, hang_t=(), hang_b=()):
        self.env = env
        self.edges = set(edges)
        self.plug_t = set(hang_t)
        self.plug_b = set(hang_b)
        for t, b in self.edges:
            self.plug_t.add(t)
            self.plug_b.add(b)
        self.active = {}
        for p in env.pairs:
            pid = p["id"]
            t, b = f"{pid}T", f"{pid}B"
            if p["mode"] == THROUGH or not p["top"] or not p["bottom"]:
                self.active[pid] = False
            elif p["mode"] == FULL:
                self.active[pid] = t not in self.plug_t and b not in self.plug_b
            else:  # HALF：仅下插孔插入断开常态
                self.active[pid] = b not in self.plug_b
        self._cable_from_t = {t: b for t, b in self.edges}

    def _neighbors(self, u):
        env = self.env
        name, direction = env.port_dir[u]
        if direction == "in":
            for o in env.devices[name]["outputs"]:
                yield f"{name}.{o}", 0, ("device", name)
            return
        if u in env.normal_from:
            dst, pair = env.normal_from[u]
            if self.active[pair["id"]]:
                yield dst, 0, ("normal", pair)
        ju = env.port_jack.get(u)
        if ju and ju in self._cable_from_t:
            yield env.jack_port[self._cable_from_t[ju]], 1, ("cable",
                                                             (ju, self._cable_from_t[ju]))

    def find(self, src, dst):
        """0-1 Dijkstra 最短路（确定性：邻接按端口 / 插孔字典序）。

        返回 (跳线数, items)；items 为 ("port", 端口) / ("edge", 描述) 交替。"""
        dist = {src: 0}
        prev = {}
        pq = [(0, src)]
        while pq:
            d, u = heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            if u == dst:
                break
            for v, w, edge in self._neighbors(u):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = (u, edge)
                    heappush(pq, (nd, v))
        if dst not in dist:
            return None
        items = [("port", dst)]
        u = dst
        while u != src:
            pu, edge = prev[u]
            items.append(("edge", edge))
            items.append(("port", pu))
            u = pu
        items.reverse()
        return dist[dst], items

    def audible(self, src, dst):
        """不计跳线数的可达性 BFS。"""
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for v, _, _ in self._neighbors(u):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    def signature(self, src, dst):
        """路径的稳定签名（不可达为 None），用于检测某条路由的通路是否变化。"""
        found = self.find(src, dst)
        if not found:
            return None
        _, items = found
        sig = []
        for kind, payload in items:
            if kind != "edge":
                continue
            if payload[0] == "device":
                sig.append(("d", payload[1]))
            elif payload[0] == "normal":
                sig.append(("n", payload[1]["id"]))
            else:
                sig.append(("c", payload[1]))
        return tuple(sig)


# ---- 启发式：最大共享插孔匹配 ---------------------------------------------


def _max_matching(rem, add):
    """rem / add 边集合之间「共享至少一个插孔」的最大二分匹配大小。"""
    rem = list(rem)
    add = list(add)
    adj = {}
    for i, (t1, b1) in enumerate(rem):
        adj[i] = [j for j, (t2, b2) in enumerate(add) if t1 == t2 or b1 == b2]
    match = {}

    def augment(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match or augment(match[j], seen):
                match[j] = i
                return True
        return False

    for i in range(len(rem)):
        augment(i, set())
    return len(match)


def _heuristic(E, G):
    rem = E - G
    add = G - E
    shared = _max_matching(rem, add)
    # 每对共享插孔的旧/新边可合成一次换接（2 次拆装）；其余旧边拆线、
    # 新边补线（各 2 次拆装）。
    return 2 * len(rem) + 2 * len(add) - 2 * shared


# ---- 宏操作展开（含原子插拔保护检查） -------------------------------------


def _gate_ok(env, edges, hang_t, hang_b, protected):
    snap = _Snap(env, edges, hang_t, hang_b)
    return all(snap.audible(src, dst) for _, src, dst in protected)


def _expand_macro(env, E, macro, protected):
    """把宏操作展开为原子插拔序列，选择每个中间快照保护路由均通声的顺序。

    返回 (atoms, None)；atoms 为 [("plug"|"unplug", jack, 端口)]。
    无安全顺序时返回 (None, reason)。"""
    kind = macro[0]

    def jack_port(j):
        return env.jack_port.get(j)

    if kind == "add":
        _, (t, b) = macro
        for order in ((t, b), (b, t)):
            j1, j2 = order
            # 第一插：单端悬空；第二插：跳线完成
            if not _gate_ok(env, E, {j1} if j1.endswith("T") else set(),
                            {j1} if j1.endswith("B") else set(), protected):
                continue
            if not _gate_ok(env, E | {(t, b)}, (), (), protected):
                return None, "补线完成后保护路由失声"
            return [("plug", j1, jack_port(j1)), ("plug", j2, jack_port(j2))], None
        return None, "补第一插即会断开保护路由（常态被插头断开且新跳线尚未完成）"

    if kind == "remove":
        _, (t, b) = macro
        rest = E - {(t, b)}
        # 优先先拔 B（半 / 全常态可随 B 拔出而恢复），再尝试先拔 T
        for first in (b, t):
            hang_t = {t} if first == b else set()
            hang_b = {b} if first == t else set()
            if not _gate_ok(env, rest, hang_t, hang_b, protected):
                continue
            return [("unplug", first, jack_port(first)),
                    ("unplug", (t if first == b else b), jack_port(t if first == b else b))], None
        return None, "拔下第一端后保护路由立即失声"

    if kind == "move-t":
        _, (t, b), (t2, _) = macro
        # 先从 t 拔下（B 端悬空，其余跳线仍在），再插入 t2
        if not _gate_ok(env, E - {(t, b)}, set(), {b}, protected):
            return None, "拔下 T 端、B 端悬空时保护路由失声"
        if not _gate_ok(env, (E - {(t, b)}) | {(t2, b)}, (), (), protected):
            return None, "换接到新 T 插孔后保护路由失声"
        return [("unplug", t, jack_port(t)), ("plug", t2, jack_port(t2))], None

    # move-b
    _, (t, b), (_, b2) = macro
    if not _gate_ok(env, E - {(t, b)}, {t}, set(), protected):
        return None, "拔下 B 端、T 端悬空时保护路由失声"
    if not _gate_ok(env, (E - {(t, b)}) | {(t, b2)}, (), (), protected):
        return None, "换接到新 B 插孔后保护路由失声"
    return [("unplug", b, jack_port(b)), ("plug", b2, jack_port(b2))], None


def _generators(E, G, env):
    """按 A* 探索偏好生成候选宏操作（不影响最优性，只影响出队顺序）。

    偏好：换接到目标边 > 补目标边 > 其他换接（临时跳线改位）>
    拔非目标旧边 > 临时补线 > 动到已是目标的边。"""
    occ_t = {t for t, b in E}
    occ_b = {b for t, b in E}
    free_t = [t for t in env.t_jacks if t not in occ_t]
    free_b = [b for b in env.b_jacks if b not in occ_b]

    goal_moves, temp_moves = [], []
    for (t, b) in sorted(E):
        for t2 in free_t:
            if t2 == t:
                continue
            e2 = (t2, b)
            (goal_moves if e2 in G else temp_moves).append(("move-t", (t, b), e2))
        for b2 in free_b:
            if b2 == b:
                continue
            e2 = (t, b2)
            (goal_moves if e2 in G else temp_moves).append(("move-b", (t, b), e2))

    goal_adds, temp_adds = [], []
    for t in free_t:
        for b in free_b:
            e = (t, b)
            if e in E:
                continue
            (goal_adds if e in G else temp_adds).append(("add", e))

    removes = [("remove", e) for e in sorted(E - G)]
    goal_removes = [("remove", e) for e in sorted(E & G)]   # 完整性：允许临时拆目标边
    return goal_moves + goal_adds + temp_moves + removes + temp_adds + goal_removes


# ---- 阻塞诊断 -------------------------------------------------------------


def _diagnose(env, E0, G, goal_snap, protected, explored, capped):
    """搜索失败时定位第一道失声点。

    乐观图 = 现有跳线 + 只用「当前空闲插孔」就能搭出的全部临时跳线，
    常态取当前生效集合。若保护路由在乐观图上仍依赖某条「目标必须拆除 /
    断开」的边，则该边是任何无中断顺序都无法跨越的第一道阻塞。"""
    occ_t = {t for t, b in E0}
    occ_b = {b for t, b in E0}
    free_t = [t for t in env.t_jacks if t not in occ_t]
    free_b = [b for b in env.b_jacks if b not in occ_b]

    opt = set(E0)
    for t in free_t:
        for b in free_b:
            opt.add((t, b))

    def reach(edges, active_override=None, src="", dst=""):
        snap = _Snap(env, edges)
        if active_override is not None:
            snap.active = dict(snap.active)
            snap.active.update(active_override)
            snap._cable_from_t = {t: b for t, b in edges}
        return snap.audible(src, dst)

    goal_active = dict(goal_snap.active)
    blockers = []
    for idx, src, dst in protected:
        gates = []
        # 目标中必须拆除的现有跳线
        for (t, b) in sorted(E0 - G):
            if not reach(opt - {(t, b)}, src=src, dst=dst):
                gates.append({
                    "type": "cable", "fromJack": t, "toJack": b,
                    "fromPort": env.jack_port.get(t), "toPort": env.jack_port.get(b),
                    "reason": f"该跳线是路由 {idx + 1} 在仅使用当前空闲插孔搭临时跳线时的"
                              f"唯一通路，但目标配置中必须拆除",
                })
        # 当前生效、目标中被插头断开的常态
        cur_snap = _Snap(env, E0)
        for p in env.pairs:
            pid = p["id"]
            if cur_snap.active[pid] and not goal_active.get(pid, False):
                override = dict(cur_snap.active)
                override[pid] = False
                if not reach(opt, active_override=override, src=src, dst=dst):
                    t, b = f"{pid}T", f"{pid}B"
                    gates.append({
                        "type": "normal", "pair": pid,
                        "mode": p["mode"], "modeLabel": MODE_LABELS[p["mode"]],
                        "fromJack": t, "toJack": b,
                        "fromPort": env.jack_port.get(t), "toPort": env.jack_port.get(b),
                        "reason": f"第 {pid} 对{MODE_LABELS[p['mode']]}常态是路由 {idx + 1} "
                                  f"在当前空闲插孔条件下的必经边，而目标跳线的插头会将其断开",
                    })
        if not gates:
            gates.append({
                "type": "ordering",
                "reason": "不存在单一第一道失声点：阻塞来自多步换线的插孔占用顺序锁死"
                          "（可用空闲插孔不足以同时绕开全部必经边）",
            })
        blockers.append({"route": idx, "source": src, "target": dst, "gates": gates})

    code = "REWIRE_SEARCH_LIMIT" if capped else "REWIRE_BLOCKED"
    message = ("搜索达到节点上限仍未判定可行顺序" if capped
               else "不存在能保持全部保护路由连续通声的拆线 / 换接 / 补线顺序")
    details = {
        "protected": [{"index": i, "source": s, "target": d}
                      for i, s, d in protected],
        "blockers": blockers,
        "freeJacks": {"T": free_t, "B": free_b},
        "exploredNodes": explored,
    }
    return RewireError(message, code=code, details=details, status=409)


# ---- 主流程 ---------------------------------------------------------------


def rewire_plan(current_config, raw_target, raw_protected):
    # 1) 解析现状配置
    devices, pairs, cur_routes, port_jack = parse_config(current_config)
    env = _Env(devices, pairs, port_jack)
    cur_eps = [(_pid(r["source"]), _pid(r["target"])) for r in cur_routes]

    # 2) 解析目标路由（设备 / 插孔对沿用现状；非法目标一次性整体拒绝）
    if not isinstance(raw_target, dict) or not isinstance(raw_target.get("routes"), list):
        raise RewireError("target 必须是 JSON 对象，形如 {\"routes\": [{source,target}, ...]}")
    target_config = {"devices": current_config.get("devices") or [],
                     "pairs": current_config.get("pairs") or [],
                     "routes": raw_target["routes"]}
    try:
        _, _, tgt_routes, _ = parse_config(target_config)
    except ConfigError as exc:
        raise RewireError(f"目标路由列表非法：{exc}", code="TARGET_INVALID",
                          details=exc.details)
    tgt_eps = [(_pid(r["source"]), _pid(r["target"])) for r in tgt_routes]

    # 3) 校验保护路由序号
    if not isinstance(raw_protected, list):
        raise RewireError("protected 必须是路由序号数组（从 0 起的整数）")
    protected, seen_idx = [], set()
    n_cur, n_tgt = len(cur_eps), len(tgt_eps)
    for x in raw_protected:
        if isinstance(x, bool) or not isinstance(x, int):
            raise RewireError(f"保护路由序号必须是整数，得到 {x!r}")
        if x < 0 or x >= n_cur or x >= n_tgt:
            raise RewireError(
                f"保护路由序号 {x} 越界：现状路由 {n_cur} 条、目标路由 {n_tgt} 条，"
                f"保护序号必须在两者中都存在（0 起）")
        if x in seen_idx:
            continue
        seen_idx.add(x)
        if cur_eps[x] != tgt_eps[x]:
            raise RewireError(
                f"路由 {x + 1} 在现状（{cur_eps[x][0]} → {cur_eps[x][1]}）与目标"
                f"（{tgt_eps[x][0]} → {tgt_eps[x][1]}）中的端点不一致，"
                f"保护路由必须是同一条信号，改接期间不能改其源 / 目标")
        protected.append((x, cur_eps[x][0], cur_eps[x][1]))
    protected.sort(key=lambda z: z[0])

    # 4) 重建现状跳线、规划目标跳线
    cur_result, _ = plan_all(current_config)
    tgt_result, _ = plan_all(target_config)

    E0 = frozenset((c["fromJack"], c["toJack"]) for c in cur_result["cables"])
    G = frozenset((c["fromJack"], c["toJack"]) for c in tgt_result["cables"])

    # 保护路由在现状必须已通声
    snap0 = _Snap(env, E0)
    baseline_bad = []
    for idx, src, dst in protected:
        if not snap0.audible(src, dst):
            baseline_bad.append(idx)
    if baseline_bad:
        conflicts = [c for c in cur_result["conflicts"] if c["route"] in baseline_bad]
        raise RewireError(
            "保护路由在当前配置下本身就不通声，无法做无中断改线："
            + "、".join(f"路由 {i + 1}" for i in baseline_bad),
            code="PROTECTED_BASELINE",
            details={"routes": baseline_bad, "conflicts": conflicts})

    # 目标必须整体可解（不产生部分方案）
    if not tgt_result["ok"]:
        raise RewireError(
            f"目标路由列表有 {len(tgt_result['conflicts'])} 条不可解，"
            "拒绝生成改线方案（请先修正目标路由）",
            code="TARGET_UNROUTABLE",
            details={"conflicts": tgt_result["conflicts"],
                     "cables": tgt_result["cables"], "pairs": tgt_result["pairs"]},
            status=409)

    old_id = {e: c["id"] for c in cur_result["cables"]
              for e in [(c["fromJack"], c["toJack"])]}
    goal_id = {e: c["id"] for c in tgt_result["cables"]
               for e in [(c["fromJack"], c["toJack"])]}

    # 5) A* 搜索
    start = frozenset(E0)
    best = {start: (0, 0, None, None)}        # state -> (g, temp_births, parent, macro)
    pq = [(_heuristic(start, G), 0, 0, 0, start)]
    counter = 0
    goal_node = None
    explored = 0
    capped = False

    while pq:
        f, tb, g, _, state = heappop(pq)
        if best[state][0] != g or best[state][1] != tb:
            continue
        explored += 1
        if explored > NODE_CAP:
            capped = True
            break
        if state == G:
            goal_node = state
            break
        for macro in _generators(set(state), G, env):
            atoms, _bad = _expand_macro(env, set(state), macro, protected)
            if atoms is None:
                continue
            nxt_edges = set(state)
            if macro[0] == "remove":
                nxt_edges.discard(macro[1])
            elif macro[0] == "add":
                nxt_edges.add(macro[1])
            else:
                nxt_edges.discard(macro[1])
                nxt_edges.add(macro[2])
            nxt = frozenset(nxt_edges)
            ng = g + 2                          # 每个宏操作 = 2 次插拔
            # 临时跳线计数：落到目标之外的边算临时占用
            nedge = macro[1] if macro[0] == "add" else (macro[2] if macro[0] != "remove" else None)
            ntb = tb + (1 if nedge is not None and nedge not in G else 0)
            prev = best.get(nxt)
            if prev is None or (ng, ntb) < (prev[0], prev[1]):
                best[nxt] = (ng, ntb, state, macro, atoms)
                counter += 1
                heappush(pq, (ng + _heuristic(nxt, G), ntb, ng, counter, nxt))

    if goal_node is None:
        goal_snap = _Snap(env, G)
        raise _diagnose(env, set(E0), set(G), goal_snap, protected, explored, capped)

    # 6) 回溯宏操作序列
    macros = []
    node = goal_node
    while best[node][2] is not None:
        _, _, parent, macro = best[node][:4]
        atoms = best[node][4]
        macros.append((macro, atoms))
        node = parent
    macros.reverse()

    return _replay(env, E0, G, macros, protected, cur_eps, tgt_eps,
                   old_id, goal_id, cur_result, tgt_result)



# ---- 回放与方案输出 -------------------------------------------------------


def _chain_sig(chain):
    if chain is None:
        return None
    return tuple((x["type"], x.get("cableId"), x.get("label"))
                 for x in chain if x["kind"] == "edge")


def _display_id(rec):
    if rec["born"] == "old":
        return f"#{rec['oldId']}"
    if rec.get("targetId") is not None:
        return f"#{rec['targetId']}"
    return f"临时{rec['tempNo']}"


def _chain_from_items(env, items, labels):
    chain = []
    for kind, payload in items:
        if kind == "port":
            chain.append({"kind": "port", "port": payload,
                          "jack": env.port_jack.get(payload)})
        elif payload[0] == "device":
            chain.append({"kind": "edge", "type": "device",
                          "label": f"{payload[1]} 内部"})
        elif payload[0] == "normal":
            p = payload[1]
            chain.append({"kind": "edge", "type": "normal",
                          "label": f"{MODE_LABELS[p['mode']]} · 第{p['id']}对"})
        else:
            t, b = payload[1]
            chain.append({"kind": "edge", "type": "cable",
                          "cableId": labels.get((t, b)),
                          "label": f"跳线 {labels.get((t, b), '?')}"})
    return chain


def _all_signatures(snap, cur_eps, tgt_eps):
    """每个序号路由（现状端点 / 目标端点）在快照上的路径签名对。"""
    out = {}
    for i in range(max(len(cur_eps), len(tgt_eps))):
        c = snap.signature(*cur_eps[i]) if i < len(cur_eps) else None
        t = snap.signature(*tgt_eps[i]) if i < len(tgt_eps) else None
        out[i] = (c, t)
    return out


def _apply_atom(edges, hang_t, hang_b, action, jack):
    """在单遍仿真上施加一个原子插拔，返回 (edges, hang_t, hang_b)。"""
    edges = set(edges)
    hang_t, hang_b = set(hang_t), set(hang_b)
    if action == "plug":
        hang = hang_t if jack.endswith("T") else hang_b
        other = hang_b if jack.endswith("T") else hang_t
        if other:                          # 与悬空端合体，跳线完成
            h = next(iter(other))
            t, b = (jack, h) if jack.endswith("T") else (h, jack)
            edges.add((t, b))
            other.clear()
        else:                              # 新增跳线的第一插，单端悬空
            hang.add(jack)
    else:                                  # unplug
        found = next((e for e in edges if jack in e), None)
        if found:                          # 拔下完成跳线的一端，另一端悬空
            edges.discard(found)
            t, b = found
            (hang_b if jack == t else hang_t).add(b if jack == t else t)
        else:                              # 悬空端收回（拆线第二拔）
            (hang_t if jack.endswith("T") else hang_b).discard(jack)
    return edges, hang_t, hang_b


def _replay(env, E0, G, macros, protected, cur_eps, tgt_eps,
            old_id, goal_id, cur_result, tgt_result):
    tokens, edge_token = {}, {}
    for e in E0:
        tid = f"old-{old_id[e]}"
        tokens[tid] = {"born": "old", "oldId": old_id[e], "segments": [],
                       "birthStep": 0, "endStep": None}
        edge_token[e] = tid

    live_edges, hang_t, hang_b = set(E0), set(), set()
    base_snap = _Snap(env, live_edges)
    prev_sigs = _all_signatures(base_snap, cur_eps, tgt_eps)
    prev_pchains = {}
    for idx, src, dst in protected:
        found = base_snap.find(src, dst)
        labels = {e: f"#{old_id[e]}" for e in E0}
        prev_pchains[idx] = _chain_from_items(env, found[1], labels) if found else None

    steps_out, temp_counter, pool = [], 0, []

    for step_i, (macro, atoms) in enumerate(macros, 1):
        kind = macro[0]
        temp_here = set()

        # 宏操作的跳线谱系登记（在宏开始时确定本宏主跳线）。
        # pool = 已拆下、随时可重新插用的物理跳线（兼容跳线优先复用，
        # 池中无可用跳线时才真正新增一根）。
        if kind == "add":
            e = macro[1]
            if pool:
                tid = pool.pop()
                rec = tokens[tid]
                rec["endStep"] = None
                rec["segments"].append(
                    {"step": step_i, "edge": e, "temporary": e not in G})
            elif e in G:
                tid = f"new-{goal_id[e]}"
                rec = tokens[tid] = {"born": "new", "targetId": goal_id[e],
                                     "birthStep": step_i, "endStep": None,
                                     "segments": [{"step": step_i, "edge": e,
                                                   "temporary": False}]}
            else:
                temp_counter += 1
                tid = f"temp-{temp_counter}"
                rec = tokens[tid] = {"born": "new", "tempNo": temp_counter,
                                     "birthStep": step_i, "endStep": None,
                                     "segments": [{"step": step_i, "edge": e,
                                                   "temporary": True}]}
                temp_here.add(tid)
            edge_token[e] = tid
            type_label = "补线"
        elif kind == "remove":
            e = macro[1]
            tid = edge_token.pop(e)
            rec = tokens[tid]
            rec["endStep"] = step_i
            pool.append(tid)
            if rec["born"] == "new" and rec.get("targetId") is None:
                temp_here.add(tid)
            type_label = "拆线"
        else:
            e_old, e_new = macro[1], macro[2]
            tid = edge_token.pop(e_old)
            edge_token[e_new] = tid
            rec = tokens[tid]
            rec["segments"].append(
                {"step": step_i, "edge": e_new, "temporary": e_new not in G})
            if e_new not in G and rec["born"] == "new" and rec.get("targetId") is None:
                temp_here.add(tid)
            type_label = "换接"

        jack_changes = [{"action": a, "actionLabel": "插入" if a == "plug" else "拔下",
                         "jack": j, "port": env.jack_port.get(j)}
                        for a, j, _ in atoms]
        checkpoints, affected = [], set()

        # 逐原子插拔：保护路由每个中间快照都必须可通声（搜索阶段已保证）
        for ai, (a, j, _p) in enumerate(atoms, 1):
            live_edges, hang_t, hang_b = _apply_atom(live_edges, hang_t, hang_b, a, j)
            cp = _Snap(env, live_edges, hang_t, hang_b)
            cp_prot = {}
            for idx, src, dst in protected:
                cp_prot[str(idx)] = cp.audible(src, dst)
            checkpoints.append({"seq": ai, "after": f"{'插入' if a == 'plug' else '拔下'} {j}",
                                "jack": j, "action": a, "protected": cp_prot})
            cp_sigs = _all_signatures(cp, cur_eps, tgt_eps)
            for i in cp_sigs:
                if cp_sigs[i] != prev_sigs[i]:
                    affected.add(i)

        assert not hang_t and not hang_b, "宏操作结束时不应有悬空插头"
        end_snap = _Snap(env, live_edges)
        end_sigs = _all_signatures(end_snap, cur_eps, tgt_eps)
        for i in end_sigs:
            if end_sigs[i] != prev_sigs[i]:
                affected.add(i)
        prev_sigs = end_sigs

        labels_now = {e: _display_id(tokens[t]) for e, t in edge_token.items()}
        pstatus = []
        for idx, src, dst in protected:
            found = end_snap.find(src, dst)
            chain = _chain_from_items(env, found[1], labels_now) if found else None
            pstatus.append({
                "index": idx, "audible": bool(found),
                "chainChanged": _chain_sig(chain) != _chain_sig(prev_pchains[idx]),
                "chain": chain,
            })
            prev_pchains[idx] = chain

        desc = _cable_descriptor(kind, macro, tokens, tid, goal_id, G)
        steps_out.append({
            "seq": step_i, "type": kind, "typeLabel": type_label,
            "cable": desc,
            "jackChanges": jack_changes,
            "affectedRoutes": sorted(affected),
            "protectedStatus": pstatus,
            "checkpoints": checkpoints,
            "temporary": bool(temp_here),
            "temporaryCables": sorted({_display_id(tokens[t]) for t in temp_here}),
            "note": _step_note(desc, atoms, temp_here, tokens),
        })

    return _build_payload(env, E0, G, tokens, edge_token, steps_out,
                          protected, cur_eps, tgt_eps, old_id, goal_id,
                          cur_result, tgt_result)


def _cable_descriptor(kind, macro, tokens, tid, goal_id, G):
    rec = tokens[tid]
    if kind == "add":
        e = macro[1]
        if rec["born"] == "old":
            return {"kind": "reuse-replug",
                    "label": (f"复用拆下的旧跳线 #{rec['oldId']}"
                              + (f" 接至目标跳线 #{goal_id[e]}" if e in G else " 临时改接")),
                    "oldId": rec["oldId"],
                    "targetId": goal_id.get(e),
                    "fromJack": e[0], "toJack": e[1]}
        if e in G:
            return {"kind": "new", "label": f"新增永久跳线 #{goal_id[e]}",
                    "fromJack": e[0], "toJack": e[1]}
        return {"kind": "temporary",
                "label": f"临时跳线 临时{rec['tempNo']}（用完即拆）",
                "fromJack": e[0], "toJack": e[1], "tempNo": rec["tempNo"]}
    if kind == "remove":
        e = macro[1]
        if rec["born"] == "old":
            return {"kind": "removed-old", "label": f"拆除旧跳线 #{rec['oldId']}",
                    "fromJack": e[0], "toJack": e[1]}
        return {"kind": "removed-temp",
                "label": f"收回临时跳线 临时{rec['tempNo']}",
                "fromJack": e[0], "toJack": e[1], "tempNo": rec["tempNo"]}
    e_old, e_new = macro[1], macro[2]
    if rec["born"] == "old" and e_new in G:
        return {"kind": "reuse",
                "label": f"复用旧跳线 #{rec['oldId']}"
                         + (f"（目标编号 #{goal_id[e_new]}）"
                            if goal_id[e_new] != rec["oldId"] else ""),
                "oldId": rec["oldId"], "targetId": goal_id[e_new],
                "oldFromJack": e_old[0], "oldToJack": e_old[1],
                "fromJack": e_new[0], "toJack": e_new[1]}
    if rec.get("targetId") and e_new in G:
        return {"kind": "perm-move",
                "label": f"永久跳线 #{rec['targetId']} 改接到位",
                "fromJack": e_new[0], "toJack": e_new[1]}
    return {"kind": "temp-move",
            "label": f"临时跳线 临时{rec['tempNo']} 改接",
            "fromJack": e_new[0], "toJack": e_new[1],
            "tempNo": rec.get("tempNo")}


def _step_note(desc, atoms, temp_here, tokens):
    changes = "、".join(f"{'插入' if a == 'plug' else '拔下'} {j}"
                       for a, j, _ in atoms)
    tail = "；涉及临时跳线" if temp_here else ""
    return f"{desc['label']}：{changes}{tail}"


def _build_payload(env, E0, G, tokens, edge_token, steps_out,
                   protected, cur_eps, tgt_eps, old_id, goal_id,
                   cur_result, tgt_result):
    # 终结分类：
    #   旧跳线活到最终状态 = reused（原位/改接/拆下后重插都算复用兼容跳线）；
    #   旧跳线结束时仍在拆下池 = removed；
    #   新跳线活到最终目标边 = added（永久补线）；
    #   新跳线中途被收回 = 临时跳线（单独列出）。
    for rec in tokens.values():
        if rec["born"] == "new":
            rec["final"] = "permanent" if rec["endStep"] is None else "temporary"

    removed, added, temps = [], [], []
    reused_ids = {tokens[t]["oldId"] for t in edge_token.values()
                  if tokens[t]["born"] == "old"}
    for rec in tokens.values():
        if rec["born"] == "old":
            if rec["oldId"] in reused_ids:
                continue
            removed.append({"oldId": rec["oldId"], "step": rec["endStep"]})
        elif rec["final"] == "permanent":
            added.append({"targetId": rec["targetId"], "step": rec["birthStep"]})
        else:
            temps.append({
                "id": f"临时{rec['tempNo']}", "tempNo": rec["tempNo"],
                "birthStep": rec["birthStep"], "endStep": rec["endStep"],
                "placements": [{"step": s["step"], "fromJack": s["edge"][0],
                                "toJack": s["edge"][1],
                                "fromPort": env.jack_port.get(s["edge"][0]),
                                "toPort": env.jack_port.get(s["edge"][1])}
                               for s in rec["segments"]],
            })
    reused = []
    for e, tid in edge_token.items():
        rec = tokens[tid]
        if rec["born"] == "old":
            reused.append({"oldId": rec["oldId"], "targetId": goal_id.get(e),
                           "moved": e not in E0,
                           "replugged": len(rec["segments"]) > 0,
                           "placements": [{"step": s["step"], "fromJack": s["edge"][0],
                                           "toJack": s["edge"][1]} for s in rec["segments"]]})
    removed.sort(key=lambda z: z["oldId"])
    reused.sort(key=lambda z: z["oldId"])
    added.sort(key=lambda z: z["targetId"])
    temps.sort(key=lambda z: z["tempNo"])

    final_labels = {e: f"#{goal_id[e]}" for e in G}
    snap_final = _Snap(env, G)
    route_status = []
    for i, (src, dst) in enumerate(tgt_eps):
        found = snap_final.find(src, dst)
        route_status.append({
            "index": i, "source": src, "target": dst, "audible": bool(found),
            "protected": i in {p[0] for p in protected},
            "chain": _chain_from_items(env, found[1], final_labels) if found else None,
        })

    snap0 = _Snap(env, E0)
    init_labels = {e: f"#{old_id[e]}" for e in E0}
    protected_init = []
    for idx, src, dst in protected:
        found = snap0.find(src, dst)
        protected_init.append({"index": idx, "source": src, "target": dst,
                               "audible": bool(found),
                               "chain": _chain_from_items(env, found[1], init_labels)
                               if found else None})

    metrics = {
        "steps": len(steps_out),
        "plugsMoved": 2 * len(steps_out),
        "disconnect": sum(1 for s in steps_out if s["type"] == "remove"),
        "move": sum(1 for s in steps_out if s["type"] in ("move-t", "move-b")),
        "add": sum(1 for s in steps_out if s["type"] == "add"),
        "temporaryCables": len(temps),
    }

    return {
        "ok": True,
        "protectedRoutes": [{"index": i, "source": s, "target": d}
                            for i, s, d in protected],
        "initialProtected": protected_init,
        "currentSummary": {
            "cableCount": cur_result["cableCount"],
            "cables": cur_result["cables"],
            "routes": [{"index": r["index"], "source": r["source"],
                        "target": r["target"], "status": r["status"],
                        "cables": r.get("cables", [])} for r in cur_result["routes"]],
            "conflictIndices": [c["route"] for c in cur_result["conflicts"]],
            "pairs": cur_result["pairs"],
        },
        "targetSummary": {
            "cableCount": tgt_result["cableCount"],
            "cables": tgt_result["cables"],
            "routes": [{"index": r["index"], "source": r["source"],
                        "target": r["target"], "status": r["status"],
                        "cables": r.get("cables", [])} for r in tgt_result["routes"]],
            "pairs": tgt_result["pairs"],
        },
        "steps": steps_out,
        "temporaryCables": temps,
        "reuse": {"reused": len(reused), "removed": len(removed),
                  "added": len(added), "temporary": len(temps),
                  "details": {"reused": reused, "removed": removed, "added": added}},
        "metrics": metrics,
        "routeStatus": route_status,
        "feedbackLoops": tgt_result["feedbackLoops"],
        "baselineOk": cur_result["ok"],
        "targetOk": True,
    }
