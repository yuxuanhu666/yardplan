"""
堆场可用空间管理器

封装为 YardSpace 类, 支持:
  - 从 nameIndex.json + spaceCurrent.json 初始化
  - 动态更新: place_container() / remove_container()
  - 查询可投放位: get_placeable_20ft() / get_placeable_40ft()
  - 打印报告: print_summary() / print_block() / print_stack()

槽位编址: blockId.bayIdx.stackIdx.tierIdx.slotSize
  - slotSize=1 → 20尺位 (奇数bayId)
  - slotSize=2 → 40尺位 (偶数bayId), 通过 relatedIndex 关联两个20尺位
  - containerSize: 1=20ft, 2=40ft, 3=45ft (40/45尺均占 slotSize=2)
"""

import json
import os
import time
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "data")


class YardSpace:
    """
    堆场可用空间管理器。

    核心数据结构
    -----------
    stacks : dict
        {(blockId, bayIdx, stackIdx) -> {
            "max_tier": int,
            "top_occupied_tier": int,
            "next_placeable_tier": int | None,
            "tiers": {tierIdx -> {
                "slot_20ft": fullSlotName,
                "slot_40ft": fullSlotName,
                "free_20ft": bool,
                "free_40ft": bool,   # 自身空 + 关联20尺均空
                "occupant": {...} | None
            }}
        }}

    使用示例
    --------
        yard = YardSpace.load()
        yard.print_summary()
        positions = yard.get_placeable_20ft(block_id="B03")
        yard.place_container("Y-B03.005.A.03", "TEMU1234567", container_size=1)
        yard.remove_container("Y-B03.005.A.03")
    """

    def __init__(self):
        # ---- 静态拓扑 ----
        self.slots_20ft = {}   # fullSlotName -> {blockId, bayIdx, stackIdx, tierIdx}
        self.slots_40ft = {}   # fullSlotName -> {..., related_20ft: [name, name]}
        self.twenty_to_forty = defaultdict(list)  # 20ft name -> [40ft names that reference it]

        # ---- 动态状态 ----
        self.occupied_20ft = set()
        self.occupied_40ft = set()
        self.slot_occupant = {}  # fullSlotName -> {"cid": str, "size": "20ft"|"40ft"|"45ft"}
        self.stacks = {}         # (blockId, bayIdx, stackIdx) -> stack_info
        self.cnt_by_size = {"20ft": 0, "40ft": 0, "45ft": 0}

    # ================================================================
    #  初始化
    # ================================================================

    @classmethod
    def load(cls, name_index_path=None, space_current_path=None):
        """
        从文件创建 YardSpace。

        Parameters
        ----------
        name_index_path  : 默认 DATA_DIR/nameIndex.json
        space_current_path : 默认 DATA_DIR/spaceCurrent.json, 传 None 则初始化空堆场
        """
        yard = cls()
        yard._load_topology(name_index_path or os.path.join(DATA_DIR, "nameIndex.json"))
        if space_current_path is not False:
            yard._load_containers(space_current_path or os.path.join(DATA_DIR, "spaceCurrent.json"))
        yard._rebuild_all_stacks()
        return yard

    def _load_topology(self, path):
        print("正在加载 nameIndex.json (解析槽位拓扑) ...")
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        print(f"  JSON 解析耗时 {time.time() - t0:.1f}s")

        yard_map = raw.get("yardNameIndexMap", {})
        del raw

        for full_name, info in yard_map.items():
            tier_idx = info.get("tierIdx", -1)
            if tier_idx <= 0:
                continue
            slot_size = info.get("slotSize", -1)
            if slot_size == 1:
                self.slots_20ft[full_name] = {
                    "blockId": info["blockId"],
                    "bayIdx":  info["bayIdx"],
                    "stackIdx": info["stackIdx"],
                    "tierIdx": tier_idx,
                }
            elif slot_size == 2:
                related = [
                    r["fullSlotName"]
                    for r in info.get("relatedIndex", [])
                    if r.get("slotSize") == 1
                ]
                self.slots_40ft[full_name] = {
                    "blockId": info["blockId"],
                    "bayIdx":  info["bayIdx"],
                    "stackIdx": info["stackIdx"],
                    "tierIdx": tier_idx,
                    "related_20ft": related,
                }
                for r_name in related:
                    self.twenty_to_forty[r_name].append(full_name)
        del yard_map
        print(f"  20尺槽位: {len(self.slots_20ft):,}  |  40尺槽位: {len(self.slots_40ft):,}")

    def _load_containers(self, path):
        print("正在加载 spaceCurrent.json (解析箱子占位) ...")
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        print(f"  JSON 解析耗时 {time.time() - t0:.1f}s")

        container_map = raw.get("containerMap", {})
        del raw
        print(f"  当前箱子数量: {len(container_map):,}")

        for _slot_name, cinfo in container_map.items():
            loc = cinfo["currentSlotLocation"]
            full_name = loc["fullSlotName"]
            cid = cinfo.get("containerId", _slot_name)
            slot_size = loc.get("slotSize", cinfo.get("containerSize", 1))
            container_size = cinfo.get("containerSize", slot_size)
            size_label = {1: "20ft", 2: "40ft", 3: "45ft"}.get(container_size, "20ft")

            self.cnt_by_size[size_label] += 1
            occ_info = {"cid": cid, "size": size_label}

            if slot_size == 1:
                self.occupied_20ft.add(full_name)
                self.slot_occupant[full_name] = occ_info
            elif slot_size == 2:
                self.occupied_40ft.add(full_name)
                self.slot_occupant[full_name] = occ_info
                for r in loc.get("relatedIndex", []):
                    r_name = r.get("fullSlotName")
                    if r_name and r.get("slotSize") == 1:
                        self.occupied_20ft.add(r_name)
                        self.slot_occupant[r_name] = occ_info

    # ================================================================
    #  Stack 构建 / 刷新
    # ================================================================

    def _rebuild_all_stacks(self):
        """全量构建 stack 数据 (仅初始化时调用一次)。"""
        stacks = defaultdict(lambda: {"max_tier": 0, "tiers": {}})

        for name, info in self.slots_20ft.items():
            key = (info["blockId"], info["bayIdx"], info["stackIdx"])
            tier = info["tierIdx"]
            stacks[key]["max_tier"] = max(stacks[key]["max_tier"], tier)
            tiers = stacks[key]["tiers"]
            if tier not in tiers:
                tiers[tier] = {}
            tiers[tier]["slot_20ft"] = name
            tiers[tier]["free_20ft"] = name not in self.occupied_20ft
            tiers[tier]["occupant"] = self.slot_occupant.get(name)

        for name, info in self.slots_40ft.items():
            key = (info["blockId"], info["bayIdx"], info["stackIdx"])
            tier = info["tierIdx"]
            stacks[key]["max_tier"] = max(stacks[key]["max_tier"], tier)
            tiers = stacks[key]["tiers"]
            if tier not in tiers:
                tiers[tier] = {}
            tiers[tier]["slot_40ft"] = name
            tiers[tier]["free_40ft"] = self._is_40ft_free(name)

        self.stacks = dict(stacks)
        for key in self.stacks:
            self._refresh_stack_top(key)

    def _is_40ft_free(self, slot_40_name):
        """40尺位可用 = 自身未被占 + 关联的两个20尺位均未被占。"""
        if slot_40_name in self.occupied_40ft:
            return False
        for r in self.slots_40ft[slot_40_name].get("related_20ft", []):
            if r in self.occupied_20ft:
                return False
        return True

    def _refresh_stack_top(self, stack_key):
        """刷新一个 stack 列的 top_occupied_tier / next_placeable_tier。"""
        sinfo = self.stacks[stack_key]
        top = 0
        for t in range(1, sinfo["max_tier"] + 1):
            td = sinfo["tiers"].get(t, {})
            slot_20 = td.get("slot_20ft")
            is_occupied = (slot_20 is not None and slot_20 in self.occupied_20ft)
            if not is_occupied:
                slot_40 = td.get("slot_40ft")
                if slot_40 is not None and slot_40 in self.occupied_40ft:
                    is_occupied = True
            if is_occupied:
                top = t
            else:
                break
        sinfo["top_occupied_tier"] = top
        next_t = top + 1
        sinfo["next_placeable_tier"] = next_t if next_t <= sinfo["max_tier"] else None

    def _refresh_tier_status(self, stack_key, tier_idx):
        """刷新某个 tier 的 free 状态。"""
        td = self.stacks[stack_key]["tiers"].get(tier_idx, {})
        slot_20 = td.get("slot_20ft")
        if slot_20:
            td["free_20ft"] = slot_20 not in self.occupied_20ft
            td["occupant"] = self.slot_occupant.get(slot_20)
        slot_40 = td.get("slot_40ft")
        if slot_40:
            td["free_40ft"] = self._is_40ft_free(slot_40)

    def _stack_key_of_20ft(self, slot_name):
        info = self.slots_20ft.get(slot_name)
        return (info["blockId"], info["bayIdx"], info["stackIdx"]) if info else None

    def _stack_key_of_40ft(self, slot_name):
        info = self.slots_40ft.get(slot_name)
        return (info["blockId"], info["bayIdx"], info["stackIdx"]) if info else None

    # ================================================================
    #  动态更新: 放箱 / 提箱
    # ================================================================

    def place_container(self, full_slot_name, container_id, container_size):
        """
        在指定槽位放入一个箱子, 自动更新所有受影响的状态。

        Parameters
        ----------
        full_slot_name : str  如 "Y-B03.005.A.03" (20尺) 或 "Y-B03.006.A.03" (40尺)
        container_id   : str  如 "TEMU1234567"
        container_size : int  1=20ft, 2=40ft, 3=45ft
        """
        size_label = {1: "20ft", 2: "40ft", 3: "45ft"}.get(container_size, "20ft")
        occ_info = {"cid": container_id, "size": size_label}
        affected = set()

        if full_slot_name in self.slots_20ft:
            self.occupied_20ft.add(full_slot_name)
            self.slot_occupant[full_slot_name] = occ_info
            info = self.slots_20ft[full_slot_name]
            sk = (info["blockId"], info["bayIdx"], info["stackIdx"])
            tier = info["tierIdx"]
            affected.add(sk)
            self._refresh_tier_status(sk, tier)
            # 关联的 40ft 槽位可能被阻塞, 刷新它们所在的 stack
            for ft_name in self.twenty_to_forty.get(full_slot_name, []):
                ft_info = self.slots_40ft[ft_name]
                ft_sk = (ft_info["blockId"], ft_info["bayIdx"], ft_info["stackIdx"])
                self._refresh_tier_status(ft_sk, ft_info["tierIdx"])
                affected.add(ft_sk)

        elif full_slot_name in self.slots_40ft:
            self.occupied_40ft.add(full_slot_name)
            self.slot_occupant[full_slot_name] = occ_info
            info = self.slots_40ft[full_slot_name]
            sk = (info["blockId"], info["bayIdx"], info["stackIdx"])
            affected.add(sk)
            self._refresh_tier_status(sk, info["tierIdx"])
            for r_name in info.get("related_20ft", []):
                self.occupied_20ft.add(r_name)
                self.slot_occupant[r_name] = occ_info
                r_info = self.slots_20ft.get(r_name)
                if r_info:
                    r_sk = (r_info["blockId"], r_info["bayIdx"], r_info["stackIdx"])
                    r_tier = r_info["tierIdx"]
                    self._refresh_tier_status(r_sk, r_tier)
                    affected.add(r_sk)
                    # 这些 20ft 被占 → 可能影响其他 40ft 的可用性
                    for ft_name in self.twenty_to_forty.get(r_name, []):
                        if ft_name != full_slot_name:
                            ft_info = self.slots_40ft[ft_name]
                            ft_sk = (ft_info["blockId"], ft_info["bayIdx"], ft_info["stackIdx"])
                            self._refresh_tier_status(ft_sk, ft_info["tierIdx"])
                            affected.add(ft_sk)
        else:
            raise ValueError(f"未知槽位: {full_slot_name}")

        for sk in affected:
            self._refresh_stack_top(sk)
        self.cnt_by_size[size_label] += 1

    def remove_container(self, full_slot_name):
        """
        从指定槽位提走一个箱子, 自动更新所有受影响的状态。

        Parameters
        ----------
        full_slot_name : str  该箱子的主槽位名 (放箱时使用的那个)
        """
        occ = self.slot_occupant.pop(full_slot_name, None)
        if occ is None:
            raise ValueError(f"槽位 {full_slot_name} 没有箱子")

        affected = set()

        if full_slot_name in self.slots_20ft:
            self.occupied_20ft.discard(full_slot_name)
            info = self.slots_20ft[full_slot_name]
            sk = (info["blockId"], info["bayIdx"], info["stackIdx"])
            affected.add(sk)
            self._refresh_tier_status(sk, info["tierIdx"])
            for ft_name in self.twenty_to_forty.get(full_slot_name, []):
                ft_info = self.slots_40ft[ft_name]
                ft_sk = (ft_info["blockId"], ft_info["bayIdx"], ft_info["stackIdx"])
                self._refresh_tier_status(ft_sk, ft_info["tierIdx"])
                affected.add(ft_sk)

        elif full_slot_name in self.slots_40ft:
            self.occupied_40ft.discard(full_slot_name)
            info = self.slots_40ft[full_slot_name]
            sk = (info["blockId"], info["bayIdx"], info["stackIdx"])
            affected.add(sk)
            self._refresh_tier_status(sk, info["tierIdx"])
            for r_name in info.get("related_20ft", []):
                self.occupied_20ft.discard(r_name)
                self.slot_occupant.pop(r_name, None)
                r_info = self.slots_20ft.get(r_name)
                if r_info:
                    r_sk = (r_info["blockId"], r_info["bayIdx"], r_info["stackIdx"])
                    self._refresh_tier_status(r_sk, r_info["tierIdx"])
                    affected.add(r_sk)
                    for ft_name in self.twenty_to_forty.get(r_name, []):
                        if ft_name != full_slot_name:
                            ft_info = self.slots_40ft[ft_name]
                            ft_sk = (ft_info["blockId"], ft_info["bayIdx"], ft_info["stackIdx"])
                            self._refresh_tier_status(ft_sk, ft_info["tierIdx"])
                            affected.add(ft_sk)

        for sk in affected:
            self._refresh_stack_top(sk)
        self.cnt_by_size[occ["size"]] -= 1

    # ================================================================
    #  查询接口
    # ================================================================

    def get_placeable_20ft(self, block_id=None):
        """
        获取所有当前可投放 20 尺位 (每个 stack 列最多 1 个: 栈顶下一层)。

        Parameters
        ----------
        block_id : str, 可选, 只返回指定箱区

        Returns
        -------
        list of dict: [{fullSlotName, blockId, bayIdx, stackIdx, tierIdx}, ...]
        """
        result = []
        for (bid, bay, stk), sinfo in self.stacks.items():
            if block_id and bid != block_id:
                continue
            nt = sinfo["next_placeable_tier"]
            if nt is None:
                continue
            td = sinfo["tiers"].get(nt, {})
            if td.get("free_20ft", False):
                result.append({
                    "fullSlotName": td["slot_20ft"],
                    "blockId": bid, "bayIdx": bay,
                    "stackIdx": stk, "tierIdx": nt,
                })
        return result

    def get_placeable_40ft(self, block_id=None):
        """
        获取所有当前可投放 40/45 尺位。
        除了本列栈顶+1层可用外, 还检查关联的相邻列也有足够支撑。

        Returns
        -------
        list of dict: [{fullSlotName, blockId, bayIdx, stackIdx, tierIdx}, ...]
        """
        result = []
        for (bid, bay, stk), sinfo in self.stacks.items():
            if block_id and bid != block_id:
                continue
            nt = sinfo["next_placeable_tier"]
            if nt is None:
                continue
            td = sinfo["tiers"].get(nt, {})
            if not td.get("free_40ft", False):
                continue
            # 额外检查: 关联的两个20尺列都必须支撑到 nt-1
            slot_40_name = td.get("slot_40ft")
            if not slot_40_name:
                continue
            if nt > 1:
                supported = True
                for r_name in self.slots_40ft[slot_40_name].get("related_20ft", []):
                    r_info = self.slots_20ft.get(r_name)
                    if not r_info:
                        continue
                    r_sk = (r_info["blockId"], r_info["bayIdx"], r_info["stackIdx"])
                    r_stack = self.stacks.get(r_sk)
                    if r_stack and r_stack["top_occupied_tier"] < nt - 1:
                        supported = False
                        break
                if not supported:
                    continue
            result.append({
                "fullSlotName": slot_40_name,
                "blockId": bid, "bayIdx": bay,
                "stackIdx": stk, "tierIdx": nt,
            })
        return result

    def get_block_stats(self):
        """返回每个箱区的统计信息。"""
        stats = defaultdict(lambda: {
            "total_20ft": 0, "occupied_20ft": 0, "free_20ft": 0,
            "total_40ft": 0, "occupied_40ft": 0, "free_40ft": 0, "blocked_40ft": 0,
        })
        for name, info in self.slots_20ft.items():
            bid = info["blockId"]
            stats[bid]["total_20ft"] += 1
            if name in self.occupied_20ft:
                stats[bid]["occupied_20ft"] += 1
            else:
                stats[bid]["free_20ft"] += 1
        for name, info in self.slots_40ft.items():
            bid = info["blockId"]
            stats[bid]["total_40ft"] += 1
            if name in self.occupied_40ft:
                stats[bid]["occupied_40ft"] += 1
            elif self._is_40ft_free(name):
                stats[bid]["free_40ft"] += 1
            else:
                stats[bid]["blocked_40ft"] += 1
        return dict(sorted(stats.items()))

    def get_stack_info(self, block_id, bay_idx, stack_idx):
        """返回指定 stack 列的详细信息。"""
        return self.stacks.get((block_id, bay_idx, stack_idx))

    @property
    def total_containers(self):
        return sum(self.cnt_by_size.values())

    @property
    def total_teu_used(self):
        return len(self.occupied_20ft & set(self.slots_20ft.keys()))

    @property
    def total_teu_free(self):
        return len(set(self.slots_20ft.keys()) - self.occupied_20ft)

    # ================================================================
    #  打印
    # ================================================================

    def print_summary(self):
        """打印堆场可用空间汇总报告。"""
        p20 = self.get_placeable_20ft()
        p40 = self.get_placeable_40ft()
        block_stats = self.get_block_stats()
        teu_total = len(self.slots_20ft)
        teu_used = self.total_teu_used
        teu_free = self.total_teu_free
        occ_40 = len(self.occupied_40ft)
        total_stacks = len(self.stacks)
        occ_stacks = sum(1 for v in self.stacks.values() if v["top_occupied_tier"] > 0)
        full_stacks = sum(1 for v in self.stacks.values() if v["next_placeable_tier"] is None)

        free_40_count = sum(1 for n in self.slots_40ft if self._is_40ft_free(n))
        blocked_40 = len(self.slots_40ft) - occ_40 - free_40_count

        print("\n" + "=" * 76)
        print("                        堆场可用空间报告")
        print("=" * 76)

        print(f"\n  箱子总数: {self.total_containers:,}")
        print(f"    20尺: {self.cnt_by_size['20ft']:,}  |  "
              f"40尺: {self.cnt_by_size['40ft']:,}  |  "
              f"45尺: {self.cnt_by_size['45ft']:,}")

        print(f"\n  {'指标':<16} {'总数':>10} {'已占用':>10} {'空闲':>10} {'利用率':>8}")
        print("  " + "-" * 60)
        r20 = teu_used / teu_total * 100 if teu_total else 0
        print(f"  {'20尺槽位':<14} {teu_total:>10,} {teu_used:>10,} {teu_free:>10,} {r20:>7.1f}%")
        r40 = occ_40 / len(self.slots_40ft) * 100 if self.slots_40ft else 0
        print(f"  {'40尺槽位':<14} {len(self.slots_40ft):>10,} {occ_40:>10,} "
              f"{free_40_count:>10,} {r40:>7.1f}%")
        print(f"  {'40尺被阻塞':<14} {'':>10} {'':>10} {blocked_40:>10,}")

        n40_45 = self.cnt_by_size['40ft'] + self.cnt_by_size['45ft']
        print(f"\n  TEU 容量: {teu_total:,}  |  已用: {teu_used:,} "
              f"(={self.cnt_by_size['20ft']:,}×1+{n40_45:,}×2)  |  空闲: {teu_free:,}")

        print(f"\n  Stack 列: {total_stacks:,}  |  有箱: {occ_stacks:,}  |  已满: {full_stacks:,}")
        print(f"  可投放 20尺: {len(p20):,}  |  可投放 40/45尺: {len(p40):,}")

        print(f"\n  {'箱区':<5} {'20总':>7} {'20占':>7} {'20空':>7} {'20%':>6}"
              f"  {'40总':>7} {'40占':>7} {'40空':>7} {'40阻':>7}")
        print("  " + "-" * 72)
        for bid, bs in block_stats.items():
            u = bs["occupied_20ft"] / bs["total_20ft"] * 100 if bs["total_20ft"] else 0
            print(f"  {bid:<5} {bs['total_20ft']:>7,} {bs['occupied_20ft']:>7,} "
                  f"{bs['free_20ft']:>7,} {u:>5.1f}%"
                  f"  {bs['total_40ft']:>7,} {bs['occupied_40ft']:>7,} "
                  f"{bs['free_40ft']:>7,} {bs['blocked_40ft']:>7,}")
        print("=" * 76)

    def print_block(self, block_id):
        """打印某个箱区内所有有箱子的 stack。"""
        items = sorted(
            [(k, v) for k, v in self.stacks.items()
             if k[0] == block_id and v["top_occupied_tier"] > 0],
            key=lambda x: x[0],
        )
        print(f"\n  箱区 {block_id}: {len(items)} 个有箱 Stack")
        print(f"  {'位置':<20} {'层高':>4} {'占顶':>4} {'下放':>4}  各层状态")
        print("  " + "-" * 72)
        for (bid, bay, stk), sinfo in items:
            self._print_stack_line(bid, bay, stk, sinfo)

    def print_stack(self, block_id, bay_idx, stack_idx):
        """打印单个 stack 列的详细信息。"""
        sinfo = self.stacks.get((block_id, bay_idx, stack_idx))
        if not sinfo:
            print(f"  Stack ({block_id}, {bay_idx}, {stack_idx}) 不存在")
            return
        self._print_stack_line(block_id, bay_idx, stack_idx, sinfo)
        print(f"    层级详情:")
        for t in range(1, sinfo["max_tier"] + 1):
            td = sinfo["tiers"].get(t, {})
            s20 = td.get("slot_20ft", "-")
            s40 = td.get("slot_40ft", "-")
            occ = td.get("occupant")
            f20 = "空" if td.get("free_20ft", True) else "占"
            f40 = "空" if td.get("free_40ft", True) else "占"
            cid = occ["cid"] if occ else "-"
            csz = occ["size"] if occ else "-"
            print(f"      T{t}: 20尺[{f20}] {s20}  |  40尺[{f40}] {s40}  |  箱号: {cid} ({csz})")

    def _print_stack_line(self, block_id, bay_idx, stack_idx, sinfo):
        pos = f"{block_id}.{bay_idx}.{stack_idx}"
        parts = []
        for t in range(1, sinfo["max_tier"] + 1):
            td = sinfo["tiers"].get(t, {})
            occ = td.get("occupant")
            if occ:
                parts.append(f"T{t}:{occ['size'][:2]}")
            elif td.get("slot_40ft") and td["slot_40ft"] in self.occupied_40ft:
                parts.append(f"T{t}:40")
            else:
                parts.append(f"T{t}: -")
        nt = sinfo["next_placeable_tier"]
        ns = str(nt) if nt else "满"
        print(f"  {pos:<20} {sinfo['max_tier']:>4} {sinfo['top_occupied_tier']:>4} "
              f"{ns:>4}  {' | '.join(parts)}")


# ================================================================
#  演示
# ================================================================

if __name__ == "__main__":
    t_start = time.time()

    yard = YardSpace.load()
    yard.print_summary()

    print(f"\n--- 示例: 查询 B01 的可投放 20 尺位 (前5个) ---")
    for p in yard.get_placeable_20ft("B01")[:5]:
        print(f"  {p['fullSlotName']}  block={p['blockId']} bay={p['bayIdx']} "
              f"stack={p['stackIdx']} tier={p['tierIdx']}")

    print(f"\n--- 示例: 查看一个 Stack 详情 ---")
    sample = None
    for k, v in yard.stacks.items():
        if v["top_occupied_tier"] > 0 and v["next_placeable_tier"] is not None:
            sample = k
            break
    if sample:
        yard.print_stack(*sample)

    print(f"\n--- 示例: 动态放箱 / 提箱 ---")
    positions = yard.get_placeable_20ft("B05")
    if positions:
        slot = positions[0]
        name = slot["fullSlotName"]
        print(f"  放箱前 B05 可投放 20尺: {len(positions)}")
        yard.place_container(name, "TEST1234567", container_size=1)
        positions_after = yard.get_placeable_20ft("B05")
        print(f"  放箱后 B05 可投放 20尺: {len(positions_after)}  (放在了 {name})")
        yard.remove_container(name)
        positions_restore = yard.get_placeable_20ft("B05")
        print(f"  提箱后 B05 可投放 20尺: {len(positions_restore)}  (恢复)")

    print(f"\n总耗时: {time.time() - t_start:.1f}s")
