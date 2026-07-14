# Agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.widgets import Button, RadioButtons, Slider

# ============================================================
# Japanese font (optional)
# ============================================================
JP_FONT_CANDIDATES = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
]


def setup_japanese_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


_JP_FONT_USED = setup_japanese_font()
if _JP_FONT_USED is None:
    print("[Warning] Japanese font was not found. Some labels may be garbled.")


# ============================================================
# Train geometry and constants
# ============================================================
CAR_W = 2.95
CAR_H = 20.0
MAX_AGENTS = 300

# 4ドア位置（左右で合計8ドア）
DOOR_Y_POSITIONS = np.array([3.0, 7.0, 13.0, 17.0], dtype=np.float32)
DOORS = np.array(
    [(0.0, y) for y in DOOR_Y_POSITIONS] + [(CAR_W, y) for y in DOOR_Y_POSITIONS],
    dtype=np.float32,
)

DOOR_WIDTH = 1.3
DOOR_HALF = DOOR_WIDTH / 2.0
DOOR_ZONE_HALF = DOOR_HALF
END_ZONE_Y = CAR_H * 0.2

HEATMAP_BINS_X = 10
HEATMAP_BINS_Y = 30

# 座席配置（左右の座席ゾーン）
ZONE_WIDTH = 0.55
SEAT_Y_RANGES = [
    (0.7, 2.3),
    (3.7, 6.3),
    (7.7, 12.3),
    (13.7, 16.3),
    (17.7, 19.3),
]

# 状態
STATE_STANDING = 0
STATE_BOARDING = 1
STATE_EXITING = 2
STATE_SEATED = 3
STATE_TO_SEAT = 4
STATE_YIELDING = 5  # ドア周辺で横に避けるなどの一時行動

STATE_COLORS = {
    STATE_STANDING: "#66aaff",
    STATE_BOARDING: "#42f58d",
    STATE_EXITING: "#ff6666",
    STATE_SEATED: "#ffffff",
    STATE_TO_SEAT: "#ffd166",
    STATE_YIELDING: "#ffd700",
}

# デフォルトルート（route.json が無ければこれを使用）
DEFAULT_ROUTE = {
    "loop": True,
    "stations": [
        {"name": "Station A", "door_side": "right", "door_open": 10, "travel_time": 18},
        {"name": "Station B", "door_side": "left", "door_open": 10, "travel_time": 18},
        {"name": "Station C", "door_side": "right", "door_open": 11, "travel_time": 20},
    ],
}


# ============================================================
# 補助関数（モジュールレベル）
# ============================================================
def point_overlaps_seat(seat: dict, x: float, y: float, min_dist: float = 0.18) -> bool:
    """座席中心からの距離で重なり判定。min_dist は安全距離（m）。"""
    dx = x - seat["x"]
    dy = y - seat["y"]
    return dx * dx + dy * dy < (min_dist * min_dist)


def clamp_point(x: float, y: float) -> tuple[float, float]:
    """車体内に厳格にクリップするマージン付きクリップ関数"""
    return float(np.clip(x, 0.08, CAR_W - 0.08)), float(np.clip(y, 0.08, CAR_H - 0.08))


# ============================================================
# シミュレーション本体
# ============================================================
class TrainCrowdingSim:
    def __init__(
        self,
        seconds: int = 60,
        fps: int = 10,
        seed: int = 12,
        initial_agents: int = 80,
        route_path: str = "route.json",
    ) -> None:
        self.seconds = seconds
        self.fps = fps
        self.seed = seed
        self.initial_agents = initial_agents
        self.rng = np.random.default_rng(seed)

        # 座席ゾーン（左右）
        self.seat_zones: list[dict] = []
        for y0, y1 in SEAT_Y_RANGES:
            self.seat_zones.append({"x_min": 0.0, "x_max": ZONE_WIDTH, "y_min": y0, "y_max": y1})
            self.seat_zones.append(
                {"x_min": CAR_W - ZONE_WIDTH, "x_max": CAR_W, "y_min": y0, "y_max": y1}
            )

        # 座席（矩形の中心座標で管理）
        self.seats: list[dict] = []
        for zone in self.seat_zones:
            cx = (zone["x_min"] + zone["x_max"]) / 2.0
            yy = zone["y_min"] + 0.25
            while yy <= zone["y_max"] - 0.2:
                self.seats.append({"x": cx, "y": yy, "occupied_by": -1, "zone": zone})
                yy += 0.55

        self.route_path = route_path
        self.load_route(route_path)

        # override 用（None のときは route の値を使う）
        self.door_duration_override: int | None = None

        # 行動更新のためのタイマー（0.2〜0.5秒ごとに局所判断）
        self.behavior_update_interval = max(1, int(round(0.3 * self.fps)))  # 0.3秒程度

        self.reset()

    def load_route(self, route_path: str) -> None:
        path = Path(route_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = DEFAULT_ROUTE
            print(f"[Info] {route_path} was not found. Built-in route is used.")
        self.loop = bool(data.get("loop", True))
        self.route = data["stations"]
        if not self.route:
            raise ValueError("route.json must contain at least one station.")

    def reset(self) -> None:
        self.x = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.y = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.tx = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.ty = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.desired_y = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.state = np.zeros(MAX_AGENTS, dtype=np.int8)
        self.target_seat_idx = np.full(MAX_AGENTS, -1, dtype=np.int16)
        self.exit_door_x = np.zeros(MAX_AGENTS, dtype=np.float32)
        self.exit_door_y = np.zeros(MAX_AGENTS, dtype=np.float32)

        self.count = 0
        self.frame = 0
        self.started_frame = 0
        self.station_index = 0
        self.schedule_state = "dwell"
        self.pending_next_index = None
        self.travel_until = 0
        self.door_open_until = -1
        self.door_open_started = 0
        self.current_dwell_duration = self.seconds_to_frames(10)
        self.spawn_queue: list[tuple[np.ndarray, int, str]] = []

        self.hist_door = []
        self.hist_center = []
        self.hist_end = []
        self.hist_total = []
        self.hist_boarding = []
        self.hist_exiting = []

        for seat in self.seats:
            seat["occupied_by"] = -1

        # 初期駅の扉を開ける（spawn_queue を作る）
        self._open_station_doors(self.route[self.station_index])
        # 初期乗客を追加。ただし add_random は壁側や座席領域に人を置かないようにする
        self.add_random(self.initial_agents)

    @property
    def max_frames(self) -> int:
        return self.seconds * self.fps

    @property
    def elapsed_seconds(self) -> float:
        return (self.frame - self.started_frame) / self.fps

    @property
    def finished(self) -> bool:
        return self.frame - self.started_frame >= self.max_frames

    @property
    def current_station_name(self) -> str:
        return self.route[self.station_index]["name"]

    def mark_started(self) -> None:
        self.started_frame = self.frame

    def seconds_to_frames(self, value: float | int) -> int:
        return max(1, int(round(float(value) * self.fps)))

    # 立ち位置レーンを中央寄せにして、壁側（椅子と壁の間）を避ける
    def standing_lanes(self) -> list[float]:
        center = CAR_W / 2.0
        return [center - 0.35, center + 0.35]

    def standing_slots(self) -> list[tuple[float, float]]:
        lanes = self.standing_lanes()
        y_step = 0.55
        slots: list[tuple[float, float]] = []
        for lane_idx, lane_x in enumerate(lanes):
            offset = (lane_idx % 2) * y_step * 0.5
            yy = 0.45 + offset
            while yy <= CAR_H - 0.45:
                # スロットが座席ゾーンの内部に入らないようにチェック
                in_seat_zone = False
                for zone in self.seat_zones:
                    if zone["y_min"] - 0.25 <= yy <= zone["y_max"] + 0.25 and zone["x_min"] - 0.25 <= lane_x <= zone["x_max"] + 0.25:
                        in_seat_zone = True
                        break
                if not in_seat_zone:
                    slots.append((lane_x, yy))
                yy += y_step
        return slots

    def is_in_seat_zone(self, x: float, y: float) -> dict | None:
        for zone in self.seat_zones:
            if zone["x_min"] <= x <= zone["x_max"] and zone["y_min"] <= y <= zone["y_max"]:
                return zone
        return None

    def door_penalty(self, y: float) -> float:
        penalty = 0.0
        for door_y in DOOR_Y_POSITIONS:
            d = abs(y - door_y)
            if d < DOOR_ZONE_HALF:
                penalty += (DOOR_ZONE_HALF - d) * 3.0
        return penalty

    def section_occupancy(self) -> tuple[int, int, int]:
        if self.count == 0:
            return 0, 0, 0
        y = self.y[: self.count]
        low = int((y < CAR_H * 0.33).sum())
        mid = int(((y >= CAR_H * 0.33) & (y <= CAR_H * 0.66)).sum())
        high = int((y > CAR_H * 0.66).sum())
        return low, mid, high

    def choose_flow_target_y(self, door_y: float) -> float:
        low, mid, high = self.section_occupancy()
        section_counts = np.array([low, mid, high], dtype=np.float32)
        emptiest = int(np.argmin(section_counts))
        if self.rng.random() < 0.55:
            if emptiest == 0:
                return float(self.rng.uniform(0.8, 6.0))
            if emptiest == 1:
                return float(self.rng.uniform(7.0, 13.0))
            return float(self.rng.uniform(14.0, 19.2))
        direction = -1.0 if door_y > CAR_H / 2.0 else 1.0
        if self.rng.random() < 0.35:
            direction *= -1.0
        return float(np.clip(door_y + direction * self.rng.uniform(2.0, 6.0), 0.8, CAR_H - 0.8))

    def choose_standing_slot(self, i: int, reserved: set[int] | None = None) -> tuple[float, float]:
        if reserved is None:
            reserved = set()
        slots = self.standing_slots()
        if not slots:
            return float(CAR_W / 2.0), float(np.clip(self.y[i], 0.4, CAR_H - 0.4))
        best_idx = 0
        best_score = 1e18
        for idx, (sx, sy) in enumerate(slots):
            if idx in reserved:
                continue
            dist = math.hypot(sx - self.x[i], sy - self.y[i])
            same_target = math.hypot(sx - self.tx[i], sy - self.ty[i])
            desired = abs(sy - self.desired_y[i]) if self.desired_y[i] > 0 else 0.0
            density = 0.0
            for j in range(self.count):
                if i == j or self.state[j] == STATE_EXITING:
                    continue
                dd = (sx - self.x[j]) ** 2 + (sy - self.y[j]) ** 2
                if dd < 0.55 * 0.55:
                    density += 1.0
            score = dist + density * 1.8 + same_target * 0.25
            score += self.door_penalty(sy) * (0.75 if self.count > 140 else 1.25)
            if self.state[i] == STATE_BOARDING:
                score += desired * 0.9
            else:
                score += desired * 0.25
            if score < best_score:
                best_score = score
                best_idx = idx
        reserved.add(best_idx)
        return slots[best_idx]

    # ---------------------------
    # 初期配置を「列を作る」方針に変更
    # ---------------------------
    def add_random(self, n: int) -> None:
        """
        初期配置を列（door queue）を意識して作成する。
        - 最初は各開いている扉（右側4つ）に列を作る（右側優先）。
        - 列が満たされたら車内の立ち位置スロットへ配置。
        - すべての配置で最小距離を守る（重なりゼロを目指す）。
        """
        n = min(n, MAX_AGENTS - self.count)
        if n <= 0:
            return

        MIN_AGENT_DIST = 0.32  # エージェント間の最小距離（m）
        placed = 0

        # 優先: 右側ドア（車外→車内の列を作る）
        # まず右側の4ドア（DOORS indices 4..7）を使って列を作る
        right_door_indices = [i for i, d in enumerate(DOORS) if d[0] == CAR_W]
        # 並ぶ列の深さ（車内へ入る方向に沿って）
        per_door_queue_depth = 6  # 1列あたり最大6人程度を列にする
        # 列のx位置（車内に入ったときの最初の列位置）
        entry_x_right = CAR_W - 0.78
        entry_x_left = 0.78

        # 右側ドアにまず割り当て
        for idx in right_door_indices:
            if placed >= n:
                break
            door = DOORS[idx]
            door_y = float(door[1])
            # 列のy座標はドア中心から上下に並べる（ホーム側で左右に並ぶイメージ）
            for q in range(per_door_queue_depth):
                if placed >= n:
                    break
                # 車外に少し離れた位置にスポーンして、entry_x に向かう
                outside_x = CAR_W + 0.45
                # 列はドア中心から縦に並ぶ（手前から奥へ）
                offset = (q - (per_door_queue_depth - 1) / 2.0) * 0.28
                sx = outside_x + self.rng.normal(0.0, 0.01)
                sy = float(np.clip(door_y + offset, 0.12, CAR_H - 0.12))
                i = self.count
                self.x[i] = sx
                self.y[i] = sy
                self.tx[i] = entry_x_right
                # 目的地は車内奥の空きへ向かうように少し奥側を指定
                self.ty[i] = float(np.clip(door_y + self.rng.normal(0.0, 0.6), 0.12, CAR_H - 0.12))
                self.desired_y[i] = self.ty[i]
                self.state[i] = STATE_BOARDING
                self.target_seat_idx[i] = -1
                self.count += 1
                placed += 1

        # 次に左側ドア（必要なら）
        left_door_indices = [i for i, d in enumerate(DOORS) if d[0] == 0.0]
        for idx in left_door_indices:
            if placed >= n:
                break
            door = DOORS[idx]
            door_y = float(door[1])
            for q in range(per_door_queue_depth):
                if placed >= n:
                    break
                outside_x = -0.45
                offset = (q - (per_door_queue_depth - 1) / 2.0) * 0.28
                sx = outside_x + self.rng.normal(0.0, 0.01)
                sy = float(np.clip(door_y + offset, 0.12, CAR_H - 0.12))
                i = self.count
                self.x[i] = sx
                self.y[i] = sy
                self.tx[i] = entry_x_left
                self.ty[i] = float(np.clip(door_y + self.rng.normal(0.0, 0.6), 0.12, CAR_H - 0.12))
                self.desired_y[i] = self.ty[i]
                self.state[i] = STATE_BOARDING
                self.target_seat_idx[i] = -1
                self.count += 1
                placed += 1

        # 列で埋めきれなかった分は車内の立ち位置スロットへ配置（重なりを避ける）
        remaining = n - placed
        if remaining > 0:
            slots = self.standing_slots()
            self.rng.shuffle(slots)
            slot_idx = 0
            for _ in range(remaining):
                if slot_idx < len(slots):
                    sx, sy = slots[slot_idx]
                    slot_idx += 1
                else:
                    sx = CAR_W * 0.5 + self.rng.normal(0.0, 0.12)
                    sy = self.rng.uniform(0.6, CAR_H - 0.6)
                sx = float(np.clip(sx, 0.08, CAR_W - 0.08))
                sy = float(np.clip(sy, 0.08, CAR_H - 0.08))
                # 他の人と最小距離を保つ
                ok = True
                for j in range(self.count):
                    ddx = sx - self.x[j]
                    ddy = sy - self.y[j]
                    if ddx * ddx + ddy * ddy < (MIN_AGENT_DIST * MIN_AGENT_DIST):
                        ok = False
                        break
                if not ok:
                    # 少しずらして再試行
                    attempts = 0
                    while attempts < 30 and not ok:
                        attempts += 1
                        sx = float(np.clip(sx + self.rng.normal(0.0, 0.06), 0.08, CAR_W - 0.08))
                        sy = float(np.clip(sy + self.rng.normal(0.0, 0.06), 0.08, CAR_H - 0.08))
                        ok = True
                        for j in range(self.count):
                            ddx = sx - self.x[j]
                            ddy = sy - self.y[j]
                            if ddx * ddx + ddy * ddy < (MIN_AGENT_DIST * MIN_AGENT_DIST):
                                ok = False
                                break
                    if not ok:
                        continue
                i = self.count
                self.x[i] = sx + self.rng.normal(0.0, 0.005)
                self.y[i] = sy + self.rng.normal(0.0, 0.005)
                self.tx[i] = self.x[i]
                self.ty[i] = self.y[i]
                self.desired_y[i] = self.y[i]
                self.state[i] = STATE_STANDING
                self.target_seat_idx[i] = -1
                self.count += 1

        # 初期配置後に空席があれば必ず埋める
        self.find_seats_for_standing(force=True)

    # ---------------------------
    # 既存ロジック（移動・衝突・着席割当など）
    # 主要関数は前回の改善版を踏襲し、局所判断を導入
    # ---------------------------

    def add_at_door(self, door: np.ndarray, n: int) -> None:
        """扉からのみ乗車。外側スポーンは扉の真横、車内へ入る際は最小距離を保つ。"""
        n = min(n, MAX_AGENTS - self.count)
        if n <= 0:
            return

        MIN_AGENT_DIST = 0.32
        start = self.count
        end = start + n
        side = -1.0 if door[0] == 0.0 else 1.0
        outside_x = -0.45 if side < 0 else CAR_W + 0.45
        entry_x = 0.78 if side < 0 else CAR_W - 0.78

        for k, i in enumerate(range(start, end)):
            queue_offset = (k - (n - 1) / 2.0) * 0.22
            sx = outside_x + self.rng.normal(0.0, 0.01)
            sy = float(np.clip(door[1] + queue_offset, 0.12, CAR_H - 0.12))

            # 外側にスポーンしてから entry_x に向かう
            self.x[i] = sx
            self.y[i] = sy
            self.tx[i] = entry_x
            self.ty[i] = float(np.clip(self.choose_flow_target_y(float(door[1])), 0.12, CAR_H - 0.12))
            self.desired_y[i] = self.ty[i]
            self.state[i] = STATE_BOARDING
            self.target_seat_idx[i] = -1

            # entry_x に入るときに既存の人と重ならないように少しオフセットを与える
            for attempt in range(6):
                conflict = False
                for j in range(self.count):
                    ddx = (entry_x) - self.x[j]
                    ddy = self.ty[i] - self.y[j]
                    if ddx * ddx + ddy * ddy < (MIN_AGENT_DIST * MIN_AGENT_DIST):
                        conflict = True
                        break
                if not conflict:
                    break
                self.ty[i] = float(np.clip(self.ty[i] + self.rng.normal(0.0, 0.12), 0.12, CAR_H - 0.12))

        self.count = end

    def move(self, speed: float, door_open: bool) -> None:
        """移動処理。エージェント間の重なりを厳密に防止する。
        さらに局所判断に基づく横ずれや隙間探索を行う。
        """
        n = self.count
        if n == 0:
            return

        MIN_AGENT_DIST = 0.32

        for i in range(n):
            # travel 中は STATE_TO_SEAT のみ移動許可
            if self.schedule_state == "travel" and self.state[i] != STATE_TO_SEAT:
                continue

            if self.state[i] == STATE_SEATED:
                continue

            # 移動速度の決定（状態依存）
            if self.state[i] == STATE_EXITING:
                actual_speed = speed * (1.35 if door_open else 0.85)
            elif self.state[i] == STATE_BOARDING:
                actual_speed = speed * (1.05 if door_open else 0.20)
            elif self.state[i] == STATE_TO_SEAT:
                actual_speed = speed * 0.80
            elif self.state[i] == STATE_YIELDING:
                actual_speed = speed * 0.45
            else:
                actual_speed = speed * (0.35 if door_open else 0.10)

            # 局所視界（前方中心120度, range 1.6m）
            view_range = 1.6
            view_angle = math.radians(120)

            # 目的方向ベクトル
            dx = float(self.tx[i] - self.x[i])
            dy = float(self.ty[i] - self.y[i])
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            vx = dx / dist
            vy = dy / dist

            # 候補生成：前方・斜め左右・横・停止
            step = min(actual_speed, dist)
            candidates = [
                (self.x[i] + vx * step, self.y[i] + vy * step),  # まっすぐ
                (self.x[i] + (vx * 0.78 - vy * 0.30) * step, self.y[i] + (vy * 0.78 + vx * 0.30) * step),  # 斜め左
                (self.x[i] + (vx * 0.78 + vy * 0.30) * step, self.y[i] + (vy * 0.78 - vx * 0.30) * step),  # 斜め右
                (self.x[i] + (-vy) * (step * 0.5), self.y[i] + (vx) * (step * 0.5)),  # 横左
                (self.x[i] + (vy) * (step * 0.5), self.y[i] + (-vx) * (step * 0.5)),  # 横右
                (self.x[i], self.y[i]),  # 待機
            ]

            best_x = float(self.x[i])
            best_y = float(self.y[i])
            best_score = 1e18

            # 局所密度・圧力を計算（波の伝播のための簡易圧力）
            local_pressure = 0.0
            for j in range(n):
                if i == j:
                    continue
                dd = math.hypot(float(self.x[j] - self.x[i]), float(self.y[j] - self.y[i]))
                if dd < 1e-6:
                    continue
                if dd < 1.2:
                    local_pressure += max(0.0, 1.2 - dd)

            for nx, ny in candidates:
                nx = float(np.clip(nx, 0.08, CAR_W - 0.08))
                ny = float(np.clip(ny, 0.08, CAR_H - 0.08))

                # 座席ゾーンの内部に無理に入らない（座席へ向かう人は除く）
                zone = self.is_in_seat_zone(nx, ny)
                seat_idx = int(self.target_seat_idx[i])
                target_zone = self.seats[seat_idx]["zone"] if 0 <= seat_idx < len(self.seats) else None
                if zone is not None and zone != target_zone and self.state[i] != STATE_EXITING:
                    continue

                # 座席と重ならない（座席が occupied か否かに関わらず近づけない）
                overlap_seat = False
                for seat in self.seats:
                    if point_overlaps_seat(seat, nx, ny, min_dist=0.40):
                        if not (self.target_seat_idx[i] >= 0 and self.seats[self.target_seat_idx[i]] is seat):
                            overlap_seat = True
                            break
                if overlap_seat:
                    continue

                # 他人との最小距離を厳密にチェック（前方優先だが、前が詰まれば横へ）
                blocked = False
                crowd = 0.0
                front_blocking = False
                for j in range(n):
                    if i == j:
                        continue
                    ddx = nx - float(self.x[j])
                    ddy = ny - float(self.y[j])
                    dd = ddx * ddx + ddy * ddy
                    if dd < (MIN_AGENT_DIST * MIN_AGENT_DIST):
                        blocked = True
                        break
                    if dd < 0.72 * 0.72:
                        crowd += 1.0
                    # 前方ブロック判定（自分の進行方向に近い相手）
                    relx = float(self.x[j] - self.x[i])
                    rely = float(self.y[j] - self.y[i])
                    if relx * vx + rely * vy > 0 and (relx * relx + rely * rely) < (0.9 * 0.9):
                        front_blocking = True

                if blocked:
                    continue

                # スコア計算：目的地距離 + 混雑 + レーン引力 - 前方が空いているか
                goal = math.hypot(float(self.tx[i] - nx), float(self.ty[i] - ny))
                lane_pull = min(abs(nx - lane) for lane in self.standing_lanes())
                door_hold = self.door_penalty(ny) if self.state[i] == STATE_STANDING else 0.0

                # 局所圧力が高いときは横移動を優先（波の伝播）
                pressure_penalty = local_pressure * 0.6

                # 前方が塞がれている場合は横移動を好む
                front_bonus = -0.8 if front_blocking and (nx, ny) != (self.x[i], self.y[i]) else 0.0

                score = goal + crowd * 0.28 + lane_pull * 0.35 + door_hold * 0.25 + pressure_penalty + front_bonus

                # 少しランダム性を入れて自然に
                score += self.rng.normal(0.0, 0.02)

                if score < best_score:
                    best_score = score
                    best_x = nx
                    best_y = ny

            # 最終的に車体内にクリップして代入
            self.x[i] = float(np.clip(best_x, 0.08, CAR_W - 0.08))
            self.y[i] = float(np.clip(best_y, 0.08, CAR_H - 0.08))

    def collide(self, collision_distance: float) -> None:
        """衝突処理をより厳密にして、エージェント同士の重なりを許さない。
        さらに波のような圧力伝播を簡易的に実装する。
        """
        n = self.count
        if n < 2:
            return

        MIN_AGENT_DIST = 0.32
        px = np.zeros(n, dtype=np.float32)
        py = np.zeros(n, dtype=np.float32)

        # pairwise push: j > i
        for i in range(n):
            if self.state[i] == STATE_SEATED:
                continue
            for j in range(i + 1, n):
                if self.state[j] == STATE_SEATED:
                    continue
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                dist = math.hypot(float(dx), float(dy)) + 1e-9
                if dist < MIN_AGENT_DIST:
                    # 両者を等しく押し離す
                    overlap = MIN_AGENT_DIST - dist
                    ux = dx / dist
                    uy = dy / dist
                    push = overlap * 0.5
                    px[i] += ux * push
                    py[i] += uy * push
                    px[j] -= ux * push
                    py[j] -= uy * push

        # 圧力伝播（近傍の移動が周囲に伝わるように小さな追加ベクトルを与える）
        pressure_influence = 0.06
        for i in range(n):
            if self.state[i] == STATE_SEATED:
                continue
            # 周囲の押し出しを少し拡散させる
            neighbors = 0
            sx = 0.0
            sy = 0.0
            for j in range(n):
                if i == j:
                    continue
                dd = math.hypot(float(self.x[j] - self.x[i]), float(self.y[j] - self.y[i]))
                if dd < 1.0:
                    neighbors += 1
                    sx += (self.x[i] - self.x[j])
                    sy += (self.y[i] - self.y[j])
            if neighbors > 0:
                px[i] += (sx / neighbors) * pressure_influence
                py[i] += (sy / neighbors) * pressure_influence

        # 適用（座席の上に押し込まないようにクリップ）
        for i in range(n):
            if self.state[i] == STATE_SEATED:
                continue
            new_x = float(self.x[i] + px[i])
            new_y = float(self.y[i] + py[i])
            # 車体内にクリップ
            new_x = float(np.clip(new_x, 0.08, CAR_W - 0.08))
            new_y = float(np.clip(new_y, 0.08, CAR_H - 0.08))
            # 座席中心に近づきすぎないようにする
            too_close_to_seat = False
            for seat in self.seats:
                if point_overlaps_seat(seat, new_x, new_y, min_dist=0.40):
                    too_close_to_seat = True
                    break
            if too_close_to_seat:
                # 元の位置に戻す（押し込まない）
                continue
            self.x[i] = new_x
            self.y[i] = new_y

    def find_seats_for_standing(self, force: bool = False) -> None:
        """空席があれば必ず最も近い人を着席させる。座席周辺は常に安全距離を保つ。"""
        if self.count == 0:
            return

        MIN_AGENT_DIST = 0.32
        candidate_states = {STATE_STANDING, STATE_BOARDING}
        reserved_agents = set(int(seat["occupied_by"]) for seat in self.seats if int(seat["occupied_by"]) >= 0)

        for seat_idx, seat in enumerate(self.seats):
            if seat["occupied_by"] != -1:
                continue
            best_i = -1
            best_score = 1e18
            for i in range(self.count):
                if i in reserved_agents or self.state[i] not in candidate_states:
                    continue
                # 座席に近い人を優先（距離）
                dist = math.hypot(seat["x"] - self.x[i], seat["y"] - self.y[i])
                if not force and dist > 3.2:
                    continue
                # 座席に座ったときに他人と重ならないか（座席周辺の安全距離）
                conflict = False
                for j in range(self.count):
                    if i == j:
                        continue
                    dd = (seat["x"] - self.x[j]) ** 2 + (seat["y"] - self.y[j]) ** 2
                    if dd < (MIN_AGENT_DIST * MIN_AGENT_DIST) and j != i:
                        conflict = True
                        break
                if conflict:
                    continue
                score = dist
                if score < best_score:
                    best_score = score
                    best_i = i
            if best_i == -1:
                continue
            # 着席確定：座席に移動させ、座席周辺に他者が入らないようにする
            seat["occupied_by"] = best_i
            reserved_agents.add(best_i)
            self.target_seat_idx[best_i] = seat_idx
            self.tx[best_i] = seat["x"]
            self.ty[best_i] = seat["y"]
            self.state[best_i] = STATE_TO_SEAT
            # 着席が確定したら、その座席周辺に他者が入らないように
            for j in range(self.count):
                if j == best_i:
                    continue
                ddx = self.x[j] - seat["x"]
                ddy = self.y[j] - seat["y"]
                if ddx * ddx + ddy * ddy < (MIN_AGENT_DIST * MIN_AGENT_DIST):
                    # 近すぎる人は少し押し出す（安全距離を確保）
                    angle = math.atan2(ddy, ddx)
                    self.x[j] = float(np.clip(seat["x"] + math.cos(angle) * MIN_AGENT_DIST, 0.08, CAR_W - 0.08))
                    self.y[j] = float(np.clip(seat["y"] + math.sin(angle) * MIN_AGENT_DIST, 0.08, CAR_H - 0.08))

    def release_seat(self, idx: int) -> None:
        seat_idx = int(self.target_seat_idx[idx])
        if 0 <= seat_idx < len(self.seats) and self.seats[seat_idx]["occupied_by"] == idx:
            self.seats[seat_idx]["occupied_by"] = -1
        for seat in self.seats:
            if seat["occupied_by"] == idx:
                seat["occupied_by"] = -1
        self.target_seat_idx[idx] = -1

    def start_alighting(self, door: np.ndarray, n: int) -> None:
        if self.count == 0 or n <= 0:
            return
        candidates = []
        for i in range(self.count):
            if self.state[i] == STATE_EXITING:
                continue
            dist = math.hypot(float(door[0] - self.x[i]), float(door[1] - self.y[i]))
            seated_penalty = 0.7 if self.state[i] == STATE_SEATED else 0.0
            to_seat_penalty = 0.4 if self.state[i] == STATE_TO_SEAT else 0.0
            candidates.append((dist + seated_penalty + to_seat_penalty, i))
        candidates.sort(key=lambda item: item[0])
        for _, i in candidates[:n]:
            self.release_seat(i)
            self.state[i] = STATE_EXITING
            self.target_seat_idx[i] = -1
            self.exit_door_x[i] = float(door[0])
            self.exit_door_y[i] = float(door[1])
            side = -1.0 if door[0] == 0.0 else 1.0
            outside_x = -0.45 if side < 0 else CAR_W + 0.45
            self.tx[i] = outside_x
            self.ty[i] = float(door[1] + self.rng.normal(0.0, 0.12))

    def remove_indices(self, indices: list[int]) -> None:
        if not indices:
            return
        remove_idx = np.array(sorted(set(indices)), dtype=np.int32)
        keep = np.ones(self.count, dtype=bool)
        keep[remove_idx] = False
        old_to_new = np.full(self.count, -1, dtype=np.int32)
        old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        for seat in self.seats:
            occ = int(seat["occupied_by"])
            if occ == -1:
                continue
            seat["occupied_by"] = int(old_to_new[occ]) if old_to_new[occ] >= 0 else -1
        new_count = int(keep.sum())
        self.x[:new_count] = self.x[: self.count][keep]
        self.y[:new_count] = self.y[: self.count][keep]
        self.tx[:new_count] = self.tx[: self.count][keep]
        self.ty[:new_count] = self.ty[: self.count][keep]
        self.desired_y[:new_count] = self.desired_y[: self.count][keep]
        self.state[:new_count] = self.state[: self.count][keep]
        self.target_seat_idx[:new_count] = self.target_seat_idx[: self.count][keep]
        self.exit_door_x[:new_count] = self.exit_door_x[: self.count][keep]
        self.exit_door_y[:new_count] = self.exit_door_y[: self.count][keep]
        self.count = new_count
        self.target_seat_idx[self.count :] = -1
        # 削除後は必ず座席を埋める
        self.find_seats_for_standing(force=True)

    def _open_station_doors(self, station: dict) -> None:
        # door_open を route の値か override で決定
        door_open_value = station.get("door_open", 8)
        if getattr(self, "door_duration_override", None) is not None:
            door_open_value = int(self.door_duration_override)
        self.current_dwell_duration = self.seconds_to_frames(door_open_value)
        self.door_open_started = self.frame
        self.door_open_until = self.frame + self.current_dwell_duration
        side_to_open = station.get("door_side", "right")
        if side_to_open == "right":
            self.active_door_mask = DOORS[:, 0] == CAR_W
        else:
            self.active_door_mask = DOORS[:, 0] == 0.0
        doors_to_use = [door for door in DOORS if self.active_door_mask[np.where((DOORS == door).all(axis=1))[0][0]]]
        # 要求どおり：扉が開くたびに各扉から5人乗車、各扉からランダムで0-5人降車
        board_per_door = 5
        is_terminal = (not self.loop) and (self.station_index == len(self.route) - 1)
        if is_terminal:
            onboard = self.count
            if onboard > 0 and doors_to_use:
                perdoor = math.ceil(onboard / len(doors_to_use))
                for door in doors_to_use:
                    self.spawn_queue.append((door, perdoor, "alight"))
            return
        for door in doors_to_use:
            alight_random = int(self.rng.integers(0, 6))
            if alight_random > 0:
                self.spawn_queue.append((door, alight_random, "alight"))
        for door in doors_to_use:
            self.spawn_queue.append((door, board_per_door, "board"))

    def update_station_schedule(self) -> None:
        if self.schedule_state == "dwell":
            if self.frame >= self.door_open_until:
                next_index = self.station_index + 1
                if next_index >= len(self.route):
                    if self.loop:
                        next_index = 0
                    else:
                        self.schedule_state = "idle"
                        self.door_open_until = self.frame + 1_000_000_000
                        return
                self.pending_next_index = next_index
                self.schedule_state = "travel"
                next_station = self.route[next_index]
                self.travel_until = self.frame + self.seconds_to_frames(next_station.get("travel_time", 0))
        elif self.schedule_state == "travel":
            if self.frame >= self.travel_until:
                self.station_index = int(self.pending_next_index)
                self.schedule_state = "dwell"
                self._open_station_doors(self.route[self.station_index])

    def process_doors(self) -> None:
        if not self.spawn_queue:
            return
        elapsed = max(0, self.frame - self.door_open_started)
        progress = elapsed / max(1, self.current_dwell_duration)
        alight_waiting = any(kind == "alight" and remain > 0 for _, remain, kind in self.spawn_queue)
        per_frame = max(1, math.ceil(sum(remain for _, remain, _ in self.spawn_queue) / max(1, self.current_dwell_duration - elapsed)))
        next_queue: list[tuple[np.ndarray, int, str]] = []
        for door, remain, kind in self.spawn_queue:
            if remain <= 0:
                continue
            if kind == "board" and alight_waiting and progress < 0.35:
                next_queue.append((door, remain, kind))
                continue
            if kind == "alight" and progress > 0.70:
                batch = remain
            else:
                batch = min(remain, per_frame)
            if kind == "board":
                self.add_at_door(door, batch)
            else:
                self.start_alighting(door, batch)
            remain -= batch
            if remain > 0:
                next_queue.append((door, remain, kind))
        self.spawn_queue = next_queue
        # 扉処理後は座席割当を強制して空席を埋める
        self.find_seats_for_standing(force=True)

    def step(self, speed: float, collision_distance: float, behavior: str) -> None:
        # ステップごとに局所行動更新を行うタイミングを入れる
        self.update_station_schedule()
        door_open = (self.schedule_state in ("dwell", "idle") and self.door_open_until >= self.frame)
        if door_open:
            self.process_doors()

        # 局所判断は一定間隔で行う（人の行動を滑らかに）
        if self.frame % max(1, self.behavior_update_interval) == 0:
            self.local_behavior_update(behavior, door_open)

        self.move(speed, door_open)
        self.collide(collision_distance)
        self.remove_exited_agents()
        self.count_zones()
        self.frame += 1

    # ---------------------------
    # 局所行動ロジック（新規）
    # ---------------------------
    def local_behavior_update(self, behavior: str, door_open: bool) -> None:
        """各エージェントが周囲を見て短期的な行動を決める（ドア前のyielding、隙間探索、波の伝播など）"""
        n = self.count
        if n == 0:
            return

        MIN_AGENT_DIST = 0.32
        view_range = 1.6
        for i in range(n):
            # 座っている人は基本的に動かない
            if self.state[i] == STATE_SEATED:
                continue

            # 目的地が出口（降車）かどうか
            # door_open が近い場合、ドア周辺の人は降車を予測して横へ避ける（yielding）
            near_any_open_door = False
            if door_open:
                # active doors mask がある場合はそれを使う
                active_doors = DOORS[getattr(self, "active_door_mask", np.ones(len(DOORS), dtype=bool))]
                for door in active_doors:
                    if math.hypot(float(self.x[i] - door[0]), float(self.y[i] - door[1])) < 1.2:
                        near_any_open_door = True
                        break

            # 降車客がいるかどうか（spawn_queue に alight があるか）
            alight_pending = any(kind == "alight" and remain > 0 for _, remain, kind in self.spawn_queue)

            # ドア前での行動：降車が予想されるときは左右へ少し移動して通路を作る
            if near_any_open_door and alight_pending and self.state[i] in (STATE_STANDING, STATE_BOARDING):
                self.state[i] = STATE_YIELDING
                left_clear = self._is_direction_clear(i, dx=-0.4, dy=0.0, radius=0.6)
                right_clear = self._is_direction_clear(i, dx=0.4, dy=0.0, radius=0.6)
                if left_clear and not right_clear:
                    self.tx[i], self.ty[i] = clamp_point(self.x[i] - 0.35, self.y[i])
                elif right_clear and not left_clear:
                    self.tx[i], self.ty[i] = clamp_point(self.x[i] + 0.35, self.y[i])
                else:
                    if self.rng.random() < 0.5:
                        self.tx[i], self.ty[i] = clamp_point(self.x[i] - 0.25, self.y[i])
                    else:
                        self.tx[i], self.ty[i] = clamp_point(self.x[i] + 0.25, self.y[i])
                continue

            # 乗車客は降車が続く間は待機する（ドアを塞がない）
            if alight_pending and self.state[i] == STATE_BOARDING:
                if door_open:
                    active_doors = DOORS[getattr(self, "active_door_mask", np.ones(len(DOORS), dtype=bool))]
                    dists = [math.hypot(float(self.x[i] - d[0]), float(self.y[i] - d[1])) for d in active_doors]
                    if dists and min(dists) < 1.2:
                        if self.rng.random() < 0.35:
                            if self.rng.random() < 0.5:
                                self.tx[i], self.ty[i] = clamp_point(self.x[i] - 0.25, self.y[i] + self.rng.normal(0.0, 0.05))
                            else:
                                self.tx[i], self.ty[i] = clamp_point(self.x[i] + 0.25, self.y[i] + self.rng.normal(0.0, 0.05))
                        else:
                            self.tx[i], self.ty[i] = self.x[i], self.y[i]
                continue

            # 通常時の行動：目的地（tx,ty）へ向かうが、前が塞がっていれば横へ移動して隙間を探す
            front_blocked = self._is_front_blocked(i, look_ahead=0.9)
            if front_blocked and self.state[i] in (STATE_BOARDING, STATE_STANDING, STATE_EXITING):
                left_clear = self._is_direction_clear(i, dx=-0.35, dy=0.0, radius=0.6)
                right_clear = self._is_direction_clear(i, dx=0.35, dy=0.0, radius=0.6)
                if left_clear and not right_clear:
                    self.tx[i], self.ty[i] = clamp_point(self.x[i] - 0.35, self.y[i])
                elif right_clear and not left_clear:
                    self.tx[i], self.ty[i] = clamp_point(self.x[i] + 0.35, self.y[i])
                elif left_clear and right_clear:
                    if self.rng.random() < 0.5:
                        self.tx[i], self.ty[i] = clamp_point(self.x[i] - 0.25, self.y[i] + 0.15)
                    else:
                        self.tx[i], self.ty[i] = clamp_point(self.x[i] + 0.25, self.y[i] + 0.15)
                else:
                    self.tx[i], self.ty[i] = self.x[i], self.y[i]
                continue

            # 奥の人は基本的に動かない（ただし空間ができたら少し進む）
            if self._is_deep_inside(i) and self.state[i] in (STATE_STANDING, STATE_BOARDING):
                self.tx[i], self.ty[i] = self.x[i], self.y[i]
                continue

            # デフォルト：現在の desired target を維持（既に設定されている tx/ty を使う）
            if self.target_seat_idx[i] >= 0:
                seat = self.seats[int(self.target_seat_idx[i])]
                self.tx[i], self.ty[i] = seat["x"], seat["y"]

    def _is_direction_clear(self, i: int, dx: float, dy: float, radius: float = 0.6) -> bool:
        """指定方向（相対）に半径 radius の空間があるかを判定する簡易関数"""
        tx = float(self.x[i] + dx)
        ty = float(self.y[i] + dy)
        tx, ty = clamp_point(tx, ty)
        for j in range(self.count):
            if i == j:
                continue
            ddx = tx - float(self.x[j])
            ddy = ty - float(self.y[j])
            if ddx * ddx + ddy * ddy < (radius * radius):
                return False
        return True

    def _is_front_blocked(self, i: int, look_ahead: float = 0.9) -> bool:
        """自分の進行方向に look_ahead の距離内で人がいるかを判定"""
        dx = float(self.tx[i] - self.x[i])
        dy = float(self.ty[i] - self.y[i])
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return False
        vx = dx / dist
        vy = dy / dist
        for j in range(self.count):
            if i == j:
                continue
            relx = float(self.x[j] - self.x[i])
            rely = float(self.y[j] - self.y[i])
            proj = relx * vx + rely * vy
            if 0 < proj < look_ahead:
                lateral = abs(relx * (-vy) + rely * vx)
                if lateral < 0.45:
                    return True
        return False

    def _is_deep_inside(self, i: int) -> bool:
        """車内奥側にいるか（入口から離れている）を判定"""
        cx = CAR_W / 2.0
        return abs(self.x[i] - cx) < 0.6 and (self.y[i] > CAR_H * 0.3 and self.y[i] < CAR_H * 0.7)

    def remove_exited_agents(self) -> None:
        """扉経由で外側へ出たエージェントのみ削除する。壁からはみ出しているだけでは削除しない（はみ出し自体を防止）。"""
        remove: list[int] = []
        for i in range(self.count):
            if self.state[i] != STATE_EXITING:
                continue
            # 外側ターゲットに到達して車体外に出たら削除
            if self.x[i] < -0.35 or self.x[i] > CAR_W + 0.35:
                remove.append(i)
        if remove:
            self.remove_indices(remove)

    def count_zones(self) -> None:
        n = self.count
        if n == 0:
            self.hist_door.append(0)
            self.hist_center.append(0)
            self.hist_end.append(0)
            self.hist_total.append(0)
            self.hist_boarding.append(0)
            self.hist_exiting.append(0)
            return
        y = self.y[:n]
        door_mask = np.zeros(n, dtype=bool)
        for door_y in DOOR_Y_POSITIONS:
            door_mask |= np.abs(y - door_y) < DOOR_ZONE_HALF
        end_mask = (~door_mask) & ((y < END_ZONE_Y) | (y > CAR_H - END_ZONE_Y))
        center_mask = ~door_mask & ~end_mask
        self.hist_door.append(int(door_mask.sum()))
        self.hist_center.append(int(center_mask.sum()))
        self.hist_end.append(int(end_mask.sum()))
        self.hist_total.append(n)
        self.hist_boarding.append(int((self.state[:n] == STATE_BOARDING).sum()))
        self.hist_exiting.append(int((self.state[:n] == STATE_EXITING).sum()))

    def heatmap(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((HEATMAP_BINS_Y, HEATMAP_BINS_X), dtype=np.float32)
        inside = (
            (self.x[: self.count] >= 0.0)
            & (self.x[: self.count] <= CAR_W)
            & (self.y[: self.count] >= 0.0)
            & (self.y[: self.count] <= CAR_H)
        )
        xs = self.x[: self.count][inside]
        ys = self.y[: self.count][inside]
        if xs.size == 0:
            return np.zeros((HEATMAP_BINS_Y, HEATMAP_BINS_X), dtype=np.float32)
        heat, _, _ = np.histogram2d(ys, xs, bins=[HEATMAP_BINS_Y, HEATMAP_BINS_X], range=[[0, CAR_H], [0, CAR_W]])
        return heat


# ============================================================
# UI / main
# ============================================================
def build_ui(args: argparse.Namespace) -> None:
    sim = TrainCrowdingSim(seconds=args.seconds, fps=args.fps, seed=args.seed, initial_agents=args.initial_agents, route_path=args.route)
    running = {"value": False, "behavior": args.behavior}

    fig = plt.figure(figsize=(15, 9), facecolor="#1a1a2e")
    fig.suptitle("Train Crowding Agent Simulation", color="white", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, left=0.05, right=0.97, top=0.93, bottom=0.28, width_ratios=[1.05, 0.72, 1.18, 1.18], wspace=0.35, hspace=0.4)

    ax_agent = fig.add_subplot(gs[:, 0])
    ax_agent.set_facecolor("#0d0d1a")
    ax_agent.set_xlim(-0.5, CAR_W + 0.5)
    ax_agent.set_ylim(-1, CAR_H + 1)
    ax_agent.set_aspect("equal")
    ax_agent.set_title("Agent Distribution", color="white", fontsize=10)
    ax_agent.tick_params(colors="gray")
    for spine in ax_agent.spines.values():
        spine.set_edgecolor("#444")

    car_rect = patches.FancyBboxPatch((0, 0), CAR_W, CAR_H, boxstyle="round,pad=0.3", linewidth=2, edgecolor="#88aaff", facecolor="none")
    ax_agent.add_patch(car_rect)

    for door_y in DOOR_Y_POSITIONS:
        ax_agent.axhspan(door_y - DOOR_ZONE_HALF, door_y + DOOR_ZONE_HALF, alpha=0.07, color="orange")
    ax_agent.axhspan(0, END_ZONE_Y, alpha=0.06, color="cyan")
    ax_agent.axhspan(CAR_H - END_ZONE_Y, CAR_H, alpha=0.06, color="cyan")
    ax_agent.scatter(DOORS[:, 0], DOORS[:, 1], marker="s", s=40, color="#ffcc44", zorder=5)

    # 座席矩形パッチ（空=白、着席=赤）
    seat_patches = []
    seat_w = 0.36
    seat_h = 0.36
    for seat in sim.seats:
        rect = patches.Rectangle((seat["x"] - seat_w / 2.0, seat["y"] - seat_h / 2.0), seat_w, seat_h, linewidth=0.6, edgecolor="black", facecolor="white", zorder=4)
        ax_agent.add_patch(rect)
        seat_patches.append(rect)

    scatter = ax_agent.scatter([], [], s=28, edgecolors='k', linewidths=0.2, zorder=6)
    txt_status = ax_agent.text(0.02, 0.98, "", transform=ax_agent.transAxes, color="white", fontsize=8, va="top", bbox=dict(facecolor="#222", alpha=0.7, pad=2))

    ax_heat = fig.add_subplot(gs[0, 1])
    ax_heat.set_facecolor("#0d0d1a")
    ax_heat.set_title("Density Heatmap", color="white", fontsize=10)
    ax_heat.set_xlabel("Left - Right", color="gray", fontsize=8)
    ax_heat.set_ylabel("Front - Back", color="gray", fontsize=8)
    ax_heat.tick_params(colors="gray")
    im_heat = ax_heat.imshow(sim.heatmap(), origin="lower", aspect="auto", extent=[0, CAR_W, 0, CAR_H], cmap="hot", vmin=0, vmax=10)
    cbar = fig.colorbar(im_heat, ax=ax_heat)
    cbar.ax.tick_params(colors="gray")
    cbar.set_label("Count", color="gray", fontsize=8)
    ax_heat.scatter(DOORS[:, 0], DOORS[:, 1], marker="s", s=35, color="#ffcc44", zorder=5)

    ax_graph = fig.add_subplot(gs[:, 2:])
    ax_graph.set_facecolor("#0d0d1a")
    ax_graph.set_title("Zone Occupancy Over Time", color="white", fontsize=10)
    ax_graph.set_xlabel("Frame", color="gray", fontsize=8)
    ax_graph.set_ylabel("Count", color="gray", fontsize=8)
    ax_graph.tick_params(colors="gray")
    for spine in ax_graph.spines.values():
        spine.set_edgecolor("#444")
    line_door, = ax_graph.plot([], [], color="orange", lw=1.5, label="Door area")
    line_center, = ax_graph.plot([], [], color="#88ff88", lw=1.5, label="Center")
    line_end, = ax_graph.plot([], [], color="#44ccff", lw=1.5, label="End")
    line_total, = ax_graph.plot([], [], color="white", lw=1, linestyle="--", label="Total", alpha=0.5)
    ax_graph.legend(fontsize=7, facecolor="#222", labelcolor="white", framealpha=0.7)
    ax_graph.grid(True, color="#333", linewidth=0.5)

    ax_pie = fig.add_subplot(gs[1, 1])
    ax_pie.set_facecolor("#0d0d1a")
    pie_colors = ["orange", "#88ff88", "#44ccff"]
    slider_bg = "#2a2a4a"

    def make_slider(left: float, bottom: float, label: str, vmin: float, vmax: float, vinit: float, step=None) -> Slider:
        ax = fig.add_axes([left, bottom, 0.18, 0.025], facecolor=slider_bg)
        kwargs = dict(valmin=vmin, valmax=vmax, valinit=vinit, color="#5566cc", label=label)
        if step is not None:
            kwargs["valstep"] = step
        slider = Slider(ax, **kwargs)
        slider.label.set_color("white")
        slider.valtext.set_color("white")
        return slider

    sl_speed = make_slider(0.05, 0.10, "Speed", 0.05, 1.5, args.speed)
    sl_col = make_slider(0.05, 0.05, "Collision dist", 0.2, 1.5, args.collision_distance)
    sl_interval = make_slider(0.30, 0.20, "Interval(f)", 10, 120, args.enter_every, 5)
    sl_door_dur = make_slider(0.30, 0.15, "Door open(f)", 2, 30, args.door_open_duration, 1)
    sl_init = make_slider(0.30, 0.10, "Init agents", 0, 300, args.initial_agents, 5)

    # スライダーで door duration を上書きできるようにする
    def _on_door_dur_change(val):
        sim.door_duration_override = int(val)

    sl_door_dur.on_changed(_on_door_dur_change)

    ax_btn_start = fig.add_axes([0.58, 0.17, 0.08, 0.04])
    ax_btn_reset = fig.add_axes([0.58, 0.11, 0.08, 0.04])
    btn_start = Button(ax_btn_start, "Start", color="#223344", hovercolor="#334455")
    btn_reset = Button(ax_btn_reset, "Reset", color="#332222", hovercolor="#443333")
    btn_start.label.set_color("white")
    btn_reset.label.set_color("white")

    ax_radio = fig.add_axes([0.70, 0.04, 0.14, 0.18], facecolor="#1a1a2e")
    radio = RadioButtons(ax_radio, ("Random walk", "Avoid crowding", "Toward exit"), activecolor="#5566cc")
    ax_radio.set_title("Behavior", color="white", fontsize=8, pad=4)
    for label in radio.labels:
        label.set_color("white")
        label.set_fontsize(8)
    behavior_map = {"Random walk": "random", "Avoid crowding": "avoid", "Toward exit": "exit"}

    def update_display() -> None:
        n = sim.count
        if n > 0:
            scatter.set_offsets(np.column_stack([sim.x[:n], sim.y[:n]]))
            colors = [STATE_COLORS.get(int(sim.state[i]), "#66aaff") for i in range(n)]
            scatter.set_facecolor(colors)
        else:
            scatter.set_offsets(np.empty((0, 2)))
        # 座席パッチの色更新（空=白、着席=赤）
        for idx, seat in enumerate(sim.seats):
            occ = int(seat["occupied_by"])
            if occ >= 0:
                seat_patches[idx].set_facecolor("#ff4444")
            else:
                seat_patches[idx].set_facecolor("#ffffff")
        door_open = sim.door_open_until >= sim.frame
        state = "running" if running["value"] else "paused"
        if sim.finished:
            state = "finished"
        txt_status.set_text(f"Time: {sim.elapsed_seconds:05.1f}/{sim.seconds}s | Frame: {sim.frame} | Agents: {n} | Door: {'open' if door_open else 'closed'} | {state} | Station: {sim.current_station_name}")
        heat = sim.heatmap()
        im_heat.set_data(heat)
        im_heat.set_clim(0, max(1, float(heat.max())))
        frames = np.arange(len(sim.hist_total))
        line_door.set_data(frames, sim.hist_door)
        line_center.set_data(frames, sim.hist_center)
        line_end.set_data(frames, sim.hist_end)
        line_total.set_data(frames, sim.hist_total)
        ax_graph.set_xlim(0, max(50, len(sim.hist_total)))
        ax_graph.set_ylim(0, max(10, sim.count + 20))
        door = sim.hist_door[-1] if sim.hist_door else 0
        center = sim.hist_center[-1] if sim.hist_center else 0
        end = sim.hist_end[-1] if sim.hist_end else 0
        ax_pie.clear()
        ax_pie.set_facecolor("#0d0d1a")
        ax_pie.set_title("Current Distribution", color="white", fontsize=10)
        if door + center + end > 0:
            _, _, autotexts = ax_pie.pie([door, center, end], labels=["Door area", "Center", "End"], colors=pie_colors, autopct="%1.0f%%", startangle=90, textprops={"color": "white", "fontsize": 8})
            for text in autotexts:
                text.set_color("black")
                text.set_fontsize(8)

    def on_start(_event) -> None:
        if sim.finished:
            return
        running["value"] = not running["value"]
        if running["value"]:
            sim.mark_started()
        btn_start.label.set_text("Pause" if running["value"] else "Start")
        fig.canvas.draw_idle()

    def on_reset(_event) -> None:
        running["value"] = False
        sim.initial_agents = int(sl_init.val)
        sim.reset()
        btn_start.label.set_text("Start")
        update_display()
        fig.canvas.draw_idle()

    def on_radio(label: str) -> None:
        running["behavior"] = behavior_map[label]

    def timer_callback() -> None:
        if not running["value"]:
            return
        if sim.finished:
            running["value"] = False
            btn_start.label.set_text("Start")
            update_display()
            fig.canvas.draw_idle()
            return
        sim.step(speed=float(sl_speed.val), collision_distance=float(sl_col.val), behavior=running["behavior"])
        update_display()
        fig.canvas.draw_idle()

    btn_start.on_clicked(on_start)
    btn_reset.on_clicked(on_reset)
    radio.on_clicked(on_radio)
    timer = fig.canvas.new_timer(interval=1000 // sim.fps)
    timer.add_callback(timer_callback)
    timer.start()
    update_display()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--initial_agents", type=int, default=80)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--collision_distance", type=float, default=0.6)
    # 間隔を半分にする（デフォルト 50 -> 25）
    parser.add_argument("--enter_every", type=int, default=25)
    parser.add_argument("--door_open_duration", type=int, default=8)
    parser.add_argument("--behavior", type=str, default="random")
    parser.add_argument("--route", type=str, default="route.json")
    args = parser.parse_args()
    build_ui(args)


if __name__ == "__main__":
    main()
