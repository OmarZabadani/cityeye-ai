"""
Traffic Logic Engine (Core Decision System)

Purpose:
This module receives vehicle detections (e.g., from YOLO),
assigns them into lanes, and simulates traffic movement
using FIFO + priority-based rules.

Pipeline:
YOLO Output → Lane Assignment → Queue Management → Priority Decision → Action Output
"""

import time
from collections import deque


# =========================================================
# Vehicle Model
# =========================================================
class Vehicle:
    def __init__(self, v_type, confidence=1.0, lane=None):
        self.type = v_type
        self.confidence = confidence
        self.lane = lane
        self.arrival_time = time.time()
        self.waiting_time = 0

    def update_waiting_time(self):
        self.waiting_time = time.time() - self.arrival_time


# =========================================================
# Lane System
# =========================================================
class Lane:
    def __init__(self, name):
        self.name = name
        self.queue = deque()

    def add_vehicle(self, vehicle: Vehicle):
        vehicle.lane = self.name
        self.queue.append(vehicle)

    def remove_vehicle(self):
        if self.queue:
            return self.queue.popleft()
        return None

    def update_waiting_times(self):
        for v in self.queue:
            v.update_waiting_time()

    def is_empty(self):
        return len(self.queue) == 0


# =========================================================
# Traffic Logic Engine
# =========================================================
class TrafficEngine:
    def __init__(self):
        self.lanes = {
            "A": Lane("A"),
            "B": Lane("B"),
            "C": Lane("C"),
            "D": Lane("D"),
        }

        self.priority_map = {
            "bus": 3,
            "truck": 3,
            "car": 2,
            "motorcycle": 1
        }

    # -----------------------------------------
    # Convert YOLO detections → vehicles
    # -----------------------------------------
    def ingest_detections(self, detections, lane="A"):
        """
        detections format:
        [
            {"type": "car", "confidence": 0.8},
            {"type": "bus", "confidence": 0.9}
        ]
        """

        for det in detections:
            vehicle = Vehicle(
                v_type=det["type"],
                confidence=det.get("confidence", 1.0)
            )

            self.lanes[lane].add_vehicle(vehicle)

    # -----------------------------------------
    # Priority calculation
    # -----------------------------------------
    def compute_priority(self, vehicle: Vehicle):
        base = self.priority_map.get(vehicle.type, 1)

        # waiting time increases priority over time
        return base + (vehicle.waiting_time * 0.1)

    # -----------------------------------------
    # Update system state
    # -----------------------------------------
    def update(self):
        for lane in self.lanes.values():
            lane.update_waiting_times()

    # -----------------------------------------
    # Select next vehicle globally
    # -----------------------------------------
    def select_next_vehicle(self):
        best_vehicle = None
        best_lane = None
        best_score = -1

        for lane in self.lanes.values():
            for vehicle in lane.queue:
                score = self.compute_priority(vehicle)

                if score > best_score:
                    best_score = score
                    best_vehicle = vehicle
                    best_lane = lane

        return best_lane, best_vehicle

    # -----------------------------------------
    # Execute one traffic step
    # -----------------------------------------
    def step(self):
        self.update()

        lane, vehicle = self.select_next_vehicle()

        if lane and vehicle:
            lane.queue.remove(vehicle)

            return {
                "action": "pass",
                "lane": lane.name,
                "vehicle": vehicle.type,
                "waiting_time": round(vehicle.waiting_time, 2)
            }

        return {
            "action": "idle",
            "lane": None,
            "vehicle": None
        }

    # -----------------------------------------
    # System snapshot (for UI/debugging)
    # -----------------------------------------
    def get_state(self):
        state = {}

        for name, lane in self.lanes.items():
            state[name] = [
                {
                    "type": v.type,
                    "waiting_time": round(v.waiting_time, 2)
                }
                for v in lane.queue
            ]

        return state