from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sledge.semantic_control.hazard_spec import (
    ActorLayer,
    HazardSemanticSpec,
    InteractionLayer,
    ObjectLayer,
    OcclusionSpec,
    ProtectionLayer,
    RiskLayer,
    RoadLayer,
    StaticObstacleSpec,
    ValidationLayer,
)
from sledge.semantic_control.spec_checker import check_spec
from sledge.semantic_control.spec_presets import normalize_spec


def _normalize_text(text: str) -> str:
    """Normalize English spacing while preserving Chinese keywords."""
    text = text.strip().lower()
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = text.replace("cut-in", "cut in").replace("round-about", "roundabout")
    text = re.sub(r"[，。！？、；：,;:!?()\[\]{}\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _has_any(text: str, keywords: Sequence[str]) -> bool:
    return any(_normalize_text(k) in text for k in keywords)


def _matched_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    return [k for k in keywords if _normalize_text(k) in text]


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slugify(value: str, fallback: str = "nl_spec") -> str:
    words = re.findall(r"[a-z0-9]+", _normalize_text(value))
    if words:
        return "_".join(words[:10])[:80]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{fallback}_{digest}"


def _range_around(value: float, ratio: float = 0.2, minimum_width: float = 0.2) -> Tuple[float, float]:
    delta = max(abs(value) * ratio, minimum_width)
    return (max(0.0, value - delta), value + delta)


def _extract_first_number(text: str, patterns: Sequence[str]) -> Optional[float]:
    for pattern in patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        for group in m.groups():
            v = _safe_float(group)
            if v is not None:
                return v
    return None


def _extract_lane_count(text: str) -> Optional[int]:
    cn_lanes = {
        "单车道": 1,
        "一车道": 1,
        "双车道": 2,
        "两车道": 2,
        "二车道": 2,
        "三车道": 3,
        "四车道": 4,
        "五车道": 5,
        "六车道": 6,
    }
    for key, value in cn_lanes.items():
        if key in text:
            return value

    patterns = [
        r"(\d+)\s*-\s*lane",
        r"(\d+)\s+lane",
        r"(\d+)\s*车道",
    ]
    v = _extract_first_number(text, patterns)
    return int(v) if v is not None else None


def _extract_speed_mps(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m/s|mps|米/秒)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(km/h|kph|公里/小时|公里每小时)", text)
    if m:
        return float(m.group(1)) / 3.6
    return None


@dataclass
class PromptParseEvidence:
    """Small debug record showing why each layer was filled."""

    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    decisions: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add_keywords(self, label: str, text: str, keywords: Sequence[str]) -> None:
        matched = _matched_keywords(text, keywords)
        if matched:
            self.matched_keywords[label] = matched

    def decide(self, field_name: str, value: object, reason: str) -> None:
        self.decisions[field_name] = f"{value} ({reason})"

    def to_dict(self) -> Dict[str, object]:
        return {
            "matched_keywords": self.matched_keywords,
            "decisions": self.decisions,
            "warnings": self.warnings,
        }


class NaturalLanguagePromptParser:
    """
    Rule-based natural-language front-end for compositional hazard specs.

    This parser intentionally infers each semantic layer separately instead of
    only routing a prompt to a fixed scenario name. That keeps the output aligned
    with the compositional template used by the primitive compiler:
        prompt -> HazardSemanticSpec -> primitive ops -> scene editing.
    """

    ROAD_KEYWORDS = {
        "roundabout": ("roundabout", "traffic circle", "环岛", "转盘"),
        "intersection": ("intersection", "crossroad", "junction", "路口", "交叉口", "十字路口", "丁字路口"),
        "unprotected_left_turn": ("unprotected left", "left turn", "左转", "无保护左转"),
        "merge": ("merge", "merging", "ramp", "入口匝道", "汇入", "并入", "合流"),
        "crosswalk": ("crosswalk", "zebra crossing", "人行横道", "斑马线"),
        "construction": ("construction", "work zone", "施工", "施工区"),
        "curve": ("curve", "curved road", "弯道", "弯路"),
    }

    ACTOR_KEYWORDS = {
        "pedestrian": ("pedestrian", "person", "walker", "行人", "路人"),
        "cyclist": ("cyclist", "bicycle", "bike", "骑行者", "自行车", "电动车"),
        "lead_vehicle": ("lead vehicle", "front vehicle", "vehicle ahead", "前车", "前方车辆"),
        "rear_vehicle": ("rear vehicle", "behind vehicle", "following vehicle", "后车", "后方车辆"),
        "cutin_vehicle": ("cut in", "cutting in", "lane change", "加塞", "插入", "变道", "并线"),
        "vehicle": ("vehicle", "car", "truck", "bus", "车辆", "汽车", "卡车", "公交"),
        "static_obstacle": ("obstacle", "barrier", "cone", "blocked lane", "障碍物", "路障", "锥桶", "占道"),
    }

    INTERACTION_KEYWORDS = {
        "crossing": ("cross", "crossing", "walk across", "横穿", "穿过", "横过", "过马路"),
        "hard_brake": ("hard brake", "sudden brake", "brakes hard", "emergency brake", "急刹", "急停", "突然刹车"),
        "oncoming": ("oncoming", "opposite direction", "opposite lane", "对向", "迎面", "对面", "直行车", "来车"),
        "small_gap": ("small gap", "narrow gap", "close gap", "小间隙", "小空隙", "距离很近", "近距离"),
        "fast": ("fast", "high speed", "speeding", "aggressive", "快速", "高速", "超速", "激进"),
        "stopped": ("stopped", "stationary", "parked", "停止", "静止", "停车"),
        "slow": ("slow", "slowly", "缓慢", "低速"),
        "collision": ("collision", "crash", "hit", "碰撞", "撞击", "相撞"),
        "near_miss": ("near miss", "near-miss", "almost collide", "险些", "差点撞", "无碰撞"),
        "yielding": ("yield", "yielding", "让行", "抢行", "汇入压力"),
    }

    OCCLUSION_KEYWORDS = {
        "occluded": ("occluded", "occlusion", "hidden", "blocked view", "limited visibility", "遮挡", "被挡住", "视野受限", "盲区"),
        "truck": ("truck", "lorry", "卡车", "货车"),
        "bus": ("bus", "公交", "大巴"),
        "parked_vehicle": ("parked vehicle", "parked car", "停放车辆", "停车车辆", "路边停车"),
        "barrier": ("barrier", "wall", "fence", "路障", "护栏", "围挡"),
        "full": ("fully occluded", "complete occlusion", "完全遮挡", "全遮挡"),
        "partial": ("partially occluded", "partial occlusion", "部分遮挡", "半遮挡"),
    }

    RISK_KEYWORDS = {
        "aggressive": ("aggressive", "dangerous", "critical", "severe", "close", "small gap", "fast", "高风险", "危险", "激进", "紧急", "距离很近", "小间隙"),
        "mild": ("mild", "slight", "low risk", "safe", "轻微", "低风险", "较安全"),
        "moderate": ("moderate", "medium", "中等", "一般风险"),
    }

    def __init__(self, normalize: bool = True, validate: bool = True) -> None:
        self.normalize = normalize
        self.validate = validate

    def parse(self, prompt: str, spec_id: Optional[str] = None, normalize: Optional[bool] = None) -> HazardSemanticSpec:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty natural-language description.")

        text = _normalize_text(prompt)
        evidence = PromptParseEvidence()

        road = self._infer_road_layer(text, evidence)
        actor = self._infer_actor_layer(text, evidence)
        obj = self._infer_object_layer(text, evidence)
        inter = self._infer_interaction_layer(text, actor, road, evidence)
        risk = self._infer_risk_layer(text, inter, evidence)
        validation = self._infer_validation_layer(obj, risk)
        protection = self._infer_protection_layer(obj)

        self._apply_compositional_repairs(text, road, actor, obj, inter, risk, evidence)

        canonical_type = self._canonical_type(road, actor, inter, obj)
        if spec_id is None:
            spec_id = self._make_spec_id(canonical_type, risk.risk_level, text)

        tags = self._build_tags(road, actor, obj, inter, risk)
        spec = HazardSemanticSpec(
            spec_id=spec_id,
            description=prompt.strip(),
            canonical_type=canonical_type,
            raw_prompt=prompt.strip(),
            road_layer=road,
            actor_layer=actor,
            object_layer=obj,
            interaction_layer=inter,
            risk_layer=risk,
            validation_layer=validation,
            protection_layer=protection,
            tags=tags,
            debug={"nl_parse": evidence.to_dict()},
        )

        do_normalize = self.normalize if normalize is None else normalize
        if do_normalize:
            spec = normalize_spec(spec)

        if self.validate:
            report = check_spec(spec, strict=False)
            spec.debug.setdefault("nl_parse", {})["spec_check"] = report.to_dict()
            if not report.valid:
                raise ValueError(f"Natural-language prompt produced an invalid HazardSemanticSpec: {report.errors}")

        return spec

    def parse_to_dict(self, prompt: str, spec_id: Optional[str] = None, normalize: Optional[bool] = None) -> Dict[str, object]:
        return self.parse(prompt=prompt, spec_id=spec_id, normalize=normalize).to_dict()

    def _infer_road_layer(self, text: str, evidence: PromptParseEvidence) -> RoadLayer:
        road = RoadLayer()
        for label, keywords in self.ROAD_KEYWORDS.items():
            evidence.add_keywords(f"road.{label}", text, keywords)

        if _has_any(text, self.ROAD_KEYWORDS["roundabout"]):
            road.road_topology = "roundabout"
            road.lane_context = "adjacent_lane"
            road.anchor_type = "adjacent_lane"
            road.anchor_region = "front"
            road.has_merge_area = True
            road.allow_lane_generation = True
            road.generated_road_layout = "roundabout_entry"
            road.num_lanes = 2
            evidence.decide("road_layer.road_topology", road.road_topology, "roundabout keyword")
        elif _has_any(text, self.ROAD_KEYWORDS["unprotected_left_turn"]):
            road.road_topology = "intersection"
            road.lane_context = "opposite_lane"
            road.anchor_type = "intersection_center"
            road.anchor_region = "front"
            road.has_intersection = True
            road.allow_lane_generation = True
            road.generated_road_layout = "unprotected_left_turn"
            road.num_lanes = 4
            evidence.decide("road_layer.generated_road_layout", road.generated_road_layout, "left-turn keyword")
        elif _has_any(text, self.ROAD_KEYWORDS["intersection"]):
            road.road_topology = "intersection"
            road.anchor_type = "intersection_center"
            road.anchor_region = "front"
            road.has_intersection = True
            evidence.decide("road_layer.road_topology", road.road_topology, "intersection keyword")
        elif _has_any(text, self.ROAD_KEYWORDS["merge"]):
            road.road_topology = "merge"
            road.lane_context = "adjacent_lane"
            road.anchor_type = "adjacent_lane"
            road.anchor_region = "front"
            road.has_merge_area = True
            evidence.decide("road_layer.road_topology", road.road_topology, "merge keyword")
        elif _has_any(text, self.ROAD_KEYWORDS["crosswalk"]):
            road.road_topology = "crosswalk_area"
            road.anchor_type = "crosswalk"
            road.anchor_region = "front"
            road.has_crosswalk = True
            evidence.decide("road_layer.road_topology", road.road_topology, "crosswalk keyword")
        elif _has_any(text, self.ROAD_KEYWORDS["construction"]):
            road.road_topology = "construction_zone"
            road.anchor_type = "ego_lane_front"
            road.anchor_region = "front"
            evidence.decide("road_layer.road_topology", road.road_topology, "construction keyword")
        elif _has_any(text, self.ROAD_KEYWORDS["curve"]):
            road.road_topology = "curve"
            road.anchor_type = "ego_future_path"
            road.anchor_region = "front"
            evidence.decide("road_layer.road_topology", road.road_topology, "curve keyword")

        if _has_any(text, self.ROAD_KEYWORDS["crosswalk"]):
            road.has_crosswalk = True
        lane_count = _extract_lane_count(text)
        if lane_count is not None:
            road.num_lanes = lane_count
            evidence.decide("road_layer.num_lanes", road.num_lanes, "lane count mentioned in prompt")

        return road

    def _infer_actor_layer(self, text: str, evidence: PromptParseEvidence) -> ActorLayer:
        actor = ActorLayer()
        for label, keywords in self.ACTOR_KEYWORDS.items():
            evidence.add_keywords(f"actor.{label}", text, keywords)

        if _has_any(text, self.ACTOR_KEYWORDS["pedestrian"]):
            actor.primary_actor = "pedestrian"
            actor.actor_role = "crossing_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "pedestrian keyword")
        elif _has_any(text, self.ACTOR_KEYWORDS["cyclist"]):
            actor.primary_actor = "cyclist"
            actor.actor_role = "crossing_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "cyclist keyword")
        elif _has_any(text, self.ACTOR_KEYWORDS["static_obstacle"]):
            actor.primary_actor = "static_obstacle"
            actor.actor_role = "blocking_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "static obstacle keyword")
        elif _has_any(text, self.INTERACTION_KEYWORDS["hard_brake"]) or _has_any(text, self.ACTOR_KEYWORDS["lead_vehicle"]):
            actor.primary_actor = "lead_vehicle"
            actor.actor_role = "braking_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "hard-brake or lead-vehicle keyword")
        elif _has_any(text, self.ACTOR_KEYWORDS["rear_vehicle"]):
            actor.primary_actor = "rear_vehicle"
            actor.actor_role = "approaching_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "rear-vehicle keyword")
        elif _has_any(text, self.ACTOR_KEYWORDS["cutin_vehicle"]):
            actor.primary_actor = "cutin_vehicle"
            actor.actor_role = "merging_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "cut-in/lane-change keyword")
        else:
            actor.primary_actor = "vehicle"
            if _has_any(text, self.INTERACTION_KEYWORDS["crossing"]):
                actor.actor_role = "crossing_actor"
            elif _has_any(text, self.INTERACTION_KEYWORDS["oncoming"]):
                actor.actor_role = "approaching_actor"
            else:
                actor.actor_role = "approaching_actor"
            evidence.decide("actor_layer.primary_actor", actor.primary_actor, "default vehicle actor")

        actor.supporting_actors = ["ego_vehicle"]
        if _has_any(text, self.ROAD_KEYWORDS["roundabout"]):
            actor.supporting_actors = [
                "ego_vehicle",
                "circulating_cross_traffic",
                "queued_entry_vehicles",
                "exiting_vehicles",
            ]

        return actor

    def _infer_object_layer(self, text: str, evidence: PromptParseEvidence) -> ObjectLayer:
        for label, keywords in self.OCCLUSION_KEYWORDS.items():
            evidence.add_keywords(f"occlusion.{label}", text, keywords)

        occlusion = OcclusionSpec()
        static_obstacle = StaticObstacleSpec()

        if _has_any(text, self.OCCLUSION_KEYWORDS["occluded"]):
            occlusion.enabled = True
            occlusion.occlusion_position = "between_ego_and_actor"
            occlusion.occlusion_level = "full" if _has_any(text, self.OCCLUSION_KEYWORDS["full"]) else "partial"

            if _has_any(text, self.OCCLUSION_KEYWORDS["bus"]):
                occlusion.occluder_type = "bus"
            elif _has_any(text, self.OCCLUSION_KEYWORDS["truck"]):
                occlusion.occluder_type = "truck"
            elif _has_any(text, self.OCCLUSION_KEYWORDS["barrier"]):
                occlusion.occluder_type = "barrier"
            elif _has_any(text, self.OCCLUSION_KEYWORDS["parked_vehicle"]):
                occlusion.occluder_type = "parked_vehicle"
            else:
                occlusion.occluder_type = "parked_vehicle"

            evidence.decide("object_layer.occlusion.enabled", True, "occlusion keyword")

        if _has_any(text, self.ACTOR_KEYWORDS["static_obstacle"]) or _has_any(text, self.ROAD_KEYWORDS["construction"]):
            static_obstacle.enabled = True
            static_obstacle.obstacle_position = "ego_lane"
            if _has_any(text, ("cone", "锥桶")):
                static_obstacle.obstacle_type = "cone"
                static_obstacle.obstacle_size = "small"
            elif _has_any(text, ("barrier", "路障", "护栏", "围挡")):
                static_obstacle.obstacle_type = "barrier"
            elif _has_any(text, ("parked vehicle", "parked car", "停放车辆", "停车车辆")):
                static_obstacle.obstacle_type = "parked_vehicle"
            elif _has_any(text, self.ROAD_KEYWORDS["construction"]):
                static_obstacle.obstacle_type = "construction_zone"
                static_obstacle.obstacle_size = "large"
            else:
                static_obstacle.obstacle_type = "barrier"
            evidence.decide("object_layer.static_obstacle.enabled", True, "obstacle/construction keyword")

        return ObjectLayer(occlusion=occlusion, static_obstacle=static_obstacle)

    def _infer_interaction_layer(
        self,
        text: str,
        actor: ActorLayer,
        road: RoadLayer,
        evidence: PromptParseEvidence,
    ) -> InteractionLayer:
        for label, keywords in self.INTERACTION_KEYWORDS.items():
            evidence.add_keywords(f"interaction.{label}", text, keywords)

        inter = InteractionLayer()
        inter.distance_relation = "medium"
        inter.speed_relation = "normal"
        inter.interaction_goal = "near_miss"

        if road.generated_road_layout == "unprotected_left_turn" or _has_any(text, self.INTERACTION_KEYWORDS["oncoming"]):
            inter.conflict_type = "oncoming_conflict"
            inter.conflict_direction = "opposite"
            inter.distance_relation = "close"
            inter.speed_relation = "fast_approach" if _has_any(text, self.INTERACTION_KEYWORDS["fast"]) else "normal"
            inter.interaction_goal = "collision_risk" if _has_any(text, self.INTERACTION_KEYWORDS["collision"]) else "near_miss"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "oncoming/left-turn keyword")
        elif road.road_topology == "roundabout":
            inter.conflict_type = "merging_conflict"
            inter.conflict_direction = self._merge_direction(text)
            inter.distance_relation = "small_gap"
            inter.speed_relation = "fast_approach" if _has_any(text, self.INTERACTION_KEYWORDS["fast"]) else "normal"
            inter.interaction_goal = "yielding_conflict"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "roundabout entry merge")
        elif actor.primary_actor == "cutin_vehicle" or _has_any(text, self.ROAD_KEYWORDS["merge"]):
            inter.conflict_type = "merging_conflict"
            inter.conflict_direction = self._merge_direction(text)
            inter.distance_relation = "small_gap" if _has_any(text, self.INTERACTION_KEYWORDS["small_gap"]) else "medium"
            inter.speed_relation = "fast_approach" if _has_any(text, self.INTERACTION_KEYWORDS["fast"]) else "normal"
            inter.interaction_goal = "yielding_conflict" if _has_any(text, self.INTERACTION_KEYWORDS["yielding"]) else "near_miss"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "cut-in/merge keyword")
        elif actor.primary_actor in {"lead_vehicle", "rear_vehicle"} or _has_any(text, self.INTERACTION_KEYWORDS["hard_brake"]):
            inter.conflict_type = "longitudinal_conflict"
            inter.conflict_direction = "rear" if actor.primary_actor == "rear_vehicle" else "front"
            inter.distance_relation = "short_headway"
            inter.speed_relation = "stopped" if _has_any(text, self.INTERACTION_KEYWORDS["stopped"]) else "slow_lead"
            if _has_any(text, self.INTERACTION_KEYWORDS["hard_brake"]):
                inter.speed_relation = "stopped"
            inter.interaction_goal = "braking_pressure"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "longitudinal braking/following keyword")
        elif actor.primary_actor == "static_obstacle":
            inter.conflict_type = "lane_blocking_conflict"
            inter.conflict_direction = "front"
            inter.distance_relation = "close"
            inter.speed_relation = "stopped"
            inter.interaction_goal = "trajectory_blocking"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "lane blocking keyword")
        elif actor.primary_actor == "vehicle" and _has_any(text, self.INTERACTION_KEYWORDS["crossing"]):
            inter.conflict_type = "crossing_path_conflict"
            inter.conflict_direction = self._crossing_direction(text)
            inter.distance_relation = "close"
            inter.speed_relation = "fast_crossing" if _has_any(text, self.INTERACTION_KEYWORDS["fast"]) else "normal"
            inter.interaction_goal = "collision_risk" if _has_any(text, self.INTERACTION_KEYWORDS["collision"]) else "near_miss"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "crossing vehicle keyword")
        elif actor.primary_actor in {"pedestrian", "cyclist"}:
            inter.conflict_type = "lateral_conflict"
            inter.conflict_direction = self._crossing_direction(text)
            inter.distance_relation = "close" if _has_any(text, self.INTERACTION_KEYWORDS["small_gap"]) else "medium"
            inter.speed_relation = "fast_crossing" if _has_any(text, self.INTERACTION_KEYWORDS["fast"]) else "normal"
            inter.interaction_goal = "near_miss"
            evidence.decide("interaction_layer.conflict_type", inter.conflict_type, "vulnerable-road-user crossing")
        else:
            inter.conflict_type = "longitudinal_conflict"
            inter.conflict_direction = "front"
            inter.distance_relation = "medium"
            inter.speed_relation = "normal"
            inter.interaction_goal = "near_miss"
            evidence.warnings.append("No strong interaction keyword was found; defaulted to front longitudinal conflict.")

        if _has_any(text, self.INTERACTION_KEYWORDS["collision"]):
            inter.interaction_goal = "collision_risk"
        if _has_any(text, self.INTERACTION_KEYWORDS["near_miss"]):
            inter.interaction_goal = "near_miss"
        if _has_any(text, self.INTERACTION_KEYWORDS["small_gap"]):
            inter.distance_relation = "small_gap" if inter.conflict_type == "merging_conflict" else "close"

        return inter

    def _infer_risk_layer(self, text: str, inter: InteractionLayer, evidence: PromptParseEvidence) -> RiskLayer:
        for label, keywords in self.RISK_KEYWORDS.items():
            evidence.add_keywords(f"risk.{label}", text, keywords)

        risk = RiskLayer()
        if _has_any(text, self.RISK_KEYWORDS["mild"]):
            risk.risk_level = "mild"
        elif _has_any(text, self.RISK_KEYWORDS["aggressive"]) or inter.interaction_goal == "collision_risk":
            risk.risk_level = "aggressive"
        else:
            risk.risk_level = "moderate"
        evidence.decide("risk_layer.risk_level", risk.risk_level, "risk keywords and interaction goal")

        ttc = _extract_first_number(
            text,
            (
                r"ttc\s*(?:=|:|<|less than|小于|低于)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*s\s*ttc",
                r"(\d+(?:\.\d+)?)\s*秒\s*(?:ttc|碰撞时间)",
            ),
        )
        if ttc is not None:
            risk.ttc_range_s = _range_around(ttc, ratio=0.25, minimum_width=0.2)
            evidence.decide("risk_layer.ttc_range_s", risk.ttc_range_s, "TTC number mentioned in prompt")

        gap = _extract_first_number(
            text,
            (
                r"gap\s*(?:=|:|<|less than|小于|低于)?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*m\s*(?:gap|间隙|距离)",
                r"(?:间隙|距离)\s*(?:=|:|小于|低于)?\s*(\d+(?:\.\d+)?)\s*米?",
            ),
        )
        if gap is not None:
            risk.gap_range_m = _range_around(gap, ratio=0.25, minimum_width=0.5)
            risk.longitudinal_distance_range_m = _range_around(gap, ratio=0.35, minimum_width=1.0)
            evidence.decide("risk_layer.gap_range_m", risk.gap_range_m, "gap/distance number mentioned in prompt")

        speed = _extract_speed_mps(text)
        if speed is not None:
            risk.target_actor_speed_mps = speed
            risk.target_relative_speed_mps = max(1.0, speed * 0.6)
            evidence.decide("risk_layer.target_actor_speed_mps", round(speed, 3), "speed number mentioned in prompt")

        risk.collision_allowed = False
        return risk

    def _infer_validation_layer(self, obj: ObjectLayer, risk: RiskLayer) -> ValidationLayer:
        validation = ValidationLayer()
        validation.require_visibility_match = obj.occlusion.enabled
        validation.require_ttc_in_range = risk.risk_level == "aggressive"
        validation.require_gap_in_range = True
        # Road generation is currently a synthesis aid; strict road-context checks
        # can be too brittle before the editor adjusts topology.
        validation.require_road_context_match = False
        return validation

    def _infer_protection_layer(self, obj: ObjectLayer) -> ProtectionLayer:
        protection = ProtectionLayer()
        protection.protect_secondary_actor = obj.occlusion.enabled
        protection.protect_static_obstacle = obj.static_obstacle.enabled
        return protection

    def _apply_compositional_repairs(
        self,
        text: str,
        road: RoadLayer,
        actor: ActorLayer,
        obj: ObjectLayer,
        inter: InteractionLayer,
        risk: RiskLayer,
        evidence: PromptParseEvidence,
    ) -> None:
        """
        Make cross-layer fields compatible after independent layer inference.
        These are not scenario templates; they repair invalid combinations.
        """
        if road.road_topology == "roundabout":
            actor.primary_actor = "cutin_vehicle"
            actor.actor_role = "merging_actor"
            inter.conflict_type = "merging_conflict"
            inter.conflict_direction = self._merge_direction(text)
            inter.distance_relation = "small_gap"
            inter.interaction_goal = "yielding_conflict"
            if risk.risk_level == "moderate" and _has_any(text, self.INTERACTION_KEYWORDS["small_gap"]):
                risk.risk_level = "aggressive"
            evidence.decide("repair.roundabout_actor", actor.primary_actor, "roundabout merge requires a merging vehicle")

        if road.generated_road_layout == "unprotected_left_turn":
            actor.primary_actor = "vehicle"
            actor.actor_role = "approaching_actor"
            inter.conflict_type = "oncoming_conflict"
            inter.conflict_direction = "opposite"
            inter.distance_relation = "close"
            if _has_any(text, self.INTERACTION_KEYWORDS["fast"]):
                inter.speed_relation = "fast_approach"
            if risk.risk_level == "moderate":
                risk.risk_level = "aggressive"
            evidence.decide("repair.left_turn_conflict", inter.conflict_type, "left turn with oncoming traffic")

        if actor.primary_actor == "static_obstacle":
            obj.static_obstacle.enabled = True
            if obj.static_obstacle.obstacle_type == "none":
                obj.static_obstacle.obstacle_type = "barrier"
            if obj.static_obstacle.obstacle_position == "none":
                obj.static_obstacle.obstacle_position = "ego_lane"

        if obj.occlusion.enabled:
            actor.secondary_actor = obj.occlusion.occluder_type
            if obj.occlusion.occlusion_position in {"", "none"}:
                obj.occlusion.occlusion_position = "between_ego_and_actor"
            if obj.occlusion.occlusion_level == "none":
                obj.occlusion.occlusion_level = "partial"

    def _merge_direction(self, text: str) -> str:
        right = ("from right", "right lane", "right side", "右侧", "右车道", "从右")
        left = ("from left", "left lane", "left side", "左侧", "左车道", "从左")
        if _has_any(text, right):
            return "right_merge"
        if _has_any(text, left):
            return "left_merge"
        return "left_merge"

    def _crossing_direction(self, text: str) -> str:
        left_to_right = ("left to right", "from left", "left side", "左向右", "从左", "左侧")
        right_to_left = ("right to left", "from right", "right side", "右向左", "从右", "右侧")
        if _has_any(text, left_to_right):
            return "left_to_right"
        if _has_any(text, right_to_left):
            return "right_to_left"
        return "right_to_left"

    def _canonical_type(
        self,
        road: RoadLayer,
        actor: ActorLayer,
        inter: InteractionLayer,
        obj: ObjectLayer,
    ) -> str:
        if road.generated_road_layout == "unprotected_left_turn":
            return "Unprotected-Left-Turn-Oncoming"
        if road.generated_road_layout == "roundabout_entry":
            return "Roundabout-Entry-Merge"
        if actor.primary_actor == "pedestrian" and obj.occlusion.enabled:
            return "Occluded-Pedestrian-Crossing"
        if actor.primary_actor == "pedestrian":
            return "Ped-Crossing"
        if actor.primary_actor == "cyclist":
            return "Cyclist-Crossing"
        if actor.primary_actor == "cutin_vehicle":
            return "Cut-in"
        if actor.primary_actor == "lead_vehicle":
            return "Hard-Brake"
        if actor.primary_actor == "static_obstacle":
            return "Lane-Blocking-Obstacle"
        if inter.conflict_type == "crossing_path_conflict":
            return "Crossing-Vehicle"
        return "Compositional-Hazard"

    def _make_spec_id(self, canonical_type: str, risk_level: str, text: str) -> str:
        base = _slugify(canonical_type, fallback="nl_spec")
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return f"{base}_{risk_level}_{digest}".lower().replace("-", "_")

    def _build_tags(
        self,
        road: RoadLayer,
        actor: ActorLayer,
        obj: ObjectLayer,
        inter: InteractionLayer,
        risk: RiskLayer,
    ) -> List[str]:
        tags = [
            "nl_parsed",
            "compositional",
            road.road_topology,
            actor.primary_actor,
            actor.actor_role,
            inter.conflict_type,
            inter.conflict_direction,
            inter.interaction_goal,
            risk.risk_level,
        ]
        if road.allow_lane_generation:
            tags.append("lane_generation")
        if road.generated_road_layout != "none":
            tags.append(road.generated_road_layout)
        if obj.occlusion.enabled:
            tags.extend(["occlusion", obj.occlusion.occluder_type])
        if obj.static_obstacle.enabled:
            tags.extend(["static_obstacle", obj.static_obstacle.obstacle_type])
        # Keep ordering stable while removing duplicates/empty tags.
        out: List[str] = []
        seen = set()
        for tag in tags:
            if not tag or tag == "none" or tag in seen:
                continue
            out.append(tag)
            seen.add(tag)
        return out


def parse_prompt_to_spec(prompt: str, spec_id: Optional[str] = None, normalize: bool = True) -> HazardSemanticSpec:
    """Convenience wrapper used by scripts and notebooks."""
    return NaturalLanguagePromptParser(normalize=normalize).parse(prompt=prompt, spec_id=spec_id)
