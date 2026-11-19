#!/usr/bin/env python3
"""Viser-only geometric pose/limit editor; it does not run physics or hardware."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
COACD_URDF = BUNDLE / "urdf" / "moresense_doge_coacd.urdf"
SIMPLE_URDF = BUNDLE / "urdf" / "moresense_doge_simple.urdf"
DEFAULT_URDF = SIMPLE_URDF if SIMPLE_URDF.is_file() else COACD_URDF if COACD_URDF.is_file() else BUNDLE / "urdf" / "moresense_doge.urdf"
EXPLORER = (-360.0, 360.0)  # Initial slider window only, never a physical limit.


def vector(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.array(default if text is None else [float(v) for v in text.split()])
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Invalid finite three-vector")
    return value


def origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    r, p, y = vector(element.get("rpy"))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    transform[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                          [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                          [-sp, cp*sr, cp*cr]]
    transform[:3, 3] = vector(element.get("xyz"))
    return transform


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    value = np.eye(4)
    value[:3, :3] = np.eye(3) + math.sin(angle)*skew + (1.-math.cos(angle))*(skew @ skew)
    return value


def without_collisions(element):
    """Structural comparison of model content while ignoring collision geometry."""
    return (element.tag, tuple(sorted(element.attrib.items())),
            (element.text or '').strip(),
            tuple(without_collisions(child) for child in element if child.tag != 'collision'))


class RobotModel:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.source_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.source_bytes).hexdigest()
        self.xml = ET.fromstring(self.source_bytes)
        self.links = {el.attrib["name"]: el for el in self.xml.findall("link")}
        self.joints = {el.attrib["name"]: el for el in self.xml.findall("joint")}
        self.names = list(self.joints)
        if len(self.links) != 13 or len(self.joints) != 12:
            raise ValueError("Expected the prepared robot with 13 links and 12 joints")
        self.parent_child = {}
        children = set()
        for name, joint in self.joints.items():
            if joint.get("type") not in ("continuous", "revolute"):
                raise ValueError(f"Unsupported joint: {name}")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            if parent not in self.links or child not in self.links or child in children:
                raise ValueError(f"Invalid parent/child: {name}")
            children.add(child)
            self.parent_child[name] = (parent, child)
            axis = joint.find("axis")
            if axis is None or np.linalg.norm(vector(axis.get("xyz"))) < 1e-12:
                raise ValueError(f"Invalid joint axis: {name}")
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError("The model must have one root link")
        self.root = roots.pop()
        self.limits = {}
        for name, joint in self.joints.items():
            limit = joint.find("limit")
            if "shoulder" in name:
                if limit is None or joint.get("type") != "revolute":
                    raise ValueError(f"Missing original shoulder limits: {name}")
                lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                    raise ValueError(f"Invalid shoulder bounds: {name}")
                self.limits[name] = (math.degrees(lower), math.degrees(upper))
            else:
                self.limits[name] = EXPLORER
        self.foot_collisions = {}
        for name, link in self.links.items():
            for collision in link.findall("collision"):
                cylinder = collision.find("geometry/cylinder")
                if cylinder is not None:
                    radius, length = float(cylinder.attrib["radius"]), float(cylinder.attrib["length"])
                    if not (math.isfinite(radius) and math.isfinite(length) and radius > 0 and length > 0):
                        raise ValueError("Invalid foot cylinder")
                    self.foot_collisions.setdefault(name, []).append((origin(collision.find("origin")), radius, length))
        self.foot_refs_approx = not bool(self.foot_collisions)
        if not self.foot_collisions:
            # Independent, approximate foot references, not physics colliders.
            refs = json.loads((BUNDLE / 'tools' / 'foot_reference.json').read_text(encoding='utf-8'))
            for name, data in refs.items():
                if name not in self.links:
                    raise ValueError(f'Missing foot link {name}')
                ref = ET.Element('origin', xyz=data['xyz'], rpy=data['rpy'])
                self.foot_collisions[name] = [(origin(ref), float(data['radius']), float(data['length']))]
        if len(self.foot_collisions) != 4:
            raise ValueError("Expected four foot references")
        self.last_load_migrated = False
        self.fk({name: 0.0 for name in self.names}, 0.145)

    def defaults(self):
        return {name: {"lower_deg": limits[0], "upper_deg": limits[1],
                       "confirmed": "shoulder" in name,
                       "source": "original_urdf" if "shoulder" in name else "unvalidated_explorer"}
                for name, limits in self.limits.items()}

    def fk(self, q_rad: dict[str, float], root_height: float) -> dict[str, np.ndarray]:
        result = {self.root: np.eye(4)}
        result[self.root][2, 3] = root_height
        pending = dict(self.joints)
        while pending:
            progress = False
            for name, joint in list(pending.items()):
                parent, child = self.parent_child[name]
                if parent not in result:
                    continue
                result[child] = (result[parent] @ origin(joint.find("origin")) @
                                 rotation(vector(joint.find("axis").get("xyz")), q_rad[name]))
                del pending[name]
                progress = True
            if not progress:
                raise ValueError("Disconnected or cyclic URDF graph")
        return result

    def diagnostics(self, q_rad: dict[str, float], root_height: float) -> dict:
        transforms = self.fk(q_rad, root_height)
        feet = {}
        for name, cylinders in self.foot_collisions.items():
            points = []
            for local, radius, length in cylinders:
                world = transforms[name] @ local
                axis_z = float(np.clip(world[2, 2], -1.0, 1.0))
                extent = abs(axis_z)*length/2 + radius*math.sqrt(max(0., 1.-axis_z*axis_z))
                points.append((float(world[2, 3]-extent), world[:3, 3].tolist()))
            bottom, center = min(points, key=lambda point: point[0])
            feet[name] = {"center_world_m": center, "bottom_z_m": bottom}
        mass, weighted = 0., np.zeros(3)
        for name, link in self.links.items():
            inertial = link.find("inertial")
            if inertial is None:
                raise ValueError(f"Missing mass in {name}")
            link_mass = float(inertial.find("mass").attrib["value"])
            if not math.isfinite(link_mass) or link_mass <= 0:
                raise ValueError(f"Invalid mass in {name}")
            world = transforms[name] @ origin(inertial.find("origin"))
            mass += link_mass
            weighted += link_mass * world[:3, 3]
        bottoms = [foot["bottom_z_m"] for foot in feet.values()]
        return {"feet": feet, "foot_height_spread_m": max(bottoms)-min(bottoms),
                "lowest_foot_z_m": min(bottoms), "approximate_com_world_m": (weighted/mass).tolist(),
                "model_mass_kg": mass, "geometry_only": True, "physics_validated": False}

    def validate(self, q_rad: dict, root_height: float, limits: dict):
        if set(q_rad) != set(self.names) or set(limits) != set(self.names):
            raise ValueError("Joint configuration must contain exactly the 12 model joints")
        if not math.isfinite(root_height) or not 0 <= root_height <= .5:
            raise ValueError("Высота корпуса должна быть от 0 до 0.5 м")
        for name in self.names:
            q = float(q_rad[name])
            lower, upper = float(limits[name]["lower_deg"]), float(limits[name]["upper_deg"])
            if not all(math.isfinite(v) for v in (q, lower, upper)) or lower >= upper:
                raise ValueError(f"{name}: нужны конечные значения и минимум < максимума")
            if type(limits[name]["confirmed"]) is not bool:
                raise ValueError(f"Invalid confirmation flag for {name}")
            original_lower, original_upper = self.limits[name]
            degrees = math.degrees(q)
            if "shoulder" in name:
                if not original_lower-1e-8 <= degrees <= original_upper+1e-8:
                    raise ValueError(f"{name}: угол вне заданных границ плеча")
                if not (abs(lower-original_lower) < 1e-9 and abs(upper-original_upper) < 1e-9):
                    raise ValueError(f"{name}: исходные границы плеча менять нельзя")
                if limits[name]["confirmed"] is not True:
                    raise ValueError(f"{name}: original shoulder limits must remain active")
            if limits[name]["confirmed"] and not lower-1e-8 <= degrees <= upper+1e-8:
                raise ValueError(f"{name}: текущий угол вне подтверждённых границ")

    def save(self, save_dir: Path, q_rad: dict, root_height: float, limits: dict) -> tuple[Path, Path]:
        self.validate(q_rad, root_height, limits)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("Исходная модель изменилась: перезапустите редактор")
        save_dir = save_dir.resolve()
        urdf_path = save_dir / "urdf" / "moresense_calibrated.urdf"
        json_path = save_dir / "pose_and_limits.json"
        if urdf_path.resolve() == self.path:
            raise ValueError("Нельзя перезаписывать исходный URDF")
        root = copy.deepcopy(self.xml)
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename", "")
            if "://" in filename:
                raise ValueError(f"Unsupported mesh URI: {filename}")
            original_mesh = (self.path.parent / filename).resolve()
            if not original_mesh.is_file():
                raise ValueError(f"Mesh not found: {original_mesh}")
            mesh.set("filename", os.path.relpath(original_mesh, urdf_path.parent).replace(os.sep, "/"))
        changed = []
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            if "shoulder" not in name and limits[name]["confirmed"]:
                joint.set("type", "revolute")
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"Missing actuator limits in {name}")
                limit.set("lower", format(math.radians(limits[name]["lower_deg"]), ".17g"))
                limit.set("upper", format(math.radians(limits[name]["upper_deg"]), ".17g"))
                changed.append(name)
        payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "source_urdf": self.path.name, "source_sha256": self.sha256,
                   "calibrated_urdf": "urdf/moresense_calibrated.urdf", "joint_order": self.names,
                   "initial_root_position_m": [0., 0., float(root_height)],
                   "initial_root_quaternion_wxyz": [1., 0., 0., 0.],
                   "initial_joint_positions_rad": {name: float(q_rad[name]) for name in self.names},
                   "joint_limits": copy.deepcopy(limits), "converted_to_revolute": changed,
                   "all_joint_limits_confirmed": all(v["confirmed"] for v in limits.values()),
                   "physics_validated": False, "diagnostics": self.diagnostics(q_rad, root_height)}
        # Each file is replaced atomically; the original URDF is never modified.
        atomic_write(urdf_path, ET.tostring(root, encoding="utf-8", xml_declaration=True))
        atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        return json_path, urdf_path

    def load_saved(self, save_dir: Path) -> tuple[dict, float, dict]:
        payload = json.loads((save_dir / "pose_and_limits.json").read_text(encoding="utf-8"))
        self.last_load_migrated = False
        if payload.get("schema_version") != 1:
            raise ValueError("Неподдерживаемая версия сохранённой позы")
        if payload.get("source_sha256") != self.sha256:
            candidates = [self.path.parent / name for name in ('moresense_doge.urdf', 'moresense_doge_coacd.urdf')]
            previous = next((p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == payload.get('source_sha256')), None)
            if previous is None:
                raise ValueError("Сохранённая поза относится к другой модели или версии файла")
            if without_collisions(ET.fromstring(previous.read_bytes())) != without_collisions(self.xml):
                raise ValueError("Изменилась модель помимо коллизий: автоматически загрузить позу нельзя")
            self.last_load_migrated = True
        position = payload.get("initial_root_position_m")
        if not isinstance(position, list) or len(position) != 3 or position[:2] != [0., 0.]:
            raise ValueError("Поддерживается только изменение высоты корпуса")
        if payload.get("initial_root_quaternion_wxyz") != [1., 0., 0., 0.]:
            raise ValueError("Сохранённый поворот корпуса не поддерживается")
        q, limits = payload["initial_joint_positions_rad"], payload["joint_limits"]
        height = float(position[2])
        self.validate(q, height, limits)
        if self.last_load_migrated:
            limits = copy.deepcopy(limits)
            for name in self.names:
                if 'shoulder' not in name:
                    limits[name]['confirmed'] = False
                    limits[name]['source'] = 'previous_collision_model_recheck_required'
        return q, height, limits


def atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def serve(model: RobotModel, host: str, port: int, save_dir: Path, smoke_seconds: float | None):
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(host=host, port=port)
    if server.get_port() != port:
        server.stop()
        raise RuntimeError(f"Port {port} is occupied; refusing an automatically changed port")
    server.gui.configure_theme(show_share_button=False)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=1.2, height=1.2, cell_size=.025, section_size=.1,
                          plane="xy", plane_opacity=.12)
    base = server.scene.add_frame("/robot", show_axes=False, position=(0., 0., .145))
    renderer = ViserUrdf(server, model.path, root_node_name="/robot", load_meshes=True,
                         load_collision_meshes=True, collision_mesh_color_override=(.1, .7, 1., .35))
    renderer.show_collision = False
    foot_labels = {name: server.scene.add_label('/feet/'+name, f"Нога {name.split('_')[1]}",
                    font_screen_scale=.8, anchor='bottom-center') for name in model.foot_collisions}
    com_sphere = server.scene.add_icosphere('/center_of_mass/sphere', radius=.006,
                                           color=(255, 210, 0), subdivisions=2)
    # The label is drawn through the shell, so the precise point remains visible
    # even when the center of mass is inside the opaque torso.
    com_dot = server.scene.add_label('/center_of_mass/dot', '🟡', depth_test=False,
                                     anchor='center-center', font_screen_scale=1.1)
    com_label = server.scene.add_label('/center_of_mass/text', 'Центр масс', depth_test=False,
                                       anchor='bottom-center', font_screen_scale=.8)
    lock = threading.RLock()
    loading = False
    sliders, controls, angle_numbers = {}, {}, {}
    defaults = model.defaults()
    server.gui.add_markdown("## Поза MoreSense Doge\nГеометрический редактор: физика не запущена. "
                            "Коллизии приближённые; проверка зазоров только визуальная. "
                            "Самопересечения автоматически не обнаруживаются и движение не ограничивается. "
                            "Углы отсчитываются от исходной CAD-позы. "
                            "Для бёдер и коленей числовой угол не ограничен. Ползунок расширяется под введённый угол. "
                            "Допустимые границы робота задаются отдельно после проверки CAD-геометрии.")
    height = server.gui.add_slider("Высота корпуса Z, м", min=0., max=.5, step=.0005, initial_value=.145)
    visuals = server.gui.add_checkbox("Внешний вид", initial_value=True)
    collisions = server.gui.add_checkbox("Коллизии", initial_value=model.path.name.endswith(('_coacd.urdf', '_simple.urdf')))
    show_com = server.gui.add_checkbox("Центр масс", initial_value=True)
    diagnostics_text = server.gui.add_markdown("")
    status = server.gui.add_markdown("Изменения сохраняются кнопкой ниже.")

    def current():
        q = {name: math.radians(float(sliders[name].value)) for name in model.names}
        limits = {name: {"lower_deg": float(parts[0].value), "upper_deg": float(parts[1].value),
                         "confirmed": bool(parts[2].value),
                         "source": "original_urdf" if "shoulder" in name else "manual_candidate"}
                  for name, parts in controls.items()}
        return q, limits

    def refresh(_event=None):
        with lock:
            if loading:
                return
            q, _ = current()
            base.position = (0., 0., float(height.value))
            renderer.update_cfg(np.array([q[name] for name in renderer.get_actuated_joint_names()]))
            renderer.show_visual, renderer.show_collision = visuals.value, collisions.value
            data = model.diagnostics(q, float(height.value))
            for name, foot in data['feet'].items():
                foot_labels[name].position = np.array(foot['center_world_m']) + np.array([0.,0.,.018])
            com = data["approximate_com_world_m"]
            for marker in (com_sphere, com_dot, com_label):
                marker.position = com
                marker.visible = show_com.value
            rows = "\n".join(f"- Нога {name.split('_')[1]}: {foot['bottom_z_m']*1000:.1f} мм" for name, foot in data["feet"].items())
            foot_description = 'Приблизительный низ стоп (прежние опорные цилиндры)' if model.foot_refs_approx else 'Низ коллизий стоп относительно пола'
            diagnostics_text.content = (f"**{foot_description}:**\n{rows}\n\n"
                f"Перепад стоп: **{data['foot_height_spread_m']*1000:.1f} мм**. "
                f"Приближённый центр масс XYZ: {com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f} м. "
                "Это геометрия; устойчивость и контакты ещё не проверены.")

    for leg in range(1, 5):
        with server.gui.add_folder(f"Нога {leg}", expand_by_default=False):
            for kind, label in (("shoulder", "Плечо"), ("hip", "Бедро"), ("knee", "Колено")):
                name = f"leg_{leg}_{kind}_joint"
                lower, upper = model.limits[name]
                with server.gui.add_folder(label):
                    slider = server.gui.add_slider("Угол, °", min=lower, max=upper, step=.1, initial_value=0.)
                    sliders[name] = slider
                    shoulder = kind == "shoulder"
                    angle_numbers[name] = server.gui.add_number("Точный угол, °", initial_value=0., step=.1,
                        **({"min": lower, "max": upper} if shoulder else {}))
                    low_control = server.gui.add_number("Минимум, °", initial_value=lower,
                                                        step=.1, disabled=shoulder)
                    high_control = server.gui.add_number("Максимум, °", initial_value=upper,
                                                         step=.1, disabled=shoulder)
                    low_button = server.gui.add_button("Текущий угол → минимум", disabled=shoulder)
                    high_button = server.gui.add_button("Текущий угол → максимум", disabled=shoulder)
                    confirmed = server.gui.add_checkbox("Исходные границы плеча" if shoulder else "Границы проверены вручную",
                                                        initial_value=defaults[name]["confirmed"], disabled=shoulder)
                    controls[name] = low_control, high_control, confirmed

                    def capture(_event, target=low_control, source=slider, flag=confirmed):
                        with lock:
                            target.value = float(source.value)
                            flag.value = False

                    def capture_high(_event, target=high_control, source=slider, flag=confirmed):
                        with lock:
                            target.value = float(source.value)
                            flag.value = False

                    low_button.on_click(capture)
                    high_button.on_click(capture_high)

                    def edited(_event, flag=confirmed):
                        with lock:
                            if not loading:
                                flag.value = False

                    if not shoulder:
                        low_control.on_update(edited)
                        high_control.on_update(edited)
                    def number_changed(_event, joint_name=name):
                        with lock:
                            value = float(angle_numbers[joint_name].value)
                            if not math.isfinite(value):
                                return
                            target = sliders[joint_name]
                            if "shoulder" not in joint_name:
                                target.min = min(target.min, math.floor(value/90)*90-90)
                                target.max = max(target.max, math.ceil(value/90)*90+90)
                            if target.value != value:
                                target.value = value
                            refresh()

                    def slider_changed(_event, joint_name=name):
                        with lock:
                            value = float(sliders[joint_name].value)
                            if angle_numbers[joint_name].value != value:
                                angle_numbers[joint_name].value = value
                            refresh()

                    angle_numbers[name].on_update(number_changed)
                    slider.on_update(slider_changed)

    align = server.gui.add_button("Нижняя стопа на пол")
    reset = server.gui.add_button("Сбросить углы в CAD-ноль")
    load = server.gui.add_button("Загрузить последнюю сохранённую позу")
    save = server.gui.add_button("Сохранить позу и подтверждённые границы")
    server.gui.add_markdown(f"Сохранение: `{save_dir.resolve()}`. "
                            "Неподтверждённые границы сохраняются только в JSON; в копии URDF они остаются continuous.")

    @align.on_click
    def _align(_event):
        with lock:
            q, _ = current()
            bottom = model.diagnostics(q, float(height.value))["lowest_foot_z_m"]
            desired = float(height.value)-bottom
            if not 0 <= desired <= .5:
                status.content = "Не удалось выровнять: нужна высота вне диапазона 0–0.5 м."
                return
            height.value = desired
            refresh()

    @reset.on_click
    def _reset(_event):
        with lock:
            for slider in sliders.values():
                slider.value = 0.
            height.value = .145
            refresh()
            status.content = "CAD-ноль восстановлен; кандидаты границ сохранены."

    @save.on_click
    def _save(event):
        with lock:
            try:
                q, limits = current()
                json_path, urdf_path = model.save(save_dir, q, float(height.value), limits)
                count = sum(value["confirmed"] for value in limits.values())
                status.content = f"Сохранено: `{json_path}` и `{urdf_path}`. Подтверждено границ: {count}/12. Физика не проверена."
                if event.client is not None:
                    event.client.send_file_download(filename=json_path.name, content=json_path.read_bytes())
            except (ValueError, OSError) as error:
                status.content = f"**Не сохранено:** {error}"

    @load.on_click
    def _load(_event):
        nonlocal loading
        with lock:
            try:
                q, root_height, limits = model.load_saved(save_dir)
                loading = True
                try:
                    height.value = root_height
                    for name in model.names:
                        value = math.degrees(q[name])
                        if "shoulder" not in name:
                            sliders[name].min = min(sliders[name].min, math.floor(value/90)*90-90)
                            sliders[name].max = max(sliders[name].max, math.ceil(value/90)*90+90)
                        sliders[name].value = math.degrees(q[name])
                        angle_numbers[name].value = value
                        lower, upper, flag = controls[name]
                        lower.value, upper.value = limits[name]["lower_deg"], limits[name]["upper_deg"]
                        flag.value = limits[name]["confirmed"]
                finally:
                    loading = False
                refresh()
                status.content = ("Загружена прежняя поза. Коллизии изменились: проверьте и заново подтвердите границы бедра и колена. Старый файл не изменён."
                                  if model.last_load_migrated else "Сохранённая поза и отметки проверки границ восстановлены.")
            except (ValueError, OSError, KeyError, TypeError) as error:
                status.content = f"**Не загружено:** {error}"

    @server.on_client_connect
    def _connect(client):
        client.camera.position = (.65, -.65, .5)
        client.camera.look_at = (.08, .08, .12)

    height.on_update(refresh)
    visuals.on_update(refresh)
    collisions.on_update(refresh)
    show_com.on_update(refresh)
    refresh()
    if (save_dir / 'pose_and_limits.json').is_file():
        _load(None)
    print(f"POSE_EDITOR_READY http://{host}:{port} source_sha256={model.sha256}", flush=True)
    started = time.monotonic()
    try:
        while smoke_seconds is None or time.monotonic()-started < smoke_seconds:
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--save-dir", type=Path, default=BUNDLE / "calibration")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check", action="store_true", help="Validate model and print CAD-zero geometry without starting Viser")
    parser.add_argument("--smoke-seconds", type=float, help="Exit after a short server startup check")
    args = parser.parse_args()
    model = RobotModel(args.urdf)
    if args.check:
        q = {name: 0. for name in model.names}
        model.validate(q, .145, model.defaults())
        print(json.dumps({"source_sha256": model.sha256, "links": len(model.links), "joints": len(model.joints),
                          "original_shoulder_limits_deg": {n: lim for n, lim in model.limits.items() if "shoulder" in n},
                          "diagnostics": model.diagnostics(q, .145)}, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        if not 1 <= args.port <= 65535:
            parser.error("Port must be between 1 and 65535")
        serve(model, args.host, args.port, args.save_dir, args.smoke_seconds)


if __name__ == "__main__":
    main()
