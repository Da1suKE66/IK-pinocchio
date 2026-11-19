"""Exact full-pose branches for the Firefly Z, Y, Y, Y, -X, Z arm."""

from itertools import product

import numpy as np
import pinocchio as pin


TAU = 2 * np.pi
EPS = 1e-10


def arm_geometry(xml, side):
    """Validate the chain before using its planar geometry. Read lengths from URDF."""
    axes = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1)]
    offsets = []
    parent = f"{side}_base_link"
    for i, axis in enumerate(axes, 1):
        joint = xml.find(f"joint[@name='{side}_joint_{i}']")
        child = f"{side}_link_{i}"
        origin = joint.find("origin")
        placement = {} if origin is None else origin.attrib
        xyz = np.fromstring(placement.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(placement.get("rpy", "0 0 0"), sep=" ")
        axis_element = joint.find("axis")
        direction = np.fromstring(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" ")
        if (joint.get("type") != "revolute" or joint.find("parent").get("link") != parent
                or joint.find("child").get("link") != child or not np.allclose(rpy, 0, atol=1e-14)
                or not np.allclose(direction, axis, rtol=0, atol=1e-14)
                or (i > 1 and abs(xyz[1]) > 1e-14)):
            raise ValueError(f"full-pose enumeration requires the Firefly planar joint layout at {child}")
        offsets.append(xyz)
        parent = child
    return np.asarray(offsets)


def angle_values(angle, lower, upper):
    """All 2π lifts inside one bounded actuator interval, including its endpoints."""
    first = int(np.ceil((lower - angle - EPS) / TAU))
    last = int(np.floor((upper - angle + EPS) / TAU))
    return [float(np.clip(angle + TAU * k, lower, upper)) for k in range(first, last + 1)]


def wrist_family_pitches(center, p3, p4, tool, phase, bend, lower, upper, seed_pitch):
    """Split the J6 phase interval wherever elbow reach or a planar joint stop changes feasibility."""
    def rotate(v, p):
        c, s = np.cos(p), np.sin(p)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])

    low, high = sorted(phase - np.sign(bend) * np.array([lower[5], upper[5]]))
    edges = [low, high]
    def circle_intersections(w, a, radius):
        # |w - Ry(p) a| = radius gives A cos(p) + B sin(p) = C in the XZ plane.
        x, y = w @ a, w[0] * a[1] - w[1] * a[0]
        amplitude = np.hypot(x, y)
        if amplitude < 1e-16:
            return
        cosine = (w @ w + a @ a - radius**2) / (2 * amplitude)
        if abs(cosine) <= 1 + EPS:
            origin, span = np.arctan2(y, x), np.arccos(np.clip(cosine, -1., 1.))
            for root in (origin - span, origin + span):
                edges.extend(angle_values(root, low, high))

    l3, l4 = np.linalg.norm(p3), np.linalg.norm(p4)
    for radius in (abs(l3 - l4), l3 + l4):
        circle_intersections(center, tool, radius)
    for j2 in (lower[1], upper[1]):
        circle_intersections(center - rotate(p3, j2), tool, l4)
    for j3 in (lower[2], upper[2]):
        circle_intersections(center, tool, np.linalg.norm(p3 + rotate(p4, j3)))
    for j4 in (lower[3], upper[3]):
        circle_intersections(center, tool + rotate(p4, -j4), l3)
    edges = np.unique(edges)
    # An isolated feasible point may lie exactly at an edge. Each open interval
    # has constant feasibility for a given elbow. Retain the seed and extra
    # representatives for downstream collision and path checks.
    inside = (edges[1:] + edges[:-1]) / 2
    seeded = [] if seed_pitch is None else angle_values(seed_pitch, low, high)
    return np.unique(np.r_[edges, inside, np.linspace(low, high, 9), seeded])


def planar_branches(flange, offsets, lower, upper, *, seed_yaw=None, seed_pitch=None):
    """Enumerate a flange pose expressed in its arm base. Report incomplete singular families."""
    solutions, rejected, singular = [], [], []
    sensitive_wrist = False
    p1, p2, p3, p4, p5, p6 = offsets
    length1, length2 = np.linalg.norm(p3), np.linalg.norm(p4)
    if min(length1, length2) < EPS:
        raise ValueError("full-pose enumeration requires two nonzero planar arm lengths")
    tilt1, tilt2 = np.arctan2(p3[0], p3[2]), np.arctan2(p4[0], p4[2])
    z_axis = flange.rotation[:, 2]
    # Flange Z is unchanged by J6. Removing the Z part of its offset puts
    # the wrist point in the J2..J4 plane (the X part is unchanged by J5).
    wrist = flange.translation - p1 - p6[2] * z_axis
    radial_position = np.hypot(wrist[0], wrist[1])
    yaws = []
    if radial_position > 1e-12:
        yaw0 = np.arctan2(wrist[1], wrist[0])
        yaws.extend((yaw0, yaw0 + np.pi))
    if radial_position <= 1e-9:
        # Every yaw defines a possible arm plane on the base axis. Solve the
        # remaining joints exactly for representative planes, including the seed.
        singular.append("base_yaw_underdetermined")
        yaws.extend(np.linspace(lower[0], upper[0], 9))
        if seed_yaw is not None:
            yaws.append(np.clip(seed_yaw, lower[0], upper[0]))
    for yaw in np.unique(yaws):
        rz = pin.rpy.rpyToMatrix(0., 0., yaw)
        local_rotation = rz.T @ flange.rotation
        direction = local_rotation[:, 2]
        radial = np.hypot(direction[0], direction[2])
        # atan2 of the two small components is ill-conditioned well before
        # the exact pole. An empty branch list here cannot certify infeasibility.
        sensitive_wrist |= radial < np.cbrt(np.finfo(float).eps)
        wrist_angles = []
        if radial > 1e-12:
            pitch0 = np.arctan2(direction[0], direction[2])
            bend0 = np.arctan2(direction[1], radial)
            wrist_angles.extend(((bend0, pitch0), (np.pi - bend0, pitch0 + np.pi)))
        if radial <= 2e-8:
            bend = np.copysign(np.pi / 2, direction[1])
            if angle_values(bend, lower[4], upper[4]):
                singular.append("wrist_pitch_underdetermined")
                combined = local_rotation @ pin.rpy.rpyToMatrix(bend, 0., 0.)
                phase = np.arctan2(combined[0, 2], combined[0, 0])
                center = (rz.T @ wrist - p2)[[0, 2]]
                tool = (p5 + [p6[0], 0., 0.])[[0, 2]]
                pitches = wrist_family_pitches(center, p3[[0, 2]], p4[[0, 2]], tool,
                                               phase, bend, lower, upper, seed_pitch)
                wrist_angles.extend((bend, pitch) for pitch in pitches)
        for bend, pitch in wrist_angles:
            if not angle_values(bend, lower[4], upper[4]):
                continue
            ry = pin.rpy.rpyToMatrix(0., pitch, 0.)
            rx = pin.rpy.rpyToMatrix(-bend, 0., 0.)
            wrist_rotation = (rz @ ry @ rx).T @ flange.rotation
            roll = np.arctan2(wrist_rotation[1, 0], wrist_rotation[0, 0])
            point = rz.T @ wrist - p2 - ry @ (p5 + np.array([p6[0], 0., 0.]))
            radius = np.hypot(point[0], point[2])
            cosine = (radius**2 - length1**2 - length2**2) / (2 * length1 * length2)
            if abs(cosine) > 1 + 1e-12:
                rejected.append({"reason": "outside_planar_reach", "yaw_rad": float(yaw),
                                 "radius_m": float(radius), "reach_m": [abs(length1-length2), length1+length2]})
                continue
            if radius < 1e-12:
                singular.append("folded_shoulder_underdetermined")
                continue
            elbow = np.arccos(np.clip(cosine, -1., 1.))
            for angle in (elbow, -elbow):
                shoulder = (np.arctan2(point[0], point[2]) - tilt1
                            - np.arctan2(length2 * np.sin(angle), length1 + length2 * np.cos(angle)))
                j3 = angle + tilt1 - tilt2
                raw = np.array([yaw, shoulder, j3, pitch - shoulder - j3, bend, roll])
                lifts = [angle_values(v, lo, hi) for v, lo, hi in zip(raw, lower, upper)]
                blocked = [i for i, values in enumerate(lifts) if not values]
                if blocked:
                    # Show the equivalent angles nearest the centers of the allowed intervals.
                    nearest = raw + TAU * np.rint(((lower + upper) / 2 - raw) / TAU)
                    rejected.append({"reason": "joint_limits", "q_rad": nearest.tolist(),
                                     "joints": [int(i + 1) for i in blocked]})
                    continue
                for candidate in product(*lifts):
                    q = np.asarray(candidate)
                    if all(np.max(np.abs(q - previous)) > EPS for previous in solutions):
                        solutions.append(q)
    if not solutions and sensitive_wrist and "wrist_pitch_underdetermined" not in singular:
        singular.append("ill_conditioned_wrist")
    return solutions, rejected, sorted(set(singular))
