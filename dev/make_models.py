"""Generates the OBJ props shipped in content/models (run once; the .obj files are committed)."""
import math
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "content", "models")


def ring(radius, y, segments):
    return [(radius * math.cos(2 * math.pi * i / segments), y, radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]


def write_obj(name, profile, segments=16, comment=""):
    """Lathe a (radius, height) profile around the Y axis and cap both ends."""
    vertices, uvs, faces = [], [], []
    for radius, y in profile:
        vertices += ring(radius, y, segments)
    total = profile[-1][1] - profile[0][1]
    for row, (_, y) in enumerate(profile):
        for i in range(segments + 1):
            uvs.append((2.0 * i / segments, (y - profile[0][1]) / total))
    for row in range(len(profile) - 1):
        for i in range(segments):
            a, b = row * segments + i, row * segments + (i + 1) % segments
            c, d = a + segments, b + segments
            ta, tb = row * (segments + 1) + i, row * (segments + 1) + i + 1
            tc, td = ta + segments + 1, tb + segments + 1
            faces.append(((a, ta), (c, tc), (d, td), (b, tb)))
    bottom, top = len(vertices), len(vertices) + 1
    vertices += [(0.0, profile[0][1], 0.0), (0.0, profile[-1][1], 0.0)]
    cap_uv = len(uvs)
    uvs.append((0.5, 0.5))
    last = (len(profile) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append(((bottom, cap_uv), (i, cap_uv), (j, cap_uv)))
        faces.append(((top, cap_uv), (last + j, cap_uv), (last + i, cap_uv)))
    with open(os.path.join(OUT, name), "w", newline="\n") as handle:
        handle.write("# NEXUS-3D-EXTREME prop: %s\n" % comment)
        for v in vertices:
            handle.write("v %.5f %.5f %.5f\n" % v)
        for t in uvs:
            handle.write("vt %.5f %.5f\n" % t)
        for face in faces:
            handle.write("f " + " ".join("%d/%d" % (v + 1, t + 1) for v, t in face) + "\n")
    return len(vertices), len(faces)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("barrel", write_obj("barrel.obj", [(0.44, 0.0), (0.50, 0.04), (0.50, 0.30), (0.47, 0.33), (0.50, 0.36), (0.50, 0.64),
                                             (0.47, 0.67), (0.50, 0.70), (0.50, 0.96), (0.44, 1.0)], 18, "steel drum, unit height"))
    print("pylon", write_obj("pylon.obj", [(0.50, 0.0), (0.50, 0.06), (0.30, 0.07), (0.07, 0.96), (0.05, 1.0)], 14, "traffic pylon, unit height"))
