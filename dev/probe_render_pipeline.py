import sys, time, math
import numpy as np
from ursina import *
from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexFormat, GeomVertexData,
    GeomEnums, InternalName, NodePath, Shader as PShader, PTA_LVecBase4f, LVecBase4f, Texture as PTexture,
    Camera as PCamera, PerspectiveLens, CullFaceAttrib, ColorBlendAttrib, TransparencyAttrib, BoundingVolume, OmniBoundingVolume)
from direct.filter.FilterManager import FilterManager

app = Ursina(development_mode=False, fullscreen=False, borderless=False, size=(960, 540), vsync=False, title='probe')

afmt = GeomVertexArrayFormat()
afmt.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
afmt.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
afmt.addColumn(InternalName.getColor(), 4, Geom.NTFloat32, Geom.CColor)
afmt.addColumn(InternalName.getTexcoord(), 2, Geom.NTFloat32, Geom.CTexcoord)
fmt = GeomVertexFormat(); fmt.addArray(afmt); fmt = GeomVertexFormat.registerFormat(fmt)
arr = fmt.getArray(0)
print('stride', arr.getStride(), [(str(arr.getColumn(i).getName()), arr.getColumn(i).getStart()) for i in range(arr.getNumColumns())])
VDT = np.dtype([('pos', '<f4', 3), ('nrm', '<f4', 3), ('col', '<f4', 4), ('uv', '<f4', 2)])
assert VDT.itemsize == arr.getStride()

def box_verts(center, size, col, flip=False):
    c = np.asarray(center, np.float32); h = np.asarray(size, np.float32) * 0.5
    normals = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], np.float32)
    verts = np.zeros(24, VDT); idx = []
    for f, n in enumerate(normals):
        a = np.array([n[1], n[2], n[0]]); b = np.cross(n, a)
        quad = [(-1,-1),(1,-1),(1,1),(-1,1)]
        for k,(s,t) in enumerate(quad):
            verts['pos'][f*4+k] = c + (n + a*s + b*t) * h
            verts['nrm'][f*4+k] = n
            verts['uv'][f*4+k] = ((s+1)/2,(t+1)/2)
        p = verts['pos'][f*4:f*4+4]
        cr = np.cross(p[1]-p[0], p[2]-p[0])
        ccw_rh = np.dot(cr, n) > 0
        tri = [0,1,2,0,2,3] if ccw_rh else [0,2,1,0,3,2]
        if flip: tri = [tri[0], tri[2], tri[1], tri[3], tri[5], tri[4]]
        idx += [f*4+i for i in tri]
    verts['col'] = col
    return verts, np.array(idx, np.uint32)

def make_node(name, verts, idx, dynamic=False):
    usage = Geom.UHDynamic if dynamic else Geom.UHStatic
    vdata = GeomVertexData(name, fmt, usage)
    vdata.uncleanSetNumRows(len(verts))
    memoryview(vdata.modifyArray(0)).cast('B')[:] = verts.tobytes()
    prim = GeomTriangles(usage); prim.setIndexType(GeomEnums.NTUint32)
    pa = prim.modifyVertices(); pa.uncleanSetNumRows(len(idx))
    memoryview(pa).cast('B')[:] = idx.astype(np.uint32).tobytes()
    geom = Geom(vdata); geom.addPrimitive(prim)
    gn = GeomNode(name); gn.addGeom(geom)
    return NodePath(gn), vdata, prim

VERT = '''#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex; in vec3 p3d_Normal; in vec4 p3d_Color; in vec2 p3d_MultiTexCoord0;
out vec3 v_world; out vec3 v_normal; out vec4 v_color; out vec2 v_uv;
void main(){ gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; v_world = (p3d_ModelMatrix * p3d_Vertex).xyz;
 v_normal = mat3(p3d_ModelMatrix) * p3d_Normal; v_color = p3d_Color; v_uv = p3d_MultiTexCoord0; }'''
FRAG = '''#version 140
uniform vec4 u_light_pos[4]; uniform vec4 u_light_col[4]; uniform vec3 u_sun_dir;
in vec3 v_world; in vec3 v_normal; in vec4 v_color; in vec2 v_uv; out vec4 fragColor;
void main(){ vec3 n = normalize(v_normal); float ndl = max(dot(n, -u_sun_dir), 0.0);
 vec3 lit = v_color.rgb * (0.25 + 0.75*ndl);
 for (int i=0;i<4;i++){ vec3 d = u_light_pos[i].xyz - v_world; float d2 = max(dot(d,d), 0.01);
   lit += v_color.rgb * u_light_col[i].rgb * (u_light_pos[i].w / d2) * max(dot(n, normalize(d)), 0.0); }
 fragColor = vec4(lit, 1.0); }'''
sh = PShader.make(PShader.SL_GLSL, VERT, FRAG)

root = NodePath('world_root'); root.reparentTo(scene)
v1, i1 = box_verts((-2, 0.5, 6), (1,1,1), (1,0.5,0.1,1))
v2, i2 = box_verts((2, 0.5, 6), (1,1,1), (0.1,0.6,1,1), flip=True)
v3, i3 = box_verts((0, -0.5, 6), (12,1,12), (0.5,0.5,0.5,1))
n1,_,_ = make_node('ccw_rh', v1, i1); n1.reparentTo(root)
n2,_,_ = make_node('cw_rh', v2, i2); n2.reparentTo(root)
n3,_,_ = make_node('floor', v3, i3); n3.reparentTo(root)
root.setShader(sh)
lp = PTA_LVecBase4f(); lc = PTA_LVecBase4f()
for i in range(4): lp.pushBack(LVecBase4f(0,0,0,0)); lc.pushBack(LVecBase4f(0,0,0,0))
lp[0] = LVecBase4f(0, 0.6, 6, 3.0); lc[0] = LVecBase4f(1.0, 0.2, 0.2, 0)
root.setShaderInput('u_light_pos', lp); root.setShaderInput('u_light_col', lc)
root.setShaderInput('u_sun_dir', (0.3, -0.8, 0.5))

camera.position = (0, 2.5, -1); camera.rotation_x = 15; camera.fov = 80
yaw = math.radians(30); pitch = math.radians(10)
print('fwd formula', (math.sin(yaw)*math.cos(pitch), math.sin(pitch), math.cos(yaw)*math.cos(pitch)))
camera.rotation_y = 30; camera.rotation_x = -10
print('camera fwd at yaw30 pitch10', camera.forward, 'right', camera.right)
camera.rotation_y = 0; camera.rotation_x = 15

# --- post process
PV = '''#version 140
uniform mat4 p3d_ModelViewProjectionMatrix; in vec4 p3d_Vertex; in vec2 p3d_MultiTexCoord0; out vec2 uv;
void main(){ gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; uv = p3d_MultiTexCoord0; }'''
PF = '''#version 140
uniform sampler2D tex; uniform float u_ab; in vec2 uv; out vec4 fragColor;
void main(){ vec2 d = (uv-0.5)*u_ab; float r = texture(tex, uv+d).r; float g = texture(tex, uv).g; float b = texture(tex, uv-d).b;
 float vig = 1.0 - 0.6*dot(uv-0.5, uv-0.5); fragColor = vec4(vec3(r,g,b)*vig, 1.0); }'''
fm = FilterManager(base.win, base.cam)
ctex = PTexture(); dtex = PTexture()
quad = fm.renderSceneInto(colortex=ctex, depthtex=dtex)
quad.setShader(PShader.make(PShader.SL_GLSL, PV, PF)); quad.setShaderInput('tex', ctex); quad.setShaderInput('u_ab', 0.03)
buf = fm.buffers[-1]
print('buffer', buf, 'num DR', buf.getNumDisplayRegions())
# viewmodel layer inside the same offscreen buffer
vm_root = NodePath('vm_root')
vm_cam_node = PCamera('vm_cam'); vlens = PerspectiveLens(); vlens.setFov(60); vlens.setAspectRatio(window.aspect_ratio); vlens.setNearFar(0.01, 50)
vm_cam_node.setLens(vlens); vm_cam = vm_root.attachNewNode(vm_cam_node)
dr = buf.makeDisplayRegion(); dr.setSort(10); dr.setClearDepthActive(True); dr.setClearColorActive(False); dr.setCamera(vm_cam)
vv, vi = box_verts((0.25, -0.2, 0.7), (0.1, 0.12, 0.6), (0.2,0.9,0.3,1))
nvm,_,_ = make_node('vm', vv, vi); nvm.reparentTo(vm_root); vm_root.setShader(sh)
vm_root.setShaderInput('u_light_pos', lp); vm_root.setShaderInput('u_light_col', lc); vm_root.setShaderInput('u_sun_dir', (0.3,-0.8,0.5))

# dynamic buffer
pv, pi = box_verts((0, 1.5, 6), (0.3,0.3,0.3), (1,1,0,1))
ndyn, dyn_vdata, dyn_prim = make_node('dyn', pv, pi, dynamic=True); ndyn.reparentTo(root)
ndyn.node().setBounds(OmniBoundingVolume()); ndyn.node().setFinal(True)

Text('HUD OK', position=(-.6, .4), scale=2)
frames = 0
def update():
    global frames
    frames += 1
    pv2 = pv.copy(); pv2['pos'][:, 0] += math.sin(frames*0.5)*1.0
    memoryview(dyn_vdata.modifyArray(0)).cast('B')[:] = pv2.tobytes()
    lp[0] = LVecBase4f(math.sin(frames*0.3)*2, 0.6, 6, 3.0)
    if frames == 6:
        base.screenshot(namePrefix='probe2.png', defaultFilename=False)
    if frames == 9:
        application.quit()
app.run()
