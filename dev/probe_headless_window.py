import sys, time
from ursina import *
from panda3d.core import loadPrcFileData
app = Ursina(development_mode=False, fullscreen=False, borderless=False, size=(960, 540), vsync=False, title='probe')
gsg = base.win.getGsg()
print('GL_VENDOR', gsg.getDriverVendor())
print('GL_RENDERER', gsg.getDriverRenderer())
print('GL_VERSION', gsg.getDriverVersion())
print('GLSL', gsg.getDriverShaderVersionMajor(), gsg.getDriverShaderVersionMinor())
print('supports glsl', gsg.getSupportsGlsl(), 'basic shaders', gsg.getSupportsBasicShaders())
print('win size', base.win.getSize(), 'aspect', window.aspect_ratio)
print('coord system', base.win.getGsg().getCoordinateSystem(), 'render cs', getDefaultCoordinateSystem() if 'getDefaultCoordinateSystem' in dir() else None)
from panda3d.core import get_default_coordinate_system, CS_yup_left, CS_zup_right
print('default cs', get_default_coordinate_system(), 'yup_left=', CS_yup_left, 'zup_right=', CS_zup_right)
e = Entity(model='cube', color=color.orange, position=(0, 0, 5), rotation=(20, 30, 0))
e2 = Entity(model='cube', color=color.azure, position=(2, 1, 5))
print('e2 panda pos', e2.getPos(), 'camera fwd', camera.forward, 'cam up', camera.up)
t = Text('PROBE HUD', position=(-.5, .4), scale=2)
frames = 0
def update():
    global frames
    frames += 1
    if frames == 5:
        base.screenshot(namePrefix='probe1.png', defaultFilename=False)
        print('screenshot saved')
    if frames == 8:
        application.quit()
app.run()
