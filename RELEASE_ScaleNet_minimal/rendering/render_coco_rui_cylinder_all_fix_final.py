import argparse
import itertools
import os
import os.path as osp
import sys
import time

import bpy
import bpy_extras
import numpy as np
from mathutils import Vector

# Don't forget:
# - Set the renderer to Cycles
# - A ground plane set as shadow catcher
# - The compositing nodes should be [Image, RenderLayers] -> AlphaOver -> Composite
# - The world shader nodes should be Sky Texture -> Background -> World Output
# - Set a background image node

if ".blend" in os.path.realpath(__file__):
    # If the .py has been packaged in the .blend
    curdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
else:
    curdir = os.path.dirname(os.path.realpath(__file__))


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def setScene():
    bpy.data.scenes["Scene"].cycles.film_transparent = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        devices = prefs.get_devices()
        if devices:
            for device in devices:
                if device.type == "CUDA":
                    device.use = True
                    break

        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.use_denoising = True
        print("GPU render!")
    except:
        print("CPU render!")


def getVertices(obj, world=False, first=False):
    """Get the vertices of the object."""
    if first:
        print("-----getVertices")
    vertices = []
    if obj.data:
        if world:
            vertices.append([obj.matrix_world @ x.co for x in obj.data.vertices])
        else:
            vertices.append([x.co for x in obj.data.vertices])
    for idx_child, child in enumerate(obj.children):
        vertices.extend(getVertices(child, world=world))
        print(idx_child)
    return vertices


def getObjBoundaries(obj):
    """Get the object boundary in image space."""
    cam = bpy.data.objects["Camera"]
    scene = bpy.context.scene
    list_co = []
    vertices = getVertices(obj, world=True, first=True)
    for coord_3d in itertools.chain(*vertices):
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, coord_3d)
        list_co.append([x for x in co_2d])
    list_co = np.asarray(list_co)[:, :2]
    retval = list_co.min(axis=0).tolist()
    retval.extend(list_co.max(axis=0).tolist())
    return retval


def changeVisibility(obj, hide):
    """Hide or show object in render."""
    obj.hide_set(hide)
    obj.hide_viewport = hide
    for child in obj.children:
        changeVisibility(child, hide)


def setCamera(pitch, roll, hfov, vfov, imh, imw, cam_pos=(0, 0, 1.6)):
    # Set camera parameters
    # uses a 35mm camera sensor model
    print(
        "=====setting camera: pitch, roll, hfov, vfov, imh, imw, cam_pos=(0, 0, 1.6)",
        pitch,
        roll,
        hfov,
        vfov,
        imh,
        imw,
        cam_pos,
    )
    bpy.data.cameras["Camera"].sensor_width = 36
    cam = bpy.data.objects["Camera"]
    cam.location = Vector(cam_pos)
    cam.rotation_euler[0] = -pitch + 90.0 * np.pi / 180
    cam.rotation_euler[1] = -roll
    cam.rotation_euler[2] = 0
    if imh > imw:
        bpy.data.cameras["Camera"].angle = vfov
    else:
        bpy.data.cameras["Camera"].angle = hfov
    bpy.data.scenes["Scene"].render.resolution_x = imw
    bpy.data.scenes["Scene"].render.resolution_y = imh
    bpy.data.scenes["Scene"].render.resolution_percentage = 100
    bpy.context.view_layer.update()


def setObjectToImagePosition(object_name, ipv, iph):
    """insertion point vertical and horizontal (ipv, iph) in relative units."""
    bpy.context.view_layer.update()

    cam = bpy.data.objects["Camera"]

    # Get the 3D position of the 2D insertion point
    # Get the viewpoint 3D coordinates
    frame = cam.data.view_frame(scene=bpy.context.scene)
    frame = [cam.matrix_world @ corner for corner in frame]

    # Perform bilinear interpolation
    top_vec = frame[0] - frame[3]
    bottom_vec = frame[1] - frame[2]
    top_pt = frame[3] + top_vec * iph
    bottom_pt = frame[2] + bottom_vec * iph
    vertical_vec = bottom_pt - top_pt
    unit_location = top_pt + vertical_vec * ipv

    # Find the intersection with the ground plane
    obj_direction = unit_location - cam.location
    length = -cam.location[2] / obj_direction[2]

    # Set the object location
    if len(bpy.data.objects[object_name].children) == 0:
        bpy.data.objects[object_name].location = cam.location + obj_direction * length
    else:
        for child_obj in bpy.data.objects[object_name].children:
            child_obj.location = cam.location + obj_direction * length

    bpy.context.view_layer.update()
    print(f"setObjectToImagePosition: {bpy.data.objects[object_name].location}")


def changeBackgroundImage(bgpath, size):
    if "background" in bpy.data.images:
        previous_background = bpy.data.images["background"]
        bpy.data.images.remove(previous_background)

    img = bpy.data.images.load(filepath=bgpath)
    img.name = "background"

    img.scale(size[0], size[1])

    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        if isinstance(node, bpy.types.CompositorNodeImage):
            node.image = img
            break
    else:
        raise Exception("Could not find the background image node!")


def setParametricSkyLighting(theta, phi, t):
    """Use the Hosek-Wilkie sky model"""

    # Compute lighting direction
    x = np.sin(theta) * np.sin(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.cos(theta)

    # Remove previous link to Background and link it with Sky Texture
    link = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].links[0]
    bpy.data.worlds["World"].node_tree.links.remove(link)
    bpy.data.worlds["World"].node_tree.links.new(
        bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].outputs["Color"],
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0],
    )
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0

    # Set Hosek-Wilkie sky texture
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sky_type = "HOSEK_WILKIE"
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_direction = Vector(
        (x, y, z),
    )
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].turbidity = t
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].ground_albedo = 0.3

    bpy.data.objects["Sun"].rotation_euler = Vector((theta, 0, -phi + np.pi))
    bpy.data.lights["Sun"].shadow_soft_size = 0.03
    bpy.data.lights["Sun"].energy = 4
    bpy.data.objects["Sun"].hide_set(False)


def setIBL(path, phi):
    """Use an IBL to light the scene"""

    # Remove previous IBL
    if "envmap" in bpy.data.images:
        previous_background = bpy.data.images["envmap"]
        bpy.data.images.remove(previous_background)

    img = bpy.data.images.load(filepath=path)
    img.name = "envmap"

    # Remove previous link to Background and link it with Environment Texture
    link = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].links[0]
    bpy.data.worlds["World"].node_tree.links.remove(link)
    bpy.data.worlds["World"].node_tree.links.new(
        bpy.data.worlds["World"]
        .node_tree.nodes["Environment Texture"]
        .outputs["Color"],
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0],
    )
    bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = img
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2.0
    bpy.data.worlds["World"].node_tree.nodes["Mapping"].rotation = Vector(
        (0, 0, phi + np.pi / 2),
    )

    bpy.data.objects["Sun"].hide_set(True)


def performRendering(
    k,
    suffix="",
    subfolder="render",
    close_blender=False,
    tmp_code="",
):
    # redirect output to log file
    logfile = "blender_render.log"
    open(logfile, "a").close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # do the rendering
    os.makedirs(os.path.join(curdir, subfolder), exist_ok=True)
    imgpath = os.path.join(curdir, f"{subfolder}/{k}{suffix}_{tmp_code}.png")
    bpy.data.scenes["Scene"].render.filepath = imgpath
    bpy.ops.render.render(write_still=True)

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

    print(">>> Rendered file in " + imgpath)

    if close_blender:
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    # parser = argparse.ArgumentParser(description="Rui's Scale Estimation Network Training")
    parser.add_argument("-img_path", type=str, default="tmp", help="")
    parser.add_argument("-tmp_code", type=str, default="iamgroot", help="")
    parser.add_argument("-npy_path", type=str, default="iamgroot", help="")
    parser.add_argument(
        "-if_grid",
        type=bool,
        default=True,
        help="if_render grid of cylinders",
    )
    parser.add_argument("-H", type=int, help="")
    parser.add_argument("-W", type=int, help="")
    parser.add_argument("-pitch", type=float, default=0.9, help="")
    parser.add_argument("-fov_v", type=float, default=0.9, help="")
    parser.add_argument("-fov_h", type=float, default=0.9, help="")
    parser.add_argument("-cam_h", type=float, default=0.9, help="")
    parser.add_argument("-insertion_points_x", type=float, default=-1, help="")
    parser.add_argument("-insertion_points_y", type=float, default=-1, help="")
    opt = parser.parse_args()

    img_path = opt.img_path
    tmp_code = opt.tmp_code
    imh = opt.H
    imw = opt.W
    insertion_points_x, insertion_points_y = (
        opt.insertion_points_x,
        opt.insertion_points_y,
    )
    pitch = opt.pitch
    roll = 0.0
    hfov = opt.fov_h
    vfov = opt.fov_v
    h_cam = opt.cam_h

    setScene()

    insertion_points = [
        (insertion_points_y, insertion_points_x),
    ]
    if insertion_points_x == -1:
        insertion_points_xy_list = np.load(
            osp.join(opt.npy_path, "tmp_insert_pts_%s.npy" % tmp_code),
        )
        insertion_points = [(item[1], item[0]) for item in insertion_points_xy_list]

    bbox_hs_list = np.load(osp.join(opt.npy_path, "tmp_bbox_hs_%s.npy" % tmp_code))

    changeBackgroundImage(img_path, (imw, imh))
    setParametricSkyLighting(np.pi / 4, np.pi / 8, 3)

    object_name = "chair"
    all_obj_names = ["Cone", "chair", "Cylinder"]

    setCamera(-pitch, roll, hfov, vfov, imh, imw, cam_pos=(0, 0, h_cam))
    for obj in all_obj_names:
        changeVisibility(bpy.data.objects[obj], hide=True)

    src_obj = bpy.data.objects[object_name]
    created_objects = []
    for idx, ((ipv, iph), bbox_h) in enumerate(zip(insertion_points, bbox_hs_list)):
        print("===============================", idx)
        # Rotate the object randomly about its y-axis
        # (just for the sake of example, won't do anything on a torus, of course...)
        new_obj = src_obj.copy()
        new_obj.name = "%s_%d" % (object_name, idx)
        changeVisibility(bpy.data.objects[new_obj.name], hide=False)
        new_obj.data = src_obj.data.copy()
        bpy.context.collection.objects.link(new_obj)
        setObjectToImagePosition(new_obj.name, ipv / imh, iph / imw)
        created_objects.append(new_obj)

        # Check if object is inside the frame. If not, resize it a tad
        original_scale = new_obj.scale.copy()

        obj_bounds = getObjBoundaries(new_obj)
        print("---obj_bounds", obj_bounds)

        print("Original scale", new_obj.scale)
        print("Original dimensions", new_obj.dimensions)
        new_obj.dimensions = new_obj.dimensions * bbox_h
        print(
            "After scale, dimensions, bbox_h",
            new_obj.scale,
            new_obj.dimensions,
            bbox_h,
        )
        print("After location", new_obj.location)

        # Set on ground, useful if scaled
        vertices = getVertices(new_obj, world=True)
        dist_to_ground = min(v.z for v in itertools.chain(*vertices))
        new_obj.location[2] -= dist_to_ground
        print("Moved the object in Z-axis by", dist_to_ground)
        new_obj.rotation_euler[2] = np.pi / 4.0
        print("After location 2", new_obj.location)
        bpy.context.view_layer.update()

    # Remove main objects
    for obj in all_obj_names:
        bpy.data.objects.remove(bpy.data.objects[obj], do_unlink=True)
    ts = time.time()
    performRendering(
        "render_{}".format("all"),
        close_blender=len(insertion_points) == 1,
        tmp_code=tmp_code,
    )
    print(f"Rendering done in {time.time() - ts:0.3f}s")
    print("Camera location", bpy.data.objects["Camera"].location)
    print("------------------------------")
