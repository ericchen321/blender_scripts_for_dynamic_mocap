# Author: Guanxiong

import bpy
import csv
import numpy as np
import mathutils


class Constants:
    r"""
    Constants used in the script.
    """
    csv_dirpath = "/home/eric/ssl/mujoco-2.1.0/dynamic_mocap/eccv_2020/raw_data/"
    subject_id = "13"
    motion_type = "run"
    trial_id = "1"
    max_num_frames = 250
    num_subframes_per_frame = 10
    num_header_rows = 5
    num_feet = 2 #  just left/right foot so far
    epsi = 1e-3
    length_scaling = 1000.0 # from mm to m
    arrow_names = ["arrow_left", "arrow_right"]
    arrow_colors = [
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 1.0])]
    arrow_radius = 0.15
    min_arrow_length = 0.1
    arrow_force_scaling = 500.0 # force of this magnitude will have arrow length of 1
    human_name = "Armature"
    plate_names = ["force_plate_left", "force_plate_right"]
    plate_colors = [
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 1.0])]
    # plate dimensions from https://www.amti.biz/product/bms400600/#specifications;
    # not sure though if it's the model used in the paper
    plate_dims = np.array([0.8, 0.3, 0.0825])
    ball_names = ["ball_left", "ball_right"]
    ball_colors = [
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 1.0])]
    ball_radius = 0.1


def read_force_plate_data(
    csv_filepath: str,
    max_num_frames: int,
    num_subframes_per_frame: int,
    num_header_rows: int):
    r"""
    Read forces and CoP positions from a csv file. Return forces
    and CoP positions of shape (max_num_frames, num_feet, 3).
    """
    forces = np.zeros((max_num_frames, Constants.num_feet, 3))
    cops = np.zeros((max_num_frames, Constants.num_feet, 3))
    with open(csv_filepath, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        frame_idx = 0
        for row_idx, row in enumerate(csv_reader):
            if row_idx < num_header_rows:
                pass
            else:
                subframe_idx = int(row[1])
                if subframe_idx == 0:
                    # check if we're reading a new frame. If so, build data struct to
                    # store forces and CoPs for each subframe
                    forces_per_frame = np.zeros(
                        (num_subframes_per_frame, Constants.num_feet, 3))
                    cops_per_frame = np.zeros(
                        (num_subframes_per_frame, Constants.num_feet, 3))
                # store forces and CoPs for this subframe
                forces_per_frame[subframe_idx, 0] = [float(row[i]) for i in range(2, 5)]
                forces_per_frame[subframe_idx, 1] = [float(row[i]) for i in range(11, 14)]
                cops_per_frame[subframe_idx, 0] = [float(row[i]) for i in range(8, 11)]
                cops_per_frame[subframe_idx, 1] = [float(row[i]) for i in range(17, 20)]
                if subframe_idx == num_subframes_per_frame - 1:
                    # compute average forces and CoPs from all subframes
                    forces[frame_idx, 0] = np.mean(forces_per_frame[:, 0], axis=0)
                    forces[frame_idx, 1] = np.mean(forces_per_frame[:, 1], axis=0)
                    cops[frame_idx, 0] = np.mean(cops_per_frame[:, 0], axis=0)
                    cops[frame_idx, 1] = np.mean(cops_per_frame[:, 1], axis=0)
                    frame_idx += 1
        print(f"Read {frame_idx} frames of data.")

    # flip forces (so measured as "force applied from foot to ground")
    forces = -1.0 * forces

    # convert data into proper units
    cops = cops / Constants.length_scaling

    return forces, cops


def get_corner_pos_object(name: str) -> np.ndarray:
    r"""
    Get the object-frame position of each corner of the bounding box
    of the object with the given name. Return an (8, 3) array.
    """
    ob = bpy.data.objects[name]
    bbox = ob.bound_box
    corner_pos_object = np.zeros((8, 3))
    for i, corner in enumerate(bbox):
        corner_pos_object[i] = mathutils.Vector(corner)
    return corner_pos_object


def get_corner_pos_world(name: str) -> np.ndarray:
    r"""
    Get the world-frame position of each corner of the bounding box
    of the object with the given name. Return an (8, 3) array.
    """
    ob = bpy.data.objects[name]
    bbox = ob.bound_box

    # get world-frame position of each corner of the bounding box
    # by CodeManX from https://blender.stackexchange.com/questions/
    # 8459/get-blender-x-y-z-and-bounding-box-with-script
    # As visualized by @zeffii from
    # https://blender.stackexchange.com/questions/32283/what-are-all-values-in-bound-box,
    # we pick the 1st corner's z value as the height of the bounding box. So really
    # we only need one corner's world-frame position. But get the rest anyway
    # for debugging purposes.
    corner_pos_world = np.zeros((8, 3))
    for i, corner in enumerate(bbox):
        corner_pos_world[i] = ob.matrix_world @ mathutils.Vector(corner)
    return corner_pos_world


def get_bbox_lower_z(name: str) -> np.ndarray:
    r"""
    Get the lower z position of the bounding box of the object
    with the given name.
    """
    corner_pos_world = get_corner_pos_world(name)
    height = corner_pos_world[0, 2]
    return np.array(height)


def set_object_location(pos: np.ndarray, name: str):
    r"""
    Set the location of the object with name `name` to `pos`.
    """
    ob = bpy.data.objects.get(name)
    ob.location = (pos[0], pos[1], pos[2])


def set_arrow_length(
    length: float,
    name: str) -> np.ndarray:
    r"""
    Set the length of the arrow with name `name` to `length`. If `length` is
    close to zero then we set the length to a min length to avoid complications.
    Return the actual length of the arrow.
    """
    ob = bpy.data.objects.get(name)
    if np.linalg.norm(length - 0.0) < Constants.epsi:
        ob.scale = (
            Constants.arrow_radius,
            Constants.arrow_radius,
            Constants.min_arrow_length)
        return np.array([Constants.min_arrow_length])
    else:
        ob.scale = (
            Constants.arrow_radius,
            Constants.arrow_radius,
            length)
        return np.array([length])


def get_foot_position(foot_name: str):
    r"""
    Get the world-frame position of the foot with name `foot_name`.
    Implementation adapted from work by @TLousky in
    # https://blender.stackexchange.com/questions/1299/
    """
    bone = bpy.data.objects["Armature"].pose.bones[foot_name]
    pos_bone = bone.id_data.matrix_world @ bone.matrix @ bone.location
    return np.array(pos_bone)


def compute_arrow_position(
    pos_arrow_start: np.ndarray,
    force: np.ndarray,
    arrow_length: np.ndarray) -> np.ndarray:
    r"""
    Compute the position of the arrow given the world-frame position
    of the point where the arrow starts, the force vector and the arrow
    length.
    """
    # normalize the force if it's not zero
    force_mag = np.linalg.norm(force)
    if force_mag < Constants.epsi:
        force_norm = force
    else:
        force_norm = force / np.linalg.norm(force)
    # scale the normalized force by half of arrow length
    vec = force_norm * arrow_length / 2.0
    # add the scaled force to the foot position
    pos_arrow = pos_arrow_start + vec
    return pos_arrow


def compute_arrow_rotation(
    pos_bone: np.ndarray,
    force: np.ndarray,
    arrow_length: np.ndarray) -> mathutils.Quaternion:
    r"""
    Compute the rotation of the arrow in quaternions.
    Implementation based on answer from "Brenel" on Blender Stack Exchange:
    https://blender.stackexchange.com/questions/1019/
    how-to-rotate-an-object-to-face-a-point
    """
    # normalize the force if it's not zero
    force_mag = np.linalg.norm(force)
    if force_mag < Constants.epsi:
        force_norm = force
    else:
        force_norm = force / np.linalg.norm(force)

    # compute the end pos of the arrow
    pos_end = pos_bone + force_norm * arrow_length

    dir = mathutils.Vector(pos_end - pos_bone)
    rotation_quat = dir.to_track_quat('Z', 'X')
    return rotation_quat


def set_arrow_rotation(
    quat: np.ndarray,
    name: str):
    r"""
    Set the rotation of the arrow with name `name` to `quat`.
    """
    ob = bpy.data.objects.get(name)
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = quat


def set_object_color(name: str, color: np.ndarray):
    r"""
    Set the color of the object with name `name`.
    # Implemented by "Raunaq" from
    # https://blender.stackexchange.com/questions/
    # 201874/how-to-add-a-color-to-a-generated-cube-within-a-python-script
    """
    # Create a material
    mat = bpy.data.materials.new("Blue")
    # Activate its nodes
    mat.use_nodes = True
    # Get the principled BSDF (created by default)
    principled = mat.node_tree.nodes['Principled BSDF']
    # Assign the color
    principled.inputs['Base Color'].default_value = color
    # Assign the material to the object
    ob = bpy.data.objects.get(name)
    ob.data.materials.append(mat)


def instantiate_plate(name: str, dims: np.ndarray, color: np.ndarray):
    r"""
    Instantiate a force plate in the scene.
    """
    # create object and set name + dims
    bpy.ops.mesh.primitive_cube_add(
        size = 1.0,
        align = 'WORLD',
        scale = dims)
    ob = bpy.context.active_object
    ob.name = name

    # color the object
    set_object_color(name, color)


def instantiate_arrow(name: str, color: np.ndarray):
    r"""
    Instantiate an arrow in the scene.
    """
    # create object and set name
    bpy.ops.mesh.primitive_cylinder_add(
        radius = Constants.arrow_radius,
        depth = 1)
    ob = bpy.context.active_object
    ob.name = name

    # color the object
    set_object_color(name, color)


def instantiate_ball(name: str, color: np.ndarray):
    r"""
    Instantiate a ball in the scene.
    """
    # create object and set name
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius = Constants.ball_radius,
        align = 'WORLD',
        location = (0, 0, 0))
    ob = bpy.context.active_object
    ob.name = name

    # color the object
    set_object_color(name, color)
    

def on_frame_change_pre(scene):
    r"""
    Handler function called before each frame change.
    """
    # get the frame idx
    frame = scene.frame_current

    # get the world-frame position of each foot
    pos_feet_world = np.zeros((Constants.num_feet, 3))
    pos_feet_world[0] = get_foot_position("mixamorig:LeftFoot")
    pos_feet_world[1] = get_foot_position("mixamorig:RightFoot")

    # update the arrow's length
    arrow_lengths = np.zeros((Constants.num_feet,))
    for i in range(Constants.num_feet):
        arrow_lengths[i] = set_arrow_length(
            np.linalg.norm(forces[frame, i]) / Constants.arrow_force_scaling,
            Constants.arrow_names[i])

    # compute the arrow's center pos from foot positions
    pos_arrows_world = np.zeros((Constants.num_feet, 3))
    for i in range(Constants.num_feet):
        pos_arrows_world[i] = compute_arrow_position(
            np.array([
                pos_feet_world[i, 0],
                pos_feet_world[i, 1],
                force_plate_z]),
            forces[frame, i],
            arrow_lengths[i])

    # update the arrow's direction and location
    rot_arrows = np.zeros((Constants.num_feet, 4))
    for i in range(Constants.num_feet):
        set_object_location(pos_arrows_world[i], Constants.arrow_names[i])
        rot_arrows[i] = compute_arrow_rotation(
            pos_feet_world[i],
            forces[frame, i],
            arrow_lengths[i])
        set_arrow_rotation(rot_arrows[i], Constants.arrow_names[i])

    # plot the spheres
    pos_balls_world = np.zeros((Constants.num_feet, 3))
    for i in range(Constants.num_feet):
        # NOTE: we assume CoP is horizontally aligned with the foot;
        # vertically aligned with the force plate
        pos_balls_world[i] = [
            pos_feet_world[i, 0], pos_feet_world[i, 1], force_plate_z]
        # set ball position
        set_object_location(
            pos_balls_world[i],
            Constants.ball_names[i])
    
    # update the plate's horizontal coordinates
    pos_plates = np.zeros((Constants.num_feet, 3))
    for i in range(Constants.num_feet):
        # compute plate location
        Oprime_CoP_world = cops[frame, i] # from plate-frame origin to CoP
        Oprime_O_world = -1.0 * get_corner_pos_object(
            Constants.plate_names[i])[1]
        pos_Oprime_world = pos_feet_world[i] - Oprime_CoP_world
        pos_plates[i] = pos_Oprime_world + Oprime_O_world
        set_object_location(
            np.array(
                [pos_plates[i, 0], pos_plates[i, 1], force_plate_z]),
            Constants.plate_names[i])


if __name__ == "__main__":
    # read forces and CoPs from csv file
    # TODO: read moments as well, maybe
    csv_filepath = f"{Constants.csv_dirpath}/" + \
        f"{Constants.subject_id}/" + \
        f"proband{Constants.subject_id}_{Constants.motion_type}_{Constants.trial_id}.csv"
    forces, cops = read_force_plate_data(
        csv_filepath,
        Constants.max_num_frames,
        Constants.num_subframes_per_frame,
        Constants.num_header_rows)
    
    # instantiate the force plates
    for i in range(Constants.num_feet):
        instantiate_plate(
            Constants.plate_names[i],
            Constants.plate_dims,
            Constants.plate_colors[i])

    # print corner positions
    corner_pos_world_init = get_corner_pos_world(
        Constants.plate_names[0])
    for i in range(8):
        print(corner_pos_world_init[i])
    
    # get the foot under which non-zero GRF is registered
    forces_mags = np.linalg.norm(
        np.linalg.norm(forces, axis = 2), axis = 0) # (num_feet,)
    max_force_bone_idx = np.argmax(forces_mags)
    
    # instantiate the force arrows
    for i in range(Constants.num_feet):
        instantiate_arrow(Constants.arrow_names[i], Constants.arrow_colors[i])

    # instantiate the CoP balls
    for i in range(Constants.num_feet):
        instantiate_ball(Constants.ball_names[i], Constants.ball_colors[i])

    # move the force plates coincide with the human on the z axis
    human_bbox_lower_z = get_bbox_lower_z(Constants.human_name)
    force_plate_z = human_bbox_lower_z - 0.5 * Constants.plate_dims[2]
    for i in range(Constants.num_feet):
        set_object_location(
            np.array([0, 0, force_plate_z]),
            Constants.plate_names[i])
    
    # register the handler
    bpy.app.handlers.frame_change_pre.append(on_frame_change_pre)
