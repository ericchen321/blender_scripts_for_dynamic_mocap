# Author: Guanxiong


import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np


class Constants:
    r"""
    Constants used in the script.
    """
    csv_dirpath = "/home/eric/ssl/dynamic_mocap/data/eccv2020/raw_data/"
    subject_id = 2
    motion_type = "fast_walk"
    trial_id = "1"
    max_num_frames = 136
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

    return forces[:frame_idx], cops[:frame_idx]


def plot_cops(
    cops: np.ndarray,
    filepath: str):
    r"""
    Plot center of pressures of the specified foot on a plate.
    """
    matplotlib.rcParams['figure.dpi'] = 300
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = 1000 / float(dpi), 1000 / float(dpi)
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = ["left", "right"]
    colors = ["b", "r"]
    for i in range(Constants.num_feet):
        ax.plot(
            cops[:, i, 0], cops[:, i, 1],
            color = colors[i],
            marker = "o",
            linestyle = "dashed",
            linewidth = 0.5,
            markersize = 2,
            label = labels[i])
    
    fig.suptitle(f"CoP trajectories")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    plt.tight_layout()

    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved plot to {filepath}.")


if __name__ == "__main__":
    # read CoPs
    csv_filepath = f"{Constants.csv_dirpath}/" + \
        f"{Constants.subject_id:02}/" + \
        f"proband{Constants.subject_id}_{Constants.motion_type}_{Constants.trial_id}.csv"
    forces, cops = read_force_plate_data(
        csv_filepath,
        Constants.max_num_frames,
        Constants.num_subframes_per_frame,
        Constants.num_header_rows)

    # plot
    plot_cops(cops, "plots/cop_trajectory.png")