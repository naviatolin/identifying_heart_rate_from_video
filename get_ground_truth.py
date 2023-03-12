import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def get_bvp(file):
    """
    Gets the ground truth bvp values from txt files
    """
    with open(file, "r") as truth:
        lines = truth.readlines()

        bvp = []
        for i in lines:
            l = i.split()
            l = [j.replace("\n","") for j in l]
            l = [float(j) for j in l]
            bvp.append(l)

    bvp = np.array(bvp).flatten()
    return bvp

def get_bpm_from_bvp(bvp, fps):
    """
    Takes a block of bvp data and converts that block into bvp
    """
    peaks = find_peaks(bvp)[0]
    num_peaks = len([j for j in peaks if bvp[j] > 0.4])
    frames = len(bvp)
    frames_to_sec = frames/fps
    bpm = num_peaks / frames_to_sec * 60

    ### Uncomment to plot
    # time = np.linspace(0,np.round(frames/fps),frames)
    # plt.plot(time, bvp)
    # plt.show()
    return bpm

if __name__ == "__main__":
    file = 'ground_truth.txt'
    bvp = get_bvp(file)
    prev_bpm = 0
    prev_bpm2 = 0
    # This window averages (we won't need it for training, but will use for video stream)
    for start in range(25):
        start_frame, end_frame = start*60, 60*start+60
        
        fps = 29.26
        bvp_block = bvp[start_frame: end_frame]
        bpm = get_bpm_from_bvp(bvp_block, fps)

        # calculate avg here
        if prev_bpm == 0:
            # set the current one to now be the new prev
            prev_bpm = bpm
            continue
        elif prev_bpm2 == 0:
            prev_bpm = bpm
            prev_bpm2 = prev_bpm
            continue
        else:
            ave_bpm = (prev_bpm + prev_bpm2 + bpm)/3
            prev_bpm = bpm
            prev_bpm2 = prev_bpm

        print("BPM: ", ave_bpm)