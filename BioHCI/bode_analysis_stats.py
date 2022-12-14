import numpy as np
import os

def main():
    print(f"\nBode Analysis and Phase Analysis comparison\n")

    path_bode_ambient = "Resources/CTS_EICS_2020/Bode/Ambient"
    path_bode_noise = "Resources/CTS_EICS_2020/Bode/Noise"
    path_phase_ambient = "Resources/CTS_EICS_2020/Phase/Ambient"
    path_phase_noise = "Resources/CTS_EICS_2020/Phase/Noise"

    bode_ambient_fnames_ls, bode_ambient_std_ls = read_dir(path_bode_ambient)
    bode_noise_fnames_ls, bode_noise_std_ls = read_dir(path_bode_noise)
    phase_ambient_fnames_ls, phase_ambient_std_ls = read_dir(path_phase_ambient)
    phase_noise_fnames_ls, phase_noise_std_ls = read_dir(path_phase_noise)

def read_dir(path: str):
    std_ls = []
    fname_ls = []

    directory = os.path.abspath(os.path.join(os.pardir, path))
    print(f"\nDirectory: {path}")

    for button_csv in os.listdir(directory):
        if ".csv" in button_csv:
            file_path = os.path.join(directory, button_csv)
            # print(f"{file_path}")

            button_data = np.genfromtxt(file_path, delimiter=',')
            # print(button_data)
            std = np.std(button_data)
            std_ls.append(std)
            fname_ls.append(button_csv)

    print(f"{fname_ls}")
    print(f"{std_ls}")
    print(f"{sum(std_ls)/len(std_ls)}")
    return fname_ls, std_ls

if __name__ == "__main__":
    main()
