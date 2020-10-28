import argparse
import time

from . import CTFModel, find_ctf


def main():
    default_amplitude_contrast = 0.07

    parser = argparse.ArgumentParser()
    parser.add_argument("mrc", help="input aligned mrc file")
    parser.add_argument(
        "--pixelsize", type=float, help="pixelsize in angstroms", required=True
    )
    parser.add_argument(
        "--voltage", type=int, help="accelerating voltage in kV", required=True
    )
    parser.add_argument(
        "--cs", type=float, help="spherical abberation in millimeters", required=True
    )
    parser.add_argument(
        "--amplitude_contrast",
        type=float,
        default=default_amplitude_contrast,
        help=f"(default = {default_amplitude_contrast})",
    )
    parser.add_argument(
        "--search-phase",
        help="search for an additional phase shift",
        action="store_true",
    )
    args = parser.parse_args()

    print("creating ctf model")
    start_time_model = time.time()
    ctf_model = CTFModel(
        args.mrc, args.pixelsize, args.voltage, args.cs, args.amplitude_contrast
    )
    print("searching for optimal values")
    start_time_search = time.time()
    z1, z2, angle_astig, phase_shift = find_ctf(ctf_model, args.search_phase)
    end_time = time.time()
    print(f"preprocessing time: {start_time_search - start_time_model:.2f}")
    print(f"search time: {end_time - start_time_search:.2f}")
    print(f"total time: {end_time - start_time_model:.2f}")
    print()
    print(f"defocus values: {z1:.4f}, {z2:.4f} microns")
    print(f"astigmatism angle: {angle_astig:.2f} degrees")
    print(f"phase shift: {phase_shift:.2f} degrees")
    print(f"cross correlation: {ctf_model.cross_correlation_score:.6f}")


if __name__ == "__main__":
    main()