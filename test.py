from uav_coverage_team3umd import main
import numpy as np


def parse_coords(coords_string: str) -> np.array:
    pairs = coords_string.strip().replace(" ", "").split(";")
    pairs = [[float(e) for e in pair.split(",")] for pair in pairs]
    coord_pairs = [np.array(pair) for pair in pairs]
    coord_pairs.append(coord_pairs[0])
    coord_pairs = np.array(coord_pairs)

    return coord_pairs


if __name__ == "__main__":
    """
    main(
        np.array((-79.7, 39.8)),
        parse_coords(
            "-80, 40; -80.05, 40.22; -79.703, 40.186; -80.02, 40.43; -79.968, 41.023; -79.818, 41.064; -79.66, 41.01; -79.698, 40.827; -79.792, 40.8; -79.79, 40.92; -79.895, 40.896; -79.902, 40.63; -79.578, 40.623; -79.558, 40.998; -79.448, 41.007; -79.4, 40.796; -79.49, 40.465; -79.853, 40.468; -79.493, 40.24; -79.475, 40.027"
        ),
        6,
        100,
        25,
        1000,
        arbitrary_large_number=50,
        ear_clipping=True,
    )
    """
    """
    # Sample Test 1
    main(
        np.array((39.65, -87.75)),
        np.array([(39.6, -87.75), (39.58, -87.75), (39.58, -87.725), (39.6, -87.725)]),
        3,
        15,
        25,
        75,
    )
    print("Test 1 completed")
    """
    # Sample Test 2
    main(
        np.array((39.65, -87.75)),
        np.array([(39.65, -87.75), (39.67, -87.75), (39.67, -87.7), (39.65, -87.7)]),
        5,
        5,
        25,
        25,
    )
    print("Test 2 completed")

    # Sample Test 3
    main(
        np.array((39.65, -87.75)),
        np.array([(39.6, -87.75), (39.55, -87.735), (39.45, -87.7), (39.6, -87.7)]),
        6,
        25,
        25,
        150,
    )
    print("Test 3 completed")
