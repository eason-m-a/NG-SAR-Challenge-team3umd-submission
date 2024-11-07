from tkinter import simpledialog
from tkinter import messagebox
import matplotlib.pyplot as plt
from earclip import cut_polygon
import numpy as np
import datetime
import os

COLORS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.75, 1, 0),
    (0, 0.75, 1),
    (1, 0, 0.75),
]
COLOR_STRINGS = ["red", "green", "blue", "light-red", "light-green", "light-blue"]
N = 500
SMALL = 0.0000001


def midpoint(points: np.array) -> np.array:
    s = np.size(points) / 2

    midx = np.sum(points[:, 0]) / s
    midy = np.sum(points[:, 1]) / s
    return np.array([midx, midy])


def angle_unit_vector(angle: float):
    rad = angle / 360 * 2 * np.pi
    return np.array([np.cos(rad), np.sin(rad)])


def line(p1: np.array, p2: np.array, c: str = "b", w=0.5):
    p1 = np.reshape(p1, (2, 1))
    p2 = np.reshape(p2, (2, 1))

    points = np.hstack((p1, p2))

    plt.plot(points[0, :], points[1, :], color=c, linewidth=w, solid_capstyle="round")


def points(points: np.array, c: str = "r"):
    p = np.reshape(points, (-1, 2))
    for i in range(p.shape[0]):
        plt.plot(p[i][0], p[i][1], marker="x", color=c)


def draw_polygon(coords: np.array, c=(0.5, 0.5, 0.5)):
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        line(p1, p2, w=2, c=c)


def draw_path(positions: np.array, color: tuple):
    for i in range(len(positions) - 1):
        line(positions[i], positions[i + 1], c=color, w=1)


def squeeze_pair(pair: np.array) -> np.array:
    return np.squeeze(np.asarray(pair))


def col_dot(a, b):
    a_reshape = np.reshape(a, (-1, 2))
    b_reshape = np.reshape(b, (-1, 2))
    return np.hstack(
        np.reshape(
            squeeze_pair(a_reshape[:, 0]) * squeeze_pair(b_reshape[:, 0]), (-1, 1)
        )
        + np.reshape(
            squeeze_pair(a_reshape[:, 1]) * squeeze_pair(b_reshape[:, 1]), (-1, 1)
        )
    )


def segment_intersect(l1: np.array, l2: np.array) -> np.array:

    def seg_intersect(
        a1: np.array, a2: np.array, b1: np.array, b2: np.array
    ) -> np.array:
        """
        Code modified from:
        https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/%7Erod/2500/notes/numpy-arrays/numpy-arrays.html
        This code is used to determine the intersection point of two lines,
        given two points (a1, a2) and (b1, b2) which lie on two respective lines
        """

        def perp(a):
            c0 = np.reshape(-a[:, 1], (-1, 1))
            c1 = np.reshape(a[:, 0], (-1, 1))
            return np.hstack((c0, c1))

        da = np.reshape(a2 - a1, (-1, 2))
        db = np.reshape(b2 - b1, (-1, 2))
        dp = np.reshape(a1 - b1, (-1, 2))
        dap = perp(da)

        denom = col_dot(dap, db)
        num = col_dot(dap, dp)

        div = num / denom.astype(float)
        div = np.reshape(div, (-1, 1))
        div[np.isnan(div)] = 0
        div[np.isinf(div)] = 999999999
        div[np.isneginf(div)] = -999999999

        c0 = np.reshape(db[:, 0], (-1, 1))
        c1 = np.reshape(db[:, 1], (-1, 1))
        mult = np.hstack((c0 * div, c1 * div))

        return mult + b1

    line1 = np.reshape(l1, (-1, 4))
    line2 = np.reshape(l2, (-1, 4))
    a1 = squeeze_pair(line1[:, 0:2])
    a2 = squeeze_pair(line1[:, 2:4])
    b1 = squeeze_pair(line2[:, 0:2])
    b2 = squeeze_pair(line2[:, 2:4])

    return seg_intersect(a1, a2, b1, b2)


def point_inside_line(line: np.array, point: np.array) -> bool:
    diff = get_line_diff(line)
    magnitude = np.reshape(np.linalg.norm(diff, axis=1), (1, -1))

    point_mag1 = np.linalg.norm(line[:, 0:2] - point, axis=1)
    point_mag2 = np.linalg.norm(line[:, 2:4] - point, axis=1)

    sum = np.reshape(point_mag1 + point_mag2, (1, -1))

    return sum < magnitude + SMALL


def find_closest(points: np.array, the_point: np.array) -> np.array:
    idx = (np.linalg.norm(points - the_point, axis=1)).argmin()
    return points[idx]


def get_line_coords(coords: np.array) -> np.array:
    shifted_up_one = np.roll(coords, shift=-1, axis=0)
    line_coords = np.hstack((coords, shifted_up_one))
    return line_coords


def get_line_diff(line: np.array) -> np.array:
    return line[:, 2:4] - line[:, 0:2]


def get_perpendicular(line_coords: np.array) -> np.array:
    diff = get_line_diff(line_coords)
    perp = np.roll(diff, shift=1, axis=1)
    perp[:, 0] *= -1

    norm = np.linalg.norm(perp, axis=1)
    norm = np.reshape(norm, (-1, 1))
    norm = np.repeat(norm, 2, axis=1)
    unit = perp / norm

    return unit


def find_closest_start(my_start: np.array, plausible_starts: np.array):
    plausible_pairs = np.reshape(plausible_starts, (-1, 2))
    big_start = np.repeat(
        np.reshape(my_start, (1, 2)), plausible_pairs.shape[0], axis=0
    )
    norms = np.linalg.norm(plausible_pairs - big_start, axis=1)

    idx = np.argmin(norms)
    closest_pair = plausible_pairs[idx, :]

    pair_num = idx

    return closest_pair, pair_num


def make_lines(
    start_pos: np.array,
    line_coords: np.array,
    angle_unit: np.array,
    scan_width: float,
    arbitrary_large_number: int,
) -> np.array:

    def get_intersection_validity(intersections: np.array):
        diff = get_line_diff(line_coords)

        magnitude = np.reshape(np.linalg.norm(diff, axis=1), (1, -1))

        line_coords1 = np.reshape(line_coords[:, 0:2], (1, -1))
        line_coords1 = np.resize(line_coords1, intersections.shape)
        line_coords2 = np.reshape(line_coords[:, 2:4], (1, -1))
        line_coords2 = np.resize(line_coords2, intersections.shape)

        diff1 = line_coords1 - intersections
        diff1 = np.reshape(diff1, (-1, 2))
        point_mag1 = np.linalg.norm(diff1, axis=1)
        point_mag1 = np.reshape(point_mag1, (-1, line_coords.shape[0]))
        diff2 = line_coords2 - intersections
        diff2 = np.reshape(diff2, (-1, 2))
        point_mag2 = np.linalg.norm(diff2, axis=1)
        point_mag2 = np.reshape(point_mag2, (-1, line_coords.shape[0]))

        sum = np.reshape(point_mag1 + point_mag2, (1, -1))
        magnitude = np.resize(magnitude, sum.shape)
        big_diff = sum - magnitude

        result = big_diff < SMALL
        result = np.reshape(result, (-1, line_coords.shape[0]))

        return result

    angle_unit_line = np.hstack((np.zeros((1, 2)), np.reshape(angle_unit, (1, -1))))
    inc_vector = get_perpendicular(angle_unit_line) * scan_width

    midp = midpoint(line_coords)
    if np.linalg.norm(start_pos - midp) < np.linalg.norm(start_pos + inc_vector - midp):
        inc_vector *= -1

    ABN = arbitrary_large_number

    # Hopefully 0.0001 of a meter (a milimeter) is within acceptable error
    starts = np.repeat(np.reshape(start_pos + 0.0001, (1, -1)), ABN, axis=0)
    displace = np.reshape(np.arange(ABN) - ABN // 2, (ABN, 1))
    displace = np.repeat(displace, 2, axis=1)
    displace = displace * inc_vector
    starts = starts + displace

    unit_displace = np.resize(angle_unit, starts.shape)
    unit_lines = np.hstack((starts, starts + unit_displace))

    for i in range(unit_lines.shape[0]):
        p1 = np.reshape(unit_lines[i][0:2], (1, -1))
        p1 = meters_to_lat_long(p1)
        p2 = np.reshape(unit_lines[i][2:4], (1, -1))
        p2 = meters_to_lat_long(p2)

    intersections = np.hstack(
        [
            np.reshape(
                segment_intersect(unit_lines, np.resize(l, unit_lines.shape)), (-1, 2)
            )
            for l in line_coords
        ]
    )

    inside_statuses = get_intersection_validity(intersections)
    inside_statuses = np.repeat(inside_statuses, 2, axis=1)

    line_points = intersections[inside_statuses]
    line_points = np.reshape(line_points, (-1, 2))

    return line_points


def simulate_coverage(
    angle: float,
    start: np.array,
    coords: np.array,
    scan_width: float,
    arbitrary_large_number: int,
):
    angle_unit = angle_unit_vector(angle)

    positions = [squeeze_pair(start)]

    line_coords = get_line_coords(coords)
    closest_start, line_num = find_closest_start(start, coords)

    lines = make_lines(
        closest_start, line_coords, angle_unit, scan_width, arbitrary_large_number
    )
    if np.size(lines) != 0:
        first = lines[0]
        closest_start, line_num = find_closest_start(first, coords)

    shifted_coords = np.roll(coords, shift=-line_num, axis=0)

    for c in shifted_coords:
        positions.append(c)
    positions.append(closest_start)

    i = 0
    while i < lines.shape[0]:
        pair1 = lines[i]
        pair2 = lines[min(i + 1, lines.shape[0] - 1)]

        last_pair = positions[-1]

        dist1 = np.linalg.norm(pair1 - last_pair)
        dist2 = np.linalg.norm(pair2 - last_pair)
        dist_between_1_2 = np.linalg.norm(pair1 - pair2)

        if dist_between_1_2 < SMALL:
            if i == lines.shape[0] - 1:
                positions.append(pair1)
                break

            lines = np.delete(lines, (i + 1), axis=0)
            continue
        elif dist1 < SMALL:
            lines = np.delete(lines, (i), axis=0)
            continue
        elif dist2 < SMALL:
            lines = np.delete(lines, (i + 1), axis=0)
            continue

        order = dist1 < dist2
        if order:
            positions.append(pair1)
            positions.append(pair2)
        else:
            positions.append(pair2)
            positions.append(pair1)

        i += 2

    positions = np.array(positions)
    position_lines = positions - np.roll(positions, shift=-1, axis=0)
    position_lines = position_lines[:-1]
    position_distances = np.linalg.norm(position_lines, axis=1)
    total_distance = np.sum(position_distances)

    return total_distance, positions


def optimize(
    start: np.array,
    coords: np.array,
    scan_width: float,
    arbitrary_large_number: int,
    n=360,
):
    distances_list = []
    angles_list = []
    positions_list = []
    for i in range(n):
        angle = i * 360.0 / n

        distance, positions = simulate_coverage(
            angle, start, coords, scan_width, arbitrary_large_number
        )

        distances_list.append(distance)
        angles_list.append(angle)
        positions_list.append(positions)

    idx = np.array(distances_list).argmin()

    return distances_list[idx], positions_list[idx]


def split_shape(coords: np.array, num_split: int):
    midp = midpoint(coords)
    line_coords = get_line_coords(coords)

    diffs = get_line_diff(line_coords)
    norms = np.linalg.norm(diffs, axis=1)
    norms = np.reshape(norms, (-1, 1))
    norms[norms == 0] = SMALL

    unit_diffs = diffs / np.repeat(norms, 2, axis=1)

    circum = np.sum(norms)
    travel_dist = circum / num_split

    curr_line = 0
    curr_dist = 0.0
    starts = [coords[curr_line]]
    polygons = []
    for i in range(num_split):
        polygons.append([])
        polygons[i].append(midp)
        polygons[i].append(starts[-1])

        curr_dist += travel_dist

        while True:
            curr_line_length = norms[curr_line]
            outside_curr_line = curr_dist > curr_line_length
            if outside_curr_line:
                curr_dist -= curr_line_length + SMALL

                curr_line += 1
                polygons[i].append(coords[curr_line])
            else:
                break

        new_start = coords[curr_line] + unit_diffs[curr_line] * curr_dist
        starts.append(new_start)

        polygons[i].append(starts[-1])
        polygons[i] = np.array(polygons[i])

    return polygons


def lat_long_to_meters(lat_long: np.matrix) -> np.matrix:
    result = np.empty_like(lat_long)
    result[:, 0] = (
        -lat_long[:, 1] * 40000 / 360 * np.cos(lat_long[:, 0] * np.pi / 360) * 1000
    )
    result[:, 1] = -lat_long[:, 0] * 40000 / 360 * 1000
    return result


def meters_to_lat_long(x_y: np.matrix):
    result = np.empty_like(x_y)
    result[:, 0] = -x_y[:, 1] * 360 / 40000 * 1 / 1000
    result[:, 1] = (
        -x_y[:, 0]
        * 360
        / 40000
        * 1
        / 1000
        * 1
        / np.cos(-x_y[:, 1] * 360 / 40000 * 1 / 1000 * np.pi / 360)
    )

    return result


def main(
    starting_coords: np.array,
    input_coords: np.array,
    uav_num: int,
    max_speed: float,
    height: float,
    scan_width: float,
    arbitrary_large_number: int = N,
    ear_clipping=False,
):
    ic = np.roll(input_coords, 1, axis=1)
    sc = np.roll(np.reshape(starting_coords, (1, -1)), 1, axis=1)

    input_adjust = lat_long_to_meters(ic)
    start_adjust = lat_long_to_meters(sc)

    if ear_clipping:
        input_aug = [np.array(pair) for pair in input_adjust]
        input_aug.append(input_aug[0])
        input_aug = np.array(input_aug)
        polygons = cut_polygon(input_aug)
    else:
        polygons = [input_adjust]

    all_pos = []
    all_distances = np.zeros((uav_num, 1))

    new_starts = np.repeat(np.reshape(start_adjust, (1, -1)), uav_num, axis=0)

    for p, polygon in enumerate(polygons):
        coords = split_shape(polygon, uav_num)

        for i, coord_array in enumerate(coords):
            start = new_starts[i]

            distance, positions = optimize(
                start, coord_array, scan_width, arbitrary_large_number
            )

            if p == len(polygons) - 1:
                positions = np.vstack((positions, np.reshape(start_adjust, (1, -1))))
                distance += np.linalg.norm(positions[-2] - positions[-1])

            adjusted_positions = meters_to_lat_long(positions)

            all_distances[i] += distance
            all_pos.append(adjusted_positions)
            draw_path(adjusted_positions, COLORS[i])

            new_starts[i] = positions[-1]
    draw_polygon(ic, c=(0.1, 0.1, 0.1))
    all_times = [d / max_speed for d in all_distances]

    all_pos = np.vstack(all_pos)
    min_x = np.min(all_pos[:, 0])
    max_x = np.max(all_pos[:, 0])
    min_y = np.min(all_pos[:, 1])
    max_y = np.max(all_pos[:, 1])
    dx = max_x - min_x
    dy = max_y - min_y

    def format_x(x):
        return f"{abs(round(x))}°{abs(round((x - round(x)) * 100))}'{'W' if x < 0 else 'E'}"

    def format_y(y):
        return f"{abs(round(y))}°{abs(round((y - round(y)) * 100))}'{'S' if y < 0 else 'N'}"

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    xticks = np.arange(min_x - dx, max_x + dx, dx)
    xlabels = [format_x(x) for x in xticks]
    plt.gca().set_xticks(xticks, xlabels)

    yticks = np.arange(min_y - dy, max_y + dy, dy)
    ylabels = [format_y(y) for y in yticks]
    plt.gca().set_yticks(yticks, ylabels)

    plt.gca().set_aspect("equal")

    plt.show()

    distances_str = [
        f"UAV #{i + 1} ({COLOR_STRINGS[i]} path) - {int(dist)} meters"
        for i, dist in enumerate(all_distances)
    ]
    times_str = [
        f"UAV #{i + 1} ({COLOR_STRINGS[i]} path) - {datetime.timedelta(seconds=int(time))}"
        for i, time in enumerate(all_times)
    ]
    nl = "\n"
    msg = f"""
        UAV PATH DISTANCES:
{nl.join(distances_str)}

        MINIMUM PATH DISTANCE: {int(np.min(all_distances))} meters
        MAXIMUM PATH DISTANCE: {int(np.max(all_distances))} meters
        MEAN PATH DISTANCE: {int(np.mean(all_distances))} meters
        MEDIAN PATH DISTANCE: {int(np.median(all_distances))} meters
        
        TOTAL DISTANCE TRAVELED: {int(np.sum(all_distances))} meters
        
        UAV SEARCH TIMES:
{nl.join(times_str)}
        
        MINIMUM TIME TAKEN: {datetime.timedelta(seconds=int(np.min(all_times)))}
        MEAN TIME TAKEN: {datetime.timedelta(seconds=int(np.mean(all_times)))}
        MEDIAN TIME TAKEN: {datetime.timedelta(seconds=int(np.median(all_times)))}
        
        MAXIMUM TIME TAKEN: {datetime.timedelta(seconds=int(np.max(all_times)))}
        JOINT COVERAGE TIME: {datetime.timedelta(seconds=int(np.sum(all_times)))}
        """
    messagebox.showinfo("Report", msg)


def ask_input():
    def get_float(msg: str):
        result = simpledialog.askfloat("Awaiting User Input", msg)
        if result is None:
            os._exit(0)

        return result

    def get_int(msg: str):
        result = simpledialog.askinteger("Awaiting User Input", msg)
        if result is None:
            os._exit(0)

        return result

    def get_str(msg: str):
        result = simpledialog.askstring("Awaiting User Input", msg)
        if result is None:
            os._exit(0)

        return result

    start_lat = get_float(
        "Please input the starting latitude as a signed decimal number."
    )
    start_long = get_float(
        "Please input the starting longitude as a signed decimal number."
    )
    start = np.array((start_lat, start_long))

    positions = []
    while True:
        msg = 'Please input a bounding point of the search area in the form "LATITUDE, LONGITUDE". Please input nothing once you have entered all bounding points.'
        nl = "\n"
        str_positions = [f"{position[0]}, {position[1]}" for position in positions]
        add = (
            f"So far:{nl + nl.join(str_positions) + nl}has been entered" + nl + nl + msg
        )

        input = get_str(add if len(positions) != 0 else msg)
        if input.strip() == "":
            break

        split = input.strip().split(",")
        if len(split) != 2:
            continue

        try:
            position = [float(split[0]), float(split[1])]
            positions.append(position)
        except Exception as e:
            continue

    positions = np.array(positions)

    uav_num = get_int("Please input the number of UAVs.")
    speed = float(get_int("Please input the maximum speed of the UAVs in m/s."))
    height = float(get_int("Please input the starting height of the UAVs in meters."))
    cover_width = float(get_int("Please sensor coverage width of the UAVs in meters."))

    earclip_enabled = False
    if (
        "y"
        == get_str(
            'Would you like to enable ear clipping (for concave bounding boxes). Input "y" if you would like it enabled. Input anything else if you would like it disabled'
        ).strip()
    ):
        earclip_enabled = True

    return start, positions, uav_num, speed, height, cover_width, earclip_enabled


if __name__ == "__main__":
    start, positions, uav_num, max_speed, height, scan_width, earclip_enabled = (
        ask_input()
    )
    main(
        start,
        positions,
        uav_num,
        max_speed,
        height,
        scan_width,
        ear_clipping=earclip_enabled,
    )
