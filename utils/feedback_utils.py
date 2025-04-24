import numpy as np
import cv2
from utils.dtw import smooth_cycle, compute_angle
import matplotlib.pyplot as plt
from dtaidistance import dtw_visualisation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import find_peaks

def compute_angle_between_lines(p1, p2, p3, p4):
    """Computes the angle between two lines formed by points (p1, p2) and (p3, p4)."""
    # Compute direction vectors for the two lines
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])  # Vector from p1 to p2
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])  # Vector from p3 to p4

    # Compute the dot product and magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the angle using the dot product formula: cos(theta) = dot(v1, v2) / (|v1| * |v2|)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

    # Return the angle in radians, use np.clip to handle floating point errors
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)  # Convert to degrees if needed

def extract_multivariate_series_for_lines(cycle_data, line_joint_quadruplets, run_args):
    """Extracts multivariate time-series data for the angles between lines formed by four joints from a cycle."""
    all_angles = []
    frames = []
    # TODO Make option?
    joints_list = [joint for tuple_joints in line_joint_quadruplets for joint in tuple_joints]
    if run_args.DTW.GAUS_FILTER:
        cycle_data = smooth_cycle(cycle_data, joints_list, sigma=run_args.DTW.SIGMA_VALUE)
    for i in range(len(cycle_data[line_joint_quadruplets[0][0] + "_x"])):
        angles = []
        for joint1, joint2, joint3, joint4 in line_joint_quadruplets:
            if joint1 == run_args.DTW.CHOOSEN_REF:
                p1 = (0, 0)
            else:
                p1 = (cycle_data[joint1 + "_x"][i], cycle_data[joint1 + "_y"][i])
            if joint2 == run_args.DTW.CHOOSEN_REF:
                p2 = (0, 0)
            else:
                p2 = (cycle_data[joint2 + "_x"][i], cycle_data[joint2 + "_y"][i])
            if joint3 == run_args.DTW.CHOOSEN_REF:
                p3 = (0, 0)
            else:
                p3 = (cycle_data[joint3 + "_x"][i], cycle_data[joint3 + "_y"][i])
            if joint4 == run_args.DTW.CHOOSEN_REF:
                p4 = (0, 0)
            else:
                p4 = (cycle_data[joint4 + "_x"][i], cycle_data[joint4 + "_y"][i])
            angles.append(compute_angle_between_lines(p1, p2, p3, p4))
        all_angles.append(angles)
        frames.append(i)
    return np.array(all_angles), frames

def extract_multivariate_series_for_single_lines(cycle_data, line_joint_pairs, run_args):
    """
    Extract the angle of joint pairs compared to a horizontal line
    """
    all_angles = []
    frames = []

    try:
        if cycle_data[line_joint_pairs[0][0]] == run_args.DTW.CHOOSEN_REF:
            check_term = "_x"
    except KeyError:
        check_term = "_x_ref"

    # TODO Make option?
    joints_list = [joint for tuple_joints in line_joint_pairs for joint in tuple_joints]
    if run_args.DTW.GAUS_FILTER:
        cycle_data = smooth_cycle(cycle_data, joints_list, sigma=run_args.DTW.SIGMA_VALUE)
    for i in range(len(cycle_data[line_joint_pairs[0][0] + check_term])):
        angles = []
        for joint1, joint2 in line_joint_pairs:
            # TODO Change json to add Hip_x/Hip_y (the reference realtive too)?
            if joint1 == run_args.DTW.CHOOSEN_REF:
                p1 = (0, 0)
            else:
                p1 = (cycle_data[joint1 + "_x"][i], cycle_data[joint1 + "_y"][i])
            if joint2 == run_args.DTW.CHOOSEN_REF:
                p2 = (0, 0)
            else:
                p2 = (cycle_data[joint2 + "_x"][i], cycle_data[joint2 + "_y"][i])
            
            angles.append(compute_angle_between_lines(p1, p2, (1,0), (0,0)))
        all_angles.append(angles)
        frames.append(i)
    return np.array(all_angles), frames

def distance_formula(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1]- p1[1]) ** 2) ** 0.5

def extract_multivariate_series_for_distances(cycle_data, joints_distance, run_args):
    """Extracts distances between certain joints for the cycle"""
    all_distances = []
    frames = []

    joints_list = [joint for tuple_joints in joints_distance for joint in tuple_joints]
    if run_args.DTW.GAUS_FILTER:
        # with normalization for distances!
        cycle_data = smooth_cycle(cycle_data, joints_list, sigma=run_args.DTW.SIGMA_VALUE, norm = False)

    for i in range(len(cycle_data[joints_distance[0][0]+ "_x"])):
        dists = []
        for joint1, joint2 in joints_distance:
            p1 = (cycle_data[joint1 + "_x"][i], cycle_data[joint1 + "_y"][i])
            p2 = (cycle_data[joint2 + "_x"][i], cycle_data[joint2 + "_y"][i])
            dist = distance_formula(p1, p2)
            dists.append(dist)
        all_distances.append(dists)
        frames.append(i)
    return np.array(all_distances), frames

def calculate_differences(list1, list2, index_pairs):
    """
    Calculates the differences between values in list1 and list2 using the given index pairs.

    Args:
    list1 (list): The first list of values.
    list2 (list): The second list of values.
    index_pairs (list of tuples): List of tuples where each tuple (i, j) corresponds to 
                                  comparing list1[i] with list2[j].

    Returns:
    list: A list of differences (list1[i] - list2[j]) for each pair (i, j).
    """
    differences = []
    for i, j in index_pairs:
        difference = list1[i] - list2[j]
        differences.append(difference)
    return differences

def calculate_similarity(user, expert, index_pairs):
    """
    Calculates the similarity between two values in a list, how similar exper value is to user
    """
    similarities = []
    for i, j in index_pairs:
        sim = user [i] / expert[j]
        similarities.append(sim)
    return similarities


def get_line_points(cycle, joints_list, frame, run_args):
    points = []
    for joints in joints_list:
        #TODO Get ref from cfg/other way?
        for joint in joints:
            x_suffix = "_x"
            y_suffix = "_y"
            # TODO Fix because only ref in json
            if joint == run_args.DTW.CHOOSEN_REF:
                x_suffix += "_ref"
                y_suffix += "_ref"
                p_x = int(cycle.get("Hip_x_ref")[frame])
                p_y = int(cycle.get("Hip_y_ref")[frame])
            else:
                p_x = int(cycle.get(joint + x_suffix)[frame] + cycle.get("Hip_x_ref")[frame])
                p_y = int(cycle.get(joint + y_suffix)[frame] + cycle.get("Hip_y_ref")[frame])
            points.append((p_x,p_y))
    return points

def draw_joint_relative_lines(joints_lines, frame, points, l_color=(0,255,0), p_color=(0,0,255), l_thickness = 2, p_radius = 2):
    for i in range(len(joints_lines)):
        j = i*4
        cv2.line(frame, points[0+j], points[1+j], color=l_color, thickness=l_thickness)  # First pair
        cv2.line(frame, points[2+j], points[3+j], color=l_color, thickness=l_thickness)  # Second pair
        for point in points:
            cv2.circle(frame, point, radius=p_radius, color=p_color, thickness=-1)

def draw_joint_single_lines(joints_lines, frame, points, l_color=(0,255,0), p_color=(0,0,255), l_thickness = 2, p_radius = 2):
    for i in range(len(joints_lines)):
        j = i*2
        cv2.line(frame, points[0+j], points[1+j], color=l_color, thickness=l_thickness)
        for point in points:
            cv2.circle(frame, point, radius=p_radius, color=p_color, thickness=-1)

def calc_draw_angle(pt1, pt2, pt3):
    v1 = np.array(pt1) - np.array(pt2)
    v2 = np.array(pt3) - np.array(pt2)
    start_angle = int(np.degrees(np.arctan2(v1[1], v1[0]))) % 360
    end_angle = int(np.degrees(np.arctan2(v2[1], v2[0]))) % 360
    return start_angle, end_angle
    
def calc_angle_diff(start_angle, end_angle):
    angle_diff = end_angle - start_angle
    # If the difference is more than 180 degrees, swap the direction for the shortest arc
    if angle_diff > 180:
        angle_diff -= 360  # Taking the shorter path by going in the reverse direction
    elif angle_diff < -180:
        angle_diff += 360
    return angle_diff

def draw_joint_angles(joint_angles, frame, points, l_color=(0,255,0), p_color=(0,0,255), l_thickness = 2, p_radius = 2, e_radius = 10):
    for i in range(len(joint_angles)):
        j = i*3
        cv2.line(frame, points[0+j], points[1+j], color=l_color, thickness=l_thickness)  # First pair
        cv2.line(frame, points[1+j], points[2+j], color=l_color, thickness=l_thickness)  # Second pair
        for point in points:
            cv2.circle(frame, point, radius=p_radius, color=p_color, thickness=-1)
        start_angle, end_angle = calc_draw_angle(points[0+j], points[1+j], points[2+j])
        angle_diff = calc_angle_diff(start_angle, end_angle)
        if angle_diff > 0:
            cv2.ellipse(frame, points[1+j], (e_radius, e_radius), 0, start_angle, start_angle + angle_diff, l_color, l_thickness)
        else:
            cv2.ellipse(frame, points[1+j], (e_radius, e_radius), 0, end_angle, end_angle - angle_diff, l_color, l_thickness)


def draw_table(frame, angles_tuple, lines_rel_tuple, lines_hor_tuple, distances_tuple, match, iter):

    # Unpack tuples
    joint_angles, user_angles, expert_angles, diff_angles, sim_angles = angles_tuple
    joint_lines, user_lines, expert_lines, diff_lines, sim_lines = lines_rel_tuple
    joint_hor_lines, user_hor_lines, expert_hor_lines, diff_hor_lines, sim_hor_lines = lines_hor_tuple
    joints_distance, user_distances, expert_distances, diff_distances, sim_distances = distances_tuple
    
    height, width, _ = frame.shape
    rows = 0
    header = False
    if len(joint_angles) > 0:
        rows += len(joint_angles) + 1 # +1 for header
    if len(joint_lines) > 0:
        header = True
        rows += len(joint_lines)
    if len(joint_hor_lines) > 0:
        header = True
        rows += len(joint_hor_lines)
    if len(joints_distance) > 0:
        rows += len(joints_distance) + 1 # +1 for header

    if header:
        rows += 1
    cols = 5
    cell_width = int(width*0.9 // cols)
    cell_height = height // 20
    # cell_width = 380
    # cell_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)
    border_colour = (255, 255, 255)

    # table_data = [
    #     ['Angle', 'User', 'Expert', 'Difference', 'Header5'],
    #     ['Row1Col1', 'Row1Col2', 'Row1Col3', 'Row1Col4', 'Row1Col5'],
    #     ['Row2Col1', 'Row2Col2', 'Row2Col3', 'Row2Col4', 'Row2Col5'],
    #     ['Row3Col1', 'Row3Col2', 'Row3Col3', 'Row3Col4', 'Row3Col5'],
    #     ['Row4Col1', 'Row4Col2', 'Row4Col3', 'Row4Col4', 'Row4Col5']
    # ]
    table_data = []
    if len(joint_angles) > 0:
        table_data.append(['Angles', 'User', 'Expert', 'Difference', 'Similarity (%)'])
        for i, angle in enumerate(joint_angles):
            row = []
            row.append(str(angle))
            row.append(f"{user_angles[match[0]][i]:.2f}")
            row.append(f"{expert_angles[match[1]][i]:.2f}")
            row.append(f"{diff_angles[iter][i]:.2f}")
            row.append(f"{sim_angles[iter][i]:.2%}")
            table_data.append(row)
    if len(joint_hor_lines) > 0 or len(joint_lines) > 0:
        table_data.append(['Lines', 'User', 'Expert', 'Difference', 'Similarity (%)'])
        for i, lines in enumerate(joint_lines):
            row = []
            row.append(str(lines))
            row.append(f"{user_lines[match[0]][i]:.2f}")
            row.append(f"{expert_lines[match[1]][i]:.2f}")
            row.append(f"{diff_lines[iter][i]:.2f}")
            row.append(f"{sim_lines[iter][i]:.2%}")
            table_data.append(row)
        for i, lines in enumerate(joint_hor_lines):
            row = []
            row.append(str(lines))
            row.append(f"{user_hor_lines[match[0]][i]:.2f}")
            row.append(f"{expert_hor_lines[match[1]][i]:.2f}")
            row.append(f"{diff_hor_lines[iter][i]:.2f}")
            row.append(f"{sim_hor_lines[iter][i]:.2%}")
            table_data.append(row)
    if len(joints_distance) > 0:        
      table_data.append(['Distances', 'User', 'Expert', 'Difference', 'Similarity (%)'])
      for i, dist in enumerate(joints_distance):
          row = []
          row.append(str(dist))
          row.append(f"{user_distances[match[0]][i]:.2f}")
          row.append(f"{expert_distances[match[1]][i]:.2f}")
          row.append(f"{diff_distances[iter][i]:.2f}")
          row.append(f"{sim_distances[iter][i]:.2%}")
          table_data.append(row)
    

    for row in range(rows):
        for col in range(cols):
            # Top-left corner (x1, y1) and bottom-right corner (x2, y2) for each cell
            if col == 0: # Making first one wider
                top_left = (col * cell_width, row * cell_height)
                bottom_right = ((col + 1) * cell_width + int(cell_width*0.5), (row + 1) * cell_height)
            else:
                top_left = (col * cell_width + int(cell_width*0.5), row * cell_height)
                bottom_right = ((col + 1) * cell_width + int(cell_width*0.5), (row + 1) * cell_height)
                
            # Draw rectangle (cell border)
            cv2.rectangle(frame, top_left, bottom_right, border_colour, 2)  # Black border

            try:
                deciding_value = abs(float(table_data[row][3]))  # Value from 4th column (deciding factor)
            except ValueError:
                deciding_value = 0

            # Map the value to a color intensity (more red for higher values)
            red_intensity = min(255, int(deciding_value * 8))  # Scale value to a reasonable range (adjust multiplier)
            red_intensity = max(0, red_intensity)  # Ensure it's at least 0
            
            text_color = (255- red_intensity, 255 - red_intensity, 255)

            
            # Add text to the cell (centered)
            text = table_data[row][col]
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            if col == 0: # First one wider
                text_x = top_left[0] + (cell_width + int(cell_width*0.5) - text_width) // 2
                text_y = top_left[1] + (cell_height + text_height) // 2
            else:
                text_x = top_left[0] + (cell_width - text_width) // 2
                text_y = top_left[1] + (cell_height + text_height) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame

def safe_hstack(arrays):
    # Filter out empty arrays (those with 0 columns, including 1D arrays with no data)
    valid_arrays = []
    for arr in arrays:
        # Handle cases where the array might be 1D or empty
        if arr.ndim == 1 and arr.size > 0:
            valid_arrays.append(arr[:, np.newaxis])  # Convert to 2D array (n, 1)
        elif arr.ndim == 2 and arr.shape[1] > 0:
            valid_arrays.append(arr)
    
    if valid_arrays:
        return np.hstack(valid_arrays)
    else:
        # If no valid arrays are found, return an empty array with the correct number of rows
        return np.empty((arrays[0].shape[0], 0))

def draw_plots(frame,  angles_tuple, lines_tuple, lines_hor_tuple, dists_tuple, path, frame1, frame2):

    user_angles, expert_angles, joint_angles = angles_tuple
    user_lines, expert_lines, joints_lines_relative = lines_tuple
    user_hor_lines, expert_hor_lines, joint_lines_hor = lines_hor_tuple
    user_distances, expert_distances, joints_distance = dists_tuple

    joints = []
    if len(joint_angles) > 0:
        joints.extend(joint_angles)
    if len(joints_lines_relative) > 0:
        joints.extend(joints_lines_relative)
    if len(joint_lines_hor) > 0:
        joints.extend(joint_lines_hor)
    if len(joints_distance) > 0:
        joints.extend(joints_distance)

    if 0 < len(joints) <= 4:
        cols = 2
    elif 4 < len(joints) <= 9:
        cols = 3
    else:
        cols = 4
    # Determine the grid layout
    num_plots = len(joints)
    if num_plots == 0:
        return frame
    rows = int(np.ceil(num_plots / cols))  # Calculate how many rows based on number of plots and columns

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    plot_width = frame_width // cols  # Width of each plot
    plot_height = frame_height // rows  # Height of each plot

    user_angles_arr = np.array(user_angles)
    expert_angles_arr = np.array(expert_angles)
    user_lines_arr = np.array(user_lines)
    expert_lines_arr = np.array(expert_lines)
    user_lines_hor_arr = np.array(user_hor_lines)
    expert_lines_hor_arr = np.array(expert_hor_lines)
    user_dists_arr = np.array(user_distances)
    expert_dists_arr = np.array(expert_distances)


    user_features = safe_hstack([user_angles_arr, user_lines_arr, user_lines_hor_arr, user_dists_arr])
    expert_features = safe_hstack([expert_angles_arr, expert_lines_arr, expert_lines_hor_arr, expert_dists_arr])

    for i, joint_tuple in enumerate(joints):
        fig, ax = dtw_visualisation.plot_warping_single_ax(user_features[:, i], expert_features[:, i], path, filename=None)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.plot(user_features[:, i], label=f"User", color='c')
        ax.plot(expert_features[:, i], label=f"Expert", color='y')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.tick_params(axis='both', colors='white')
        ax.set_title(f"DTW Alignment: {joint_tuple}", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame", color='white', fontsize=10, fontweight='bold')
        #ax.set_ylabel(f"Degree angle of {angle}", color='white', fontsize=10, fontweight='bold')
        ax.set_ylabel(f"Degree", color='white', fontsize=10, fontweight='bold')
        ax.grid(True, color='white', linestyle='-', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')  # Set the border color to white

        # Plot dots at the specified timestamps
        ax.plot(frame1, user_features[frame1, i], marker='o', markersize=10, color='b')
        ax.plot(frame2, expert_features[frame2, i], marker='o', markersize=10, color='b')



        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #frame_height, frame_width = frame.shape[:2]
        plot_img_resized = cv2.resize(plot_img, (frame_width, frame_height))
        # frame = cv2.addWeighted(frame, 0, plot_img_resized, 1, 0)
        plt.close(fig)
                # Calculate the position on the frame where the plot should be placed
        row_idx = i // cols  # Determine the row index
        col_idx = i % cols  # Determine the column index
        y_start = row_idx * plot_height  # Starting y-coordinate
        x_start = col_idx * plot_width  # Starting x-coordinate

        # Ensure that the plot fits in the calculated area
        if plot_img_resized.shape[0] != plot_height or plot_img_resized.shape[1] != plot_width:
            plot_img_resized = cv2.resize(plot_img_resized, (plot_width, plot_height))

        # Place the resized plot image onto the frame
        frame[y_start:y_start + plot_height, x_start:x_start + plot_width] = plot_img_resized


    return frame

# function for choosing the ID of the skier that we want to track
def choose_id(results_list, video_path):
            
    # Extract the first frame image_id
    first_frame_id = results_list[0]["image_id"]

    # Filter out only the keypoints from the first frame
    first_frame_data = [entry for entry in results_list if entry["image_id"] == first_frame_id]

    # Load the first frame from the video
    cap = cv2.VideoCapture(video_path)

    success, image = cap.read()
    cap.release()

    if not success:
        print("Could not read the first frame from the video.")
    else:
        
        # select different colors for different IDs
        np.random.seed(42)  # Ensure reproducibility
        colors = {person["idx"]: tuple(np.random.randint(0, 255, 3).tolist()) for person in first_frame_data}
        
        # Dictionary to store keypoints by ID
        person_keypoints = {}
        
        # Draw keypoints and IDs on the image
        for person in first_frame_data:
            idx = person["idx"]  # Person ID
            keypoints = np.array(person["keypoints"]).reshape(-1, 3)[:, :2]  # Extract (x, y) keypoints
            color = colors[idx]
            
            person_keypoints[idx] = keypoints  # Store keypoints for click detectio
            
            # Draw keypoints
            for (x, y) in keypoints:
                cv2.circle(image, (int(x), int(y)), 5, color, -1)  # Red keypoints
            
            # Display the ID near the head (assuming first keypoint is the head)
            head_x, head_y = map(int, keypoints[0])
            cv2.putText(image, str(idx), (head_x, head_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        selected_id = None

        # Mouse click callback function
        def select_id(event, x, y, flags, param):
            nonlocal selected_id
            if event == cv2.EVENT_LBUTTONDOWN:
                min_dist = float('inf')
                closest_id = None
                
                # Find the closest keypoint to the click position
                for pid, keypoints in person_keypoints.items():
                    for kx, ky in keypoints:
                        dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_id = pid
                
                if closest_id is not None:
                    selected_id = closest_id
                    print(f"You selected ID: {selected_id}")
        
        # Show image with OpenCV
        cv2.imshow("Click on the ID you want to select", image)
        cv2.setMouseCallback("Click on the ID you want to select", select_id)
        
        # Wait for selection
        while selected_id is None:
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        
        return selected_id


def save_feedback(feedback_dict, frame, feedback, range_):
    for f in range(frame - range_, frame + range_ + 1):
        if f >= 0:
            feedback_dict[f] = feedback
            
# feedback for wide legs
def feedback_wide_legs(expert_distances, user_distances, diff_distances, path, feedback_range):
    
    feedback_per_frame = {}

    expert_distances_arr = np.concatenate(expert_distances)
    user_distances_arr = np.concatenate(user_distances)
    diff_distances_arr = np.concatenate(diff_distances)
    #print(f"max: {np.max(diff_distances_arr)}, min: {np.min(diff_distances_arr)}")
    # find both minima and maxima for expert cycle to determine the phase he is in
    expert_peaks_pos = find_peaks(expert_distances_arr, height=0)
    expert_peaks_neg = find_peaks(-expert_distances_arr, height=-float("inf"))
    # compare minimum of expert to matched user frame
    if len(expert_peaks_neg[0]):
        for peak_min_idx in expert_peaks_neg[0]:
            #print(f"Expert dist at legs together: {expert_distances_arr[peak_min_idx]}, idx = {peak_min_idx}")
            matches = [pair for pair in path if pair[1] == peak_min_idx]
            avg_dist_user = 0
            if len(matches) <= 0:
                print("Could not find any matches to expert maxima")
            for match in matches:
                #print(f"User dist legs together: {user_distances_arr[match[0]]}")
                avg_dist_user += user_distances_arr[match[0]]
            avg_dist_user /= len(matches)
            #print(f"User avg dist legs together: {avg_dist_user}")
            if abs(avg_dist_user-expert_distances_arr[peak_min_idx]) < 3: 
                feedback =  "You are doing great and bringing the legs close together!"
                print(feedback)
                save_feedback(feedback_per_frame, peak_min_idx, feedback, feedback_range)
            elif abs(avg_dist_user-expert_distances_arr[peak_min_idx]) < 10: 
                continue
            elif abs(avg_dist_user-expert_distances_arr[peak_min_idx]) < 15: 
                feedback = "You might have to bring your legs closer together. You can do this by trying to move the arms faster and the legs will follow."
                print(feedback)
                save_feedback(feedback_per_frame, peak_min_idx, feedback, feedback_range)
            else:
                feedback = "You should bring your legs closer together. You can do this by trying to move the arms faster and the legs will follow."
                print(feedback)
                save_feedback(feedback_per_frame, peak_min_idx, feedback, feedback_range)
            #print(f"Difference at legs together = {abs(avg_dist_user-expert_distances_arr[peak_min_idx])}") 
            print("")       
    # compare maximum of expert to matched user frame
    if len(expert_peaks_pos[0]):
        for peak_max_idx in expert_peaks_pos[0]:
            #print(f"Expert dist at push: {expert_distances_arr[peak_max_idx]}, idx = {peak_max_idx}")
            matches = [pair for pair in path if pair[1] == peak_max_idx]
            avg_dist_user = 0
            if len(matches) <= 0:
                print("Could not find any matches to expert maxima")
            for match in matches:
                #print(f"User dist at push: {user_distances_arr[match[0]]}")
                avg_dist_user += user_distances_arr[match[0]]
            avg_dist_user /= len(matches)
            #print(f"User avg dist at push: {avg_dist_user}")
            if abs(avg_dist_user-expert_distances_arr[peak_max_idx]) < 3:
                feedback = "You are doing great and really using your legs to push yourself forward!"
                print(feedback)
                save_feedback(feedback_per_frame, peak_max_idx, feedback, feedback_range)
            elif abs(avg_dist_user-expert_distances_arr[peak_max_idx]) < 10: 
                continue
            elif abs(avg_dist_user-expert_distances_arr[peak_max_idx]) < 15:
                if avg_dist_user-expert_distances_arr[peak_max_idx] > 0:
                    feedback = "You are pushing with your feet a lot more than the expert your data is compared to."
                    print(feedback)
                    save_feedback(feedback_per_frame, peak_max_idx, feedback, feedback_range)
                else:
                    feedback = "It looks like you should try to use your legs more and try to push yourself more forward."
                    print(feedback)
                    save_feedback(feedback_per_frame, peak_max_idx, feedback, feedback_range)
            else:
                feedback = "You should use your legs more and try to push yourself more forward."
                print(feedback)
                save_feedback(feedback_per_frame, peak_max_idx, feedback, feedback_range)
            #print(f"Difference at push = {abs(avg_dist_user-expert_distances_arr[peak_max_idx])}")
            print("")  
    
    return feedback_per_frame
    
# def feedback_shift_weight(expert_distances, user_distances, diff_distances, expert_angles, user_angles,  diff_angles,expert_horizontal_lines, user_horizontal_lines, diff_lines_horizontal, path, feedback_range):
#     feedback_per_frame = {}

#     # based on distances between the feet, we want to determine what phase the skier is in
#     expert_distances_arr = np.concatenate(expert_distances)
#     user_distances_arr = np.concatenate(user_distances)
#     diff_distances_arr = np.concatenate(diff_distances)

#     # find both minima and maxima for expert cycle to determine the phase he is in
#     expert_peaks_pos = find_peaks(expert_distances_arr, height=0)
#     expert_peaks_neg = find_peaks(-expert_distances_arr, height=-float("inf"))
    
#     # compare minimum of expert to matched user frame, that is when the feet are together
#     if len(expert_peaks_neg[0]):
#         for peak_min_idx in expert_peaks_neg[0]:
#             # find the matches
#             matches = [pair for pair in path if pair[1] == peak_min_idx]
#             if len(matches) <= 0:
#                 print("Could not find any matches to expert maxima")
#             for match in matches:
#                 right_diff = abs(expert_angles[peak_min_idx][0]-user_angles[match[0]][0])
#                 left_diff = abs(expert_angles[peak_min_idx][1]-user_angles[match[0]][1])
#                 horizontal_diff = abs(expert_horizontal_lines[peak_min_idx]-user_horizontal_lines[match[0]])
#                 sum_diffs = right_diff + left_diff + horizontal_diff
#                 vals = [right_diff, left_diff, horizontal_diff]
#                 max_idx = vals.index(max(vals))
#                 #print(f"Legs together sum: {sum_diffs}, max idx: {max_idx}, right: {right_diff}, left: {left_diff}, horizontal: {horizontal_diff}")
                
#                 # print(f"When Legs are together, right side:{abs(expert_angles[peak_min_idx][0]-user_angles[match[0]][0])} Expert angle {expert_angles[peak_min_idx][0]}, matched user angle {user_angles[match[0]][0]}")
#                 # print(f"When Legs are together, left side: {abs(expert_angles[peak_min_idx][1]-user_angles[match[0]][1])} Expert angle {expert_angles[peak_min_idx][1]}, matched user angle {user_angles[match[0]][1]}")
#                 # print(f"When Legs are together, horizontal: {abs(expert_horizontal_lines[peak_min_idx]-user_horizontal_lines[match[0]])} Expert angle {expert_horizontal_lines[peak_min_idx]}, matched user angle {user_horizontal_lines[match[0]]}")
#                 expert_hip_angle_diff = expert_angles[peak_min_idx][0] - expert_angles[peak_min_idx][1]
#                 expert_normalized_diff = expert_hip_angle_diff / (expert_angles[peak_min_idx][0] + expert_angles[peak_min_idx][1])
#                 user_hip_angle_diff = user_angles[peak_min_idx][0] - user_angles[peak_min_idx][1]
#                 user_normalized_diff = user_hip_angle_diff / (user_angles[peak_min_idx][0] + user_angles[peak_min_idx][1])
#                 expert_dist = expert_distances_arr[peak_min_idx]
#                 user_dist = user_distances_arr[match[0]]
#                 alpha = 0.7
#                 beta = 0.3
#                 weight_shift_score = alpha * expert_normalized_diff + beta * expert_dist
                
#                 print("Together", expert_hip_angle_diff, user_hip_angle_diff, expert_dist, user_dist)
#     if len(expert_peaks_pos[0]):
#         for peak_max_idx in expert_peaks_pos[0]:
#             # find the matches
#             matches = [pair for pair in path if pair[1] == peak_max_idx]
#             if len(matches) <= 0:
#                 print("Could not find any matches to expert maxima")
#             for match in matches:
                
#                 right_diff = abs(expert_angles[peak_max_idx][0]-user_angles[match[0]][0])
#                 left_diff = abs(expert_angles[peak_max_idx][1]-user_angles[match[0]][1])
#                 horizontal_diff = abs(expert_horizontal_lines[peak_max_idx]-user_horizontal_lines[match[0]])
#                 sum_diffs = right_diff + left_diff + horizontal_diff
#                 vals = [right_diff, left_diff, horizontal_diff]
#                 max_idx = vals.index(max(vals))
#                 #print(f"At push: {sum_diffs}, max idx: {max_idx}, right: {right_diff}, left: {left_diff}, horizontal: {horizontal_diff}")
#                 # if sum_diffs > 15:
#                 #     print("Sum ist bigger than 15 at push!")
#                 # print(f"At push, right side: {abs(expert_angles[peak_max_idx][0]-user_angles[match[0]][0])} Expert angle {expert_angles[peak_max_idx][0]}, matched user angle {user_angles[match[0]][0]}")
#                 # print(f"At push, left side: {abs(expert_angles[peak_max_idx][1]-user_angles[match[0]][1])} Expert angle {expert_angles[peak_max_idx][1]}, matched user angle {user_angles[match[0]][1]}")
#                 # print(f"At push, horizontal: {abs(expert_horizontal_lines[peak_max_idx]-user_horizontal_lines[match[0]])} Expert angle {expert_horizontal_lines[peak_max_idx]}, matched user angle {user_horizontal_lines[match[0]]}")
                
#                 expert_hip_angle_diff = expert_angles[peak_max_idx][0] - expert_angles[peak_max_idx][1]
#                 expert_normalized_diff = expert_hip_angle_diff / (expert_angles[peak_max_idx][0] + expert_angles[peak_max_idx][1])
#                 user_hip_angle_diff = user_angles[peak_max_idx][0] - user_angles[peak_max_idx][1]
#                 user_normalized_diff = user_hip_angle_diff / (user_angles[peak_max_idx][0] + user_angles[peak_max_idx][1])
#                 expert_dist = expert_distances_arr[peak_max_idx]
#                 user_dist = user_distances_arr[match[0]]
#                 alpha = 0.7
#                 beta = 0.3
#                 weight_shift_score = alpha * expert_normalized_diff + beta * expert_dist
                
#                 print("Push", expert_hip_angle_diff, user_hip_angle_diff, expert_dist, user_dist)
#     return feedback_per_frame

# def feedback_shift_weight(
#     expert_distances, user_distances, diff_distances,
#     expert_angles, user_angles, diff_angles,
#     expert_horizontal_lines, user_horizontal_lines, diff_lines_horizontal,
#     path, feedback_range
# ):
#     feedback_per_frame = {}

#     # Flatten arrays
#     expert_distances_arr = np.concatenate(expert_distances)
#     user_distances_arr = np.concatenate(user_distances)

#     # Identify stride keyframes
#     expert_peaks_max = find_peaks(expert_distances_arr, height=0)[0]  # Push-offs
#     expert_peaks_min = find_peaks(-expert_distances_arr, height=-float("inf"))[0]  # Legs together

#     # Parameters
#     alpha = 0.7  # Weight for angle diff
#     beta = 0.3   # Weight for foot distance

#     def normalized_hip_diff(angles):
#         # angles = (right, left)
#         return (angles[0] - angles[1]) / (angles[0] + angles[1] + 1e-5)  # Prevent division by 0

#     # Use expert's stride max to normalize distances
#     stride_max = np.max(expert_distances_arr)

#     # Loop through all key phases (minima = glide, maxima = push)
#     for phase_type, peak_indices in [('Glide', expert_peaks_min), ('Push', expert_peaks_max)]:
#         for expert_idx in peak_indices:
#             # Find all user frames matched to this expert frame
#             user_frame_indices = [pair[0] for pair in path if pair[1] == expert_idx]
#             if not user_frame_indices:
#                 continue

#             # Compute average user angles and distance
#             avg_user_angles = np.mean([user_angles[i] for i in user_frame_indices], axis=0)
#             avg_user_dist = np.mean([user_distances_arr[i] for i in user_frame_indices])

#             # Expert values
#             expert_diff = normalized_hip_diff(expert_angles[expert_idx])
#             expert_dist_norm = expert_distances_arr[expert_idx] / stride_max
#             expert_score = alpha * expert_diff + beta * expert_dist_norm

#             # User values (averaged)
#             user_diff = normalized_hip_diff(avg_user_angles)
#             user_dist_norm = avg_user_dist / stride_max
#             user_score = alpha * user_diff + beta * user_dist_norm


#             error = abs(user_score - expert_score)
#             print("Error", error)
#             feedback = generate_feedback_comment(phase_type, error)
#             print(feedback)

#             # Save frame-level feedback
#             # feedback_per_frame[user_idx] = {
#             #     "phase": phase_type,
#             #     "user_score": user_score,
#             #     "expert_score": expert_score,
#             #     "error": error,
#             #     "comment": generate_feedback_comment(phase_type, error)
#             # }

#     return feedback_per_frame


# def generate_feedback_comment(phase, error, low=0.05, med=0.15):
#     if error < low:
#         return f"{phase}: Excellent weight shift — closely matches expert."
#     elif error < med:
#         return f"{phase}: Decent weight transfer, but could be smoother or more complete."
#     else:
#         return f"{phase}: Needs better weight transfer — try shifting more clearly to one side."

# def feedback_shift_weight(
#     expert_distances, user_distances, diff_distances,
#     expert_angles, user_angles, diff_angles,
#     expert_horizontal_lines, user_horizontal_lines, diff_lines_horizontal,
#     path, feedback_range,
#     num_samples_per_phase=5
# ):
#     feedback_per_frame = {}

#     # Flatten arrays
#     expert_distances_arr = np.concatenate(expert_distances)
#     user_distances_arr = np.concatenate(user_distances)

#     # Identify push-offs and glides
#     expert_peaks_max = find_peaks(expert_distances_arr)[0]  # Push
#     expert_peaks_min = find_peaks(-expert_distances_arr)[0]  # Glide

#     # Combine and sort peaks to define full stride cycles
#     all_peaks = np.sort(np.concatenate([expert_peaks_min, expert_peaks_max]))

#     alpha = 0.7
#     beta = 0.3

#     def hip_diff(angles):
#         return angles[0] - angles[1]

#     for i in range(len(all_peaks) - 1):
#         start = all_peaks[i]
#         end = all_peaks[i + 1]
#         phase_type = "Glide→Push" if expert_distances_arr[start] < expert_distances_arr[end] else "Push→Glide"

#         # Sample expert indices between start and end
#         sampled_idxs = np.linspace(start, end, num_samples_per_phase, dtype=int)

#         for expert_idx in sampled_idxs:
#             # Find matched user frames from DTW path
#             user_frame_indices = [pair[0] for pair in path if pair[1] == expert_idx]
#             if not user_frame_indices:
#                 continue

#             # Average user data
#             avg_user_angles = np.mean([user_angles[i] for i in user_frame_indices], axis=0)
#             avg_user_dist = np.mean([user_distances_arr[i] for i in user_frame_indices])

#             # Expert data
#             expert_diff = hip_diff(expert_angles[expert_idx])
#             expert_dist = expert_distances_arr[expert_idx]
#             expert_score = alpha * expert_diff + beta * expert_dist

#             # User data
#             user_diff = hip_diff(avg_user_angles)
#             user_score = alpha * user_diff + beta * avg_user_dist

#             error = abs(user_score - expert_score)
#             print(error)
#             feedback = generate_feedback_comment(phase_type, error)
#             print(feedback)
            
#             feedback_per_frame[expert_idx] = feedback

#             # # Feedback per user frame
#             # for user_idx in user_frame_indices:
#             #     feedback_per_frame[user_idx] = {
#             #         "phase": phase_type,
#             #         "user_score": user_score,
#             #         "expert_score": expert_score,
#             #         "error": error,
#             #         "comment": generate_feedback_comment(phase_type, error)
#             #     }

#     return feedback_per_frame


# def generate_feedback_comment(phase, error, low=5.0, med=15.0):
#     if error < low:
#         return f"{phase}: Excellent weight shift — very close to expert."
#     elif error < med:
#         return f"{phase}: Decent weight transfer — consider smoother motion or fuller extension."
#     else:
#         return f"{phase}: Needs improvement — weight shift is off compared to expert."

# def feedback_shift_weight(
#     expert_distances, user_distances, diff_distances,
#     expert_angles, user_angles, diff_angles,
#     expert_horizontal_lines, user_horizontal_lines, diff_lines_horizontal,
#     path, feedback_range,
#     alpha=0.7,
#     beta=0.3,
#     error_threshold=10.0  # threshold for "close enough"
# ):
#     feedback_per_phase = {}

#     expert_distances_arr = np.concatenate(expert_distances)
#     user_distances_arr = np.concatenate(user_distances)

#     expert_peaks_max = find_peaks(expert_distances_arr)[0]
#     expert_peaks_min = find_peaks(-expert_distances_arr)[0]
#     all_peaks = np.sort(np.concatenate([expert_peaks_min, expert_peaks_max]))

#     def hip_diff(angles):
#         return angles[0] - angles[1]

#     for i in range(len(all_peaks) - 1):
#         start = all_peaks[i]
#         end = all_peaks[i + 1]

#         if start == end:
#             continue

#         direction = "glide → push" if expert_distances_arr[start] < expert_distances_arr[end] else "push → glide"
#         expert_idxs = np.arange(start, end + 1)
#         matched_pairs = [pair for pair in path if start <= pair[1] <= end]

#         if not matched_pairs:
#             continue

#         good_match_count = 0
#         total_matches = len(matched_pairs)

#         for user_idx, expert_idx in matched_pairs:
#             expert_diff = hip_diff(expert_angles[expert_idx])
#             expert_dist = expert_distances_arr[expert_idx]
#             expert_score = alpha * expert_diff + beta * expert_dist

#             user_diff = hip_diff(user_angles[user_idx])
#             user_dist = user_distances_arr[user_idx]
#             user_score = alpha * user_diff + beta * user_dist

#             error = abs(user_score - expert_score)
#             #print(error)
#             if error < error_threshold:
#                 good_match_count += 1

#         match_percentage = good_match_count / total_matches * 100
#         feedback = generate_feedback_consistency_comment(direction, match_percentage)
#         feedback_per_phase[start] = feedback
#         print(feedback)

#         # feedback = {
#         #     "phase": direction,
#         #     "frame_range": f"{start}-{end}",
#         #     "match_percentage": round(match_percentage, 1),
#         #     "matched_frames": total_matches,
#         #     "good_matches": good_match_count,
#         #     "comment": generate_feedback_consistency_comment(direction, match_percentage),
#         #     "expert_frames": list(expert_idxs),
#         #     "user_frames": [pair[0] for pair in matched_pairs]
#         # }
#         # feedback_per_phase.append(feedback)

#     return feedback_per_phase


# def generate_feedback_consistency_comment(phase, percentage, good=75, okay=50):
#     if percentage >= good:
#         return f"{phase.capitalize()}: Excellent consistency — strong technique!"
#     elif percentage >= okay:
#         return f"{phase.capitalize()}: Decent execution, but improve timing or posture."
#     else:
#         return f"{phase.capitalize()}: Needs work — weight shift is off from expert motion."

def feedback_shift_weight(
    expert_distances, user_distances, diff_distances,
    expert_angles, user_angles, diff_angles,
    expert_horizontal_lines, user_horizontal_lines, diff_lines_horizontal,
    path, feedback_range,
    alpha=0.85,
    beta=0.15,
    error_threshold=10.0,  # threshold for "good match"
    window_size=10  # ±5 frames around each peak
):
    feedback_per_phase = {}

    expert_distances_arr = np.concatenate(expert_distances)
    user_distances_arr = np.concatenate(user_distances)

    expert_peaks_max = find_peaks(expert_distances_arr)[0]
    expert_peaks_min = find_peaks(-expert_distances_arr)[0]
    all_peaks = np.sort(np.concatenate([expert_peaks_min, expert_peaks_max]))

    def hip_diff(angles):
        return angles[0] - angles[1]

    for peak_idx in all_peaks:
        # Define the window around each peak (±5 frames)
        start = max(0, peak_idx - window_size)
        end = min(len(expert_distances_arr) - 1, peak_idx + window_size)

        # Determine if it's glide → push or push → glide based on min or max
        if peak_idx in expert_peaks_max:
            direction = "push → glide"
        else:
            direction = "glide → push"

        # Collect the expert frames and the matched user frames in this window
        expert_idxs = np.arange(start, end + 1)
        matched_pairs = [pair for pair in path if start <= pair[1] <= end]

        if not matched_pairs:
            continue

        good_match_count = 0
        total_matches = len(matched_pairs)

        for user_idx, expert_idx in matched_pairs:
            # Calculate score for expert and user at this frame
            expert_diff = hip_diff(expert_angles[expert_idx])
            expert_dist = expert_distances_arr[expert_idx]
            expert_score = alpha * expert_diff + beta * expert_dist

            user_diff = hip_diff(user_angles[user_idx])
            user_dist = user_distances_arr[user_idx]
            user_score = alpha * user_diff + beta * user_dist

            # Calculate the error between user and expert
            error = abs(user_score - expert_score)
            if error < error_threshold:
                good_match_count += 1

        match_percentage = good_match_count / total_matches * 100
        feedback = generate_feedback_consistency_comment(direction, match_percentage)
        save_feedback(feedback_per_phase, peak_idx, feedback, 5)
        print(feedback)

        # # Prepare feedback
        # feedback = {
        #     "phase": direction,
        #     "peak_idx": peak_idx,
        #     "frame_range": f"{start}-{end}",
        #     "match_percentage": round(match_percentage, 1),
        #     "matched_frames": total_matches,
        #     "good_matches": good_match_count,
        #     "comment": generate_feedback_consistency_comment(direction, match_percentage),
        #     "expert_frames": list(expert_idxs),
        #     "user_frames": [pair[0] for pair in matched_pairs]
        # }
        # feedback_per_phase.append(feedback)

    return feedback_per_phase


def generate_feedback_consistency_comment(phase, percentage, good=75, okay=50):
    if percentage >= good:
        return f"{phase.capitalize()}: Excellent consistency — strong technique!"
    elif percentage >= okay:
        return f"{phase.capitalize()}: Decent execution, but improve timing or posture."
    else:
        return f"{phase.capitalize()}: Needs work — weight shift is off from expert motion."