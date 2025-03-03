import numpy as np
import cv2
from utils.dtw import smooth_cycle, compute_angle

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
            # TODO Change json to add Hip_x/Hip_y (the reference realtive too)?
            if joint1 == run_args.DTW.CHOOSEN_REF:
                p1 = (cycle_data[joint1 + "_x_ref"][i], cycle_data[joint1 + "_y_ref"][i])
            else:
                p1 = (cycle_data[joint1 + "_x"][i], cycle_data[joint1 + "_y"][i])
            p2 = (cycle_data[joint2 + "_x"][i], cycle_data[joint2 + "_y"][i])
            if joint3 == run_args.DTW.CHOOSEN_REF:
                p3 = (cycle_data[joint3 + "_x_ref"][i], cycle_data[joint3 + "_y_ref"][i])
            else:
                p3 = (cycle_data[joint3 + "_x"][i], cycle_data[joint3 + "_y"][i])
            p4 = (cycle_data[joint4 + "_x"][i], cycle_data[joint4 + "_y"][i])
            angles.append(compute_angle_between_lines(p1, p2, p3, p4))
        all_angles.append(angles)
        frames.append(i)
    return np.array(all_angles), frames

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

# def get_line_points(user_cycle, joints_list, frame1, frame2, run_args, expert_cycle = None):
#     points = []
#     for joints in joints_list:
#         #TODO Get ref from cfg/other way?
#         for joint in joints:
#             x_suffix = "_x"
#             y_suffix = "_y"
#             # TODO Fix because only ref in json
#             if joint == run_args.DTW.CHOOSEN_REF:
#                 x_suffix += "_ref"
#                 y_suffix += "_ref"
#                 if expert_cycle == None:
#                     p_x = int(user_cycle.get("Hip_x_ref")[frame1])
#                     p_y = int(user_cycle.get("Hip_y_ref")[frame1])
#                 else:
#                     # TODO Make this a param/overlay exper skier or draw lines on only user
#                     # p_x = int(expert_cycle.get(joint + x_suffix)[frame2] + user_cycle.get("Hip_x_ref")[frame1])
#                     # p_y = int(expert_cycle.get(joint + y_suffix)[frame2] + user_cycle.get("Hip_y_ref")[frame1])
#                     p_x = int(expert_cycle.get("Hip_x_ref")[frame2])
#                     p_y = int(expert_cycle.get("Hip_y_ref")[frame2])
#             else:
#                 if expert_cycle == None:
#                     p_x = int(user_cycle.get(joint + x_suffix)[frame1] + user_cycle.get("Hip_x_ref")[frame1])
#                     p_y = int(user_cycle.get(joint + y_suffix)[frame1] + user_cycle.get("Hip_y_ref")[frame1])
#                 else:
#                     # TODO Make this a param/overlay exper skier or draw lines on only user
#                     # p_x = int(expert_cycle.get(joint + x_suffix)[frame2] + user_cycle.get("Hip_x_ref")[frame1])
#                     # p_y = int(expert_cycle.get(joint + y_suffix)[frame2] + user_cycle.get("Hip_y_ref")[frame1])
#                     p_x = int(expert_cycle.get(joint + x_suffix)[frame2] + expert_cycle.get("Hip_x_ref")[frame2])
#                     p_y = int(expert_cycle.get(joint + y_suffix)[frame2] + expert_cycle.get("Hip_y_ref")[frame2])
#             points.append((p_x,p_y))
#     return points

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

# def draw_lines_and_text(user_frame, cycle, shoulder_hip_joints, frame1, frame2, expert_cycle, 
#                              user_lines, expert_lines, diff_user_expert, i, run_args):
#     """
#     Draws lines on the skier and annotates user and expert shoulder/hip angles along with their difference.

#     Parameters:
#     - user_frame: The frame on which the lines and text will be drawn.
#     - cycle: Data related to the user's cycle.
#     - shoulder_hip_joints: Joints information for the shoulder and hips.
#     - frame1: Frame index for the user data.
#     - frame2: Frame index for the expert data.
#     - expert_cycle: Data related to the expert's cycle.
#     - user_lines: Array of user shoulder/hip angles.
#     - expert_lines: Array of expert shoulder/hip angles.
#     - diff_user_expert: Difference between user and expert angles.
#     - i: The current index in the difference array.

#     Returns:
#     - user_frame: The frame with drawn lines and text.
#     """
#     # Get line points for user and expert
#     user_points = get_line_points(cycle, shoulder_hip_joints, frame1, frame2, run_args)
#     expert_points = get_line_points(cycle, shoulder_hip_joints, frame1, frame2, run_args, expert_cycle)

#     # Draw user lines (blue)
#     cv2.line(user_frame, user_points[0], user_points[1], color=(255, 0, 0), thickness=2)  # First pair
#     cv2.line(user_frame, user_points[2], user_points[3], color=(255, 0, 0), thickness=2)  # Second pair

#     # Draw expert lines (yellow)
#     cv2.line(user_frame, expert_points[0], expert_points[1], color=(255, 255, 0), thickness=2)  # First pair
#     cv2.line(user_frame, expert_points[2], expert_points[3], color=(255, 255, 0), thickness=2)  # Second pair

#     # Origin point for text (using hip reference point)
#     text_origin = (int(cycle.get("Hip_x_ref")[frame1]), int(cycle.get("Hip_y_ref")[frame1]))

#     # Offsets for text placement
#     x_offset = 30
#     y_offset = 20

#     # Text content (angles and difference)
#     user_text = "User angle between shoulder/hip: " + str(user_lines[frame1][0])
#     expert_text = "Expert angle between shoulder/hip: " + str(expert_lines[frame2][0])
#     diff_text = "Difference: " + str(diff_user_expert[i])

#     # Get the sizes of the text boxes
#     (text_width_user, text_height_user), _ = cv2.getTextSize(user_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
#     (text_width_expert, text_height_expert), _ = cv2.getTextSize(expert_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
#     (text_width_diff, text_height_diff), _ = cv2.getTextSize(diff_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)

#     # Draw background rectangles for the text (with a slight padding for better fit)
#     cv2.rectangle(user_frame, (text_origin[0] + x_offset, text_origin[1] - text_height_user - 5), 
#                   (text_origin[0] + x_offset + text_width_user + 5, text_origin[1] + 5), (0, 0, 0), -1)  # User text
#     cv2.rectangle(user_frame, (text_origin[0] + x_offset, text_origin[1] + y_offset - text_height_expert - 5), 
#                   (text_origin[0] + x_offset + text_width_expert + 5, text_origin[1] + y_offset + 5), (0, 0, 0), -1)  # Expert text
#     cv2.rectangle(user_frame, (text_origin[0] + x_offset, text_origin[1] - y_offset - text_height_diff - 5), 
#                   (text_origin[0] + x_offset + text_width_diff + 5, text_origin[1] - y_offset + 5), (0, 0, 0), -1)  # Difference text

#     # Draw the text onto the frame
#     cv2.putText(user_frame, user_text, (text_origin[0] + x_offset, text_origin[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
#     cv2.putText(user_frame, expert_text, (text_origin[0] + x_offset, text_origin[1] + y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
#     cv2.putText(user_frame, diff_text, (text_origin[0] + x_offset, text_origin[1] - y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

#     return user_frame

def draw_joint_lines(joints_lines, frame, points, l_color=(0,255,0), p_color=(0,0,255), l_thickness = 2, p_radius = 2):
    for i in range(len(joints_lines)):
        j = i*4
        cv2.line(frame, points[0+j], points[1+j], color=l_color, thickness=l_thickness)  # First pair
        cv2.line(frame, points[2+j], points[3+j], color=l_color, thickness=l_thickness)  # Second pair
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


def draw_table(frame, angles_tuple, lines_tuple, match, iter):

    # Unpack tuples
    joint_angles, user_angles, expert_angles, diff_angles, sim_angles = angles_tuple
    joint_lines, user_lines, expert_lines, diff_lines, sim_lines = lines_tuple

    height, width, _ = frame.shape
    rows = 0
    if len(joint_angles) > 0:
        rows += len(joint_angles) + 1 # +1 for header
    if len(joint_lines) > 0:
        rows += len(joint_lines) + 1 # +1 for header
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

    table_data = [['Angles', 'User', 'Expert', 'Difference', 'Similarity (%)']]
    for i, angle in enumerate(joint_angles):
        row = []
        row.append(str(angle))
        row.append(f"{user_angles[match[0]][i]:.2f}")
        row.append(f"{expert_angles[match[1]][i]:.2f}")
        row.append(f"{diff_angles[iter][i]:.2f}")
        row.append(f"{sim_angles[iter][i]:.2%}")
        table_data.append(row)

    table_data.append(['Lines', 'User', 'Expert', 'Difference', 'Similarity (%)'])
    for i, lines in enumerate(joint_lines):
        row = []
        row.append(str(lines))
        row.append(f"{user_lines[match[0]][i]:.2f}")
        row.append(f"{expert_lines[match[1]][i]:.2f}")
        row.append(f"{diff_lines[iter][i]:.2f}")
        row.append(f"{sim_lines[iter][i]:.2%}")
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
            # if col == 1:
            #     top_left = (col * cell_width + int(cell_width*0.5), row * cell_height)
            #     bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
            # else:
            #     top_left = (col * cell_width, row * cell_height)
            #     bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
                
            # Draw rectangle (cell border)
            cv2.rectangle(frame, top_left, bottom_right, border_colour, 2)  # Black border
            
            # Add text to the cell (centered)
            text = table_data[row][col]
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            if col == 0:
                text_x = top_left[0] + (cell_width + int(cell_width*0.5) - text_width) // 2
                text_y = top_left[1] + (cell_height + text_height) // 2
            else:
                text_x = top_left[0] + (cell_width - text_width) // 2
                text_y = top_left[1] + (cell_height + text_height) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame


