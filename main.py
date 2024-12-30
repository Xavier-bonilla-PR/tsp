# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import combinations, permutations
# import random

# # Set random seed for reproducibility
# np.random.seed(42)

# # Generate 7 random points
# num_points = 7
# points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)]

# # Calculate distances between all pairs of points
# distances = []
# point_pairs = []
# for (i, p1), (j, p2) in combinations(enumerate(points), 2):
#     distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
#     distances.append(distance)
#     point_pairs.append((i, j))

# # Create distance matrix for TSP
# distance_matrix = np.zeros((num_points, num_points))
# for (i, j), dist in zip(point_pairs, distances):
#     distance_matrix[i, j] = dist
#     distance_matrix[j, i] = dist

# # Brute force TSP to find absolute shortest path
# def optimal_tsp(distances, start=0):
#     n = len(distances)
#     # Generate all possible permutations of points excluding the start point
#     other_points = list(range(1, n))
#     min_path = None
#     min_distance = float('inf')
    
#     for perm in permutations(other_points):
#         path = (start,) + perm + (start,)
#         distance = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))
        
#         if distance < min_distance:
#             min_distance = distance
#             min_path = path
    
#     return list(min_path), min_distance

# # Get optimal TSP path and distance
# tsp_path, total_tsp_distance = optimal_tsp(distance_matrix)

# # Create figure with three subplots
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# # Plot points and all connections
# ax1.scatter([p[0] for p in points], [p[1] for p in points], c='blue', s=100)
# for i, (p1, p2) in enumerate(zip(*zip(*point_pairs))):
#     x1, y1 = points[p1]
#     x2, y2 = points[p2]
#     line = ax1.plot([x1, x2], [y1, y2], 'gray', alpha=0.3)
#     # Add distance labels at midpoint
#     mid_x = (x1 + x2) / 2
#     mid_y = (y1 + y2) / 2
#     ax1.annotate(f'{distances[i]:.2f}', (mid_x, mid_y), alpha=0.5)

# # Number the points
# for i, (x, y) in enumerate(points):
#     ax1.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')

# ax1.set_title('All Points and Distances')
# ax1.grid(True)
# ax1.set_xlabel('X coordinate')
# ax1.set_ylabel('Y coordinate')

# # Create distance matrix heatmap
# im = ax2.imshow(distance_matrix, cmap='YlOrRd')
# plt.colorbar(im, ax=ax2, label='Distance')

# # Add distance values to cells
# for i in range(num_points):
#     for j in range(num_points):
#         if i != j:
#             text = ax2.text(j, i, f'{distance_matrix[i, j]:.2f}',
#                           ha='center', va='center')

# ax2.set_title('Distance Matrix Heatmap')
# ax2.set_xlabel('Point Number')
# ax2.set_ylabel('Point Number')
# ax2.set_xticks(range(num_points))
# ax2.set_yticks(range(num_points))
# ax2.set_xticklabels([f'P{i}' for i in range(num_points)])
# ax2.set_yticklabels([f'P{i}' for i in range(num_points)])

# # Plot optimal TSP path
# ax3.scatter([p[0] for p in points], [p[1] for p in points], c='blue', s=100)
# # Draw path segments with different colors to show order
# colors = plt.cm.rainbow(np.linspace(0, 1, len(tsp_path)-1))
# for i, color in enumerate(colors):
#     p1 = points[tsp_path[i]]
#     p2 = points[tsp_path[i + 1]]
#     ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color, linewidth=2)
#     # Add arrows and segment numbers
#     mid_x = (p1[0] + p2[0]) / 2
#     mid_y = (p1[1] + p2[1]) / 2
#     dx = p2[0] - p1[0]
#     dy = p2[1] - p1[1]
#     ax3.arrow(mid_x, mid_y, dx/10, dy/10, head_width=0.2, head_length=0.3, fc=color, ec=color)
#     ax3.text(mid_x, mid_y, str(i+1), bbox=dict(facecolor='white', alpha=0.7))

# # Number the points
# for i, (x, y) in enumerate(points):
#     ax3.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')

# ax3.set_title(f'Optimal TSP Path (Total Distance: {total_tsp_distance:.2f})')
# ax3.grid(True)
# ax3.set_xlabel('X coordinate')
# ax3.set_ylabel('Y coordinate')

# plt.tight_layout()
# plt.show()

# # Print the path and distance
# print("Optimal TSP Path:", ' -> '.join(f'P{i}' for i in tsp_path))
# print(f"Total Distance: {total_tsp_distance:.2f}")

###################################
#Shows ratio bar chart and optimal and similar ratio path
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import combinations, permutations
# import random

# # Set random seed for reproducibility
# np.random.seed(42)

# # Generate 7 random points
# num_points = 7
# points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)]

# # Calculate distances between all pairs of points
# distances = []
# point_pairs = []
# for (i, p1), (j, p2) in combinations(enumerate(points), 2):
#     distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
#     distances.append(distance)
#     point_pairs.append((i, j))

# # Create distance matrix for TSP
# distance_matrix = np.zeros((num_points, num_points))
# for (i, j), dist in zip(point_pairs, distances):
#     distance_matrix[i, j] = dist
#     distance_matrix[j, i] = dist

# def calculate_continuous_ratios(path, distances):
#     """Calculate the ratios between consecutive line segments."""
#     path_distances = []
#     ratios = []
    
#     # Calculate distances for each segment in the path
#     for i in range(len(path)-1):
#         dist = distance_matrix[path[i]][path[i+1]]
#         path_distances.append(dist)
    
#     # Calculate ratios between consecutive segments
#     for i in range(len(path_distances)-1):
#         ratio = path_distances[i+1] / path_distances[i]
#         ratios.append(ratio)
    
#     return path_distances, ratios

# def evaluate_path_continuity(ratios):
#     """Calculate how well the ratios follow a continuous pattern."""
#     if not ratios:
#         return float('inf')
#     return np.var(ratios)

# def find_three_different_paths(distances, start=0):
#     n = len(distances)
#     other_points = list(range(1, n))
#     paths = []
#     path_metrics = []
    
#     for perm in permutations(other_points):
#         path = (start,) + perm + (start,)
#         distance = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))
#         _, ratios = calculate_continuous_ratios(path, distances)
#         continuity_score = evaluate_path_continuity(ratios)
        
#         path_metrics.append((list(path), distance, continuity_score))
    
#     # Sort by different criteria to get diverse paths
#     # 1. Best balance of distance and continuity
#     sorted_by_balance = sorted(path_metrics, key=lambda x: x[1] * x[2])
#     # 2. Best distance
#     sorted_by_distance = sorted(path_metrics, key=lambda x: x[1])
#     # 3. Best continuity
#     sorted_by_continuity = sorted(path_metrics, key=lambda x: x[2])
    
#     return [sorted_by_balance[0], sorted_by_distance[0], sorted_by_continuity[0]]

# # Get three different paths
# paths = find_three_different_paths(distance_matrix)
# path_types = ["Balanced Path", "Shortest Path", "Most Continuous Path"]

# # Create visualization
# fig = plt.figure(figsize=(20, 10))
# gs = plt.GridSpec(2, 3)

# # Plot the three paths
# for idx, (path, distance, continuity) in enumerate(paths):
#     ax = fig.add_subplot(gs[0, idx])
    
#     # Calculate path distances and ratios
#     path_distances, ratios = calculate_continuous_ratios(path, distance_matrix)
    
#     # Plot points and path
#     ax.scatter([p[0] for p in points], [p[1] for p in points], c='blue', s=100)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(path)-1))
    
#     for i, color in enumerate(colors):
#         p1 = points[path[i]]
#         p2 = points[path[i + 1]]
#         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color, linewidth=2)
#         # Add segment lengths
#         mid_x = (p1[0] + p2[0]) / 2
#         mid_y = (p1[1] + p2[1]) / 2
#         ax.text(mid_x, mid_y, f'{path_distances[i]:.2f}', 
#                 bbox=dict(facecolor='white', alpha=0.7))
    
#     # Number the points
#     for i, (x, y) in enumerate(points):
#         ax.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
#     ax.set_title(f'{path_types[idx]}\nDistance: {distance:.2f}, Variance: {continuity:.2f}')
#     ax.grid(True)
#     ax.set_xlabel('X coordinate')
#     ax.set_ylabel('Y coordinate')

#     # Plot ratio bar charts
#     ax_ratio = fig.add_subplot(gs[1, idx])
#     segment_nums = range(1, len(ratios) + 1)
#     ax_ratio.bar(segment_nums, ratios)
#     ax_ratio.set_title('Ratios Between Consecutive Segments')
#     ax_ratio.set_xlabel('Segment Number')
#     ax_ratio.set_ylabel('Ratio (Length[n+1] / Length[n])')
#     ax_ratio.grid(True)
    
#     # Print detailed information
#     print(f"\n{path_types[idx]}:")
#     print("Path:", ' -> '.join(f'P{i}' for i in path))
#     print("Segment Lengths:", ', '.join(f'{d:.2f}' for d in path_distances))
#     print("Segment Ratios:", ', '.join(f'{r:.2f}' for r in ratios))
#     print(f"Total Distance: {distance:.2f}")
#     print(f"Ratio Variance: {continuity:.2f}")

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import random

# Set random seed for reproducibility
np.random.seed(42)

def calculate_path_metrics(path, distance_matrix):
    """Calculate distance and ratios for a path."""
    segments = []
    total_distance = 0
    for i in range(len(path)-1):
        dist = distance_matrix[path[i]][path[i+1]]
        segments.append(dist)
        total_distance += dist
    
    ratios = []
    for i in range(len(segments)-1):
        ratio = segments[i+1] / segments[i]
        ratios.append(ratio)
    
    ratio_variance = np.var(ratios) if ratios else float('inf')
    
    return total_distance, segments, ratios, ratio_variance

# Generate points and distance matrix
num_points = 7
points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)]
distances = []
point_pairs = []
for (i, p1), (j, p2) in combinations(enumerate(points), 2):
    distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    distances.append(distance)
    point_pairs.append((i, j))

distance_matrix = np.zeros((num_points, num_points))
for (i, j), dist in zip(point_pairs, distances):
    distance_matrix[i, j] = dist
    distance_matrix[j, i] = dist

# Find all possible paths and their distances
all_paths = []
for perm in permutations(range(1, num_points)):
    path = (0,) + perm + (0,)
    distance, segments, ratios, variance = calculate_path_metrics(path, distance_matrix)
    all_paths.append((list(path), distance, segments, ratios, variance))

# Sort paths by distance only
all_paths.sort(key=lambda x: x[1])  # Sort by total distance

# Get shortest, middle, and longest paths
shortest_path = all_paths[0]
middle_path = all_paths[len(all_paths)//2]
longest_path = all_paths[-1]

paths = [shortest_path, middle_path, longest_path]
titles = ["Shortest Path (Most Optimal)", "Middle Path", "Longest Path (Least Optimal)"]

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

for idx, (path, distance, segments, ratios, variance) in enumerate(paths):
    # Plot path
    ax_path = axes[idx, 0]
    ax_path.scatter([p[0] for p in points], [p[1] for p in points], c='blue', s=100)
    
    # Draw path segments with lengths
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i in range(len(path)-1):
        p1 = points[path[i]]
        p2 = points[path[i+1]]
        ax_path.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i], linewidth=2)
        
        # Add segment length and order
        mid_x, mid_y = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2
        ax_path.text(mid_x, mid_y, f'{i+1}: {segments[i]:.2f}', 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Number points
    for i, (x, y) in enumerate(points):
        ax_path.annotate(f'P{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax_path.set_title(f'{titles[idx]}\nTotal Distance: {distance:.2f}\nVariance: {variance:.4f}')
    ax_path.grid(True)
    ax_path.set_xlabel('X coordinate')
    ax_path.set_ylabel('Y coordinate')
    
    # Plot ratios
    ax_ratio = axes[idx, 1]
    if ratios:
        bars = ax_ratio.bar(range(1, len(ratios)+1), ratios)
        
        # Add ratio values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax_ratio.text(bar.get_x() + bar.get_width()/2., height,
                         f'{ratios[i]:.2f}',
                         ha='center', va='bottom')
    
    ax_ratio.set_title('Ratios Between Consecutive Segments')
    ax_ratio.set_xlabel('Segment Pair')
    ax_ratio.set_ylabel('Ratio (Length[n+1] / Length[n])')
    ax_ratio.grid(True)
    
    # Print detailed analysis
    print(f"\n{titles[idx]}:")
    print(f"Path: {' -> '.join(f'P{i}' for i in path)}")
    print(f"Segment lengths: {', '.join(f'{s:.2f}' for s in segments)}")
    print(f"Segment ratios: {', '.join(f'{r:.2f}' for r in ratios)}")
    print(f"Variance of ratios: {variance:.4f}")
    print(f"Total distance: {distance:.2f}")

plt.tight_layout()
plt.show()
