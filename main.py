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
