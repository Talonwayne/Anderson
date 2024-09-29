import pygame, sys, math
from copy import deepcopy
import noise

# Initialize pygame and setup the display
clock = pygame.time.Clock()
from pygame.locals import *

pygame.init()
pygame.display.set_caption('3D Terrain')
screen = pygame.display.set_mode((500, 500), 0, 32)

# Camera position, rotation, and movement speed
camera_pos = [0, 0, 4.5]  # [x, y, z]
camera_rotation = 0  # Y-axis rotation in degrees
camera_speed = 0.2  # Movement speed
rotation_speed = 2  # Rotation speed

# Field of view and fog settings
FOV = 90
FOG = False


# Function to offset a polygon in 3D space
def offset_polygon(polygon, offset):
    for point in polygon:
        point[0] += offset[0]
        point[1] += offset[1]
        point[2] += offset[2]


# Function to project a 3D polygon to 2D space
def project_polygon(polygon):
    projected_points = []
    for point in polygon:
        x_angle = math.atan2(point[0], point[2])
        y_angle = math.atan2(point[1], point[2])
        x = x_angle / math.radians(FOV) * screen.get_width() + screen.get_height() // 2
        y = y_angle / math.radians(FOV) * screen.get_width() + screen.get_width() // 2
        projected_points.append([x, y])
    return projected_points


# Function to generate and project a polygon based on camera position and rotation
def gen_polygon(polygon_base, polygon_data, camera_pos, camera_rotation):
    generated_polygon = deepcopy(polygon_base)

    # Adjust the polygon position by subtracting the camera position
    offset_polygon(generated_polygon, [polygon_data['pos'][0] - camera_pos[0],
                                       polygon_data['pos'][1] - camera_pos[1],
                                       polygon_data['pos'][2] - camera_pos[2]])

    # Rotate polygon based on camera rotation
    for point in generated_polygon:
        # Rotate around the Y-axis
        x, z = point[0], point[2]
        point[0] = x * math.cos(math.radians(camera_rotation)) - z * math.sin(math.radians(camera_rotation))
        point[2] = x * math.sin(math.radians(camera_rotation)) + z * math.cos(math.radians(camera_rotation))

    return project_polygon(generated_polygon)


# Base polygon data
poly_data = {'pos': [0, 0, 4.5], 'rot': [0, 0, 0]}

# Base square polygon used to generate the terrain
square_polygon = [
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
]

polygons = []


# Function to generate a row of polygons for the terrain
def generate_poly_row(y):
    global polygons
    for x in range(30):
        poly_copy = deepcopy(square_polygon)
        offset_polygon(poly_copy, [x - 15, 5, y + 5])
        water = True
        depth = 0
        for corner in poly_copy:
            v = noise.pnoise2(corner[0] / 10, corner[2] / 10, octaves=2) * 3
            v2 = noise.pnoise2(corner[0] / 30 + 1000, corner[2] / 30)
            if v < 0:
                depth -= v
                v = 0
            else:
                water = False
            corner[1] -= v * 4.5

        # Handle coloring
        if water:
            c = (0, min(255, max(0, 150 - depth * 25)), min(255, max(0, 255 - depth * 25)))
        else:
            c = (30 - v * 10 + v2 * 30, 50 + v2 * 40 + v * 30, 50 + v * 10)

        # Add polygon to the front of the list
        polygons = [[poly_copy, c]] + polygons


# Generate initial terrain
next_row = 0
for y in range(26):
    generate_poly_row(y)
    next_row += 1


# Function to dynamically generate terrain based on camera movement
def check_and_generate_terrain():
    global next_row, polygons
    # Generate new rows in front of the camera if needed
    while polygons[-1][0][0][2] < -poly_data['pos'][2] - camera_pos[2]:
        for i in range(30):
            polygons.pop(len(polygons) - 1)
        generate_poly_row(next_row)
        next_row += 1


# Main loop
while True:
    # Update display and background
    bg_surf = pygame.Surface(screen.get_size())
    bg_surf.fill((100, 200, 250))
    display = screen.copy()
    display.blit(bg_surf, (0, 0))
    bg_surf.set_alpha(120)

    # Check and generate new terrain rows based on camera position
    check_and_generate_terrain()

    # Render each polygon with the updated camera position and rotation
    for i, polygon in enumerate(polygons):
        if FOG:
            if (i % 90 == 0) and (i != 0) and (i < 30 * 18):
                display.blit(bg_surf, (0, 0))
        render_poly = gen_polygon(polygon[0], poly_data, camera_pos, camera_rotation)
        poly2 = deepcopy(render_poly)
        for v in poly2:
            v[1] = 100 - v[1] * 0.2
            v[0] = 500 - v[0]
        pygame.draw.polygon(display, polygon[1], render_poly)
        d = polygon[0][0][1]
        if d < 5:
            pygame.draw.polygon(display, (
            min(max(0, d * 20) + 150, 255), min(max(0, d * 20) + 150, 255), min(max(0, d * 20) + 150, 255)), poly2)

    display.set_alpha(150)
    screen.blit(display, (0, 0))

    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

    # Camera movement and rotation
    keys = pygame.key.get_pressed()
    if keys[K_w]:  # Move forward
        camera_pos[2] -= camera_speed * math.cos(math.radians(camera_rotation))
        camera_pos[0] -= camera_speed * math.sin(math.radians(camera_rotation))
    if keys[K_s]:  # Move backward
        camera_pos[2] += camera_speed * math.cos(math.radians(camera_rotation))
        camera_pos[0] += camera_speed * math.sin(math.radians(camera_rotation))
    if keys[K_a]:  # Rotate left
        camera_rotation += rotation_speed
    if keys[K_d]:  # Rotate right
        camera_rotation -= rotation_speed

    # Update display
    pygame.display.update()
    clock.tick(60)
