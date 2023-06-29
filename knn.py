import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

pygame.init()

screen_size = (500, 550)
game_size = (500, 500)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("KNN_Alpha_Version")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
step_size = 25

def draw_points(points, colors):
    for point, color in zip(points, colors):
        pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 5)

def draw_grid(area):
    x_min, y_min, x_max, y_max = area

    for x in range(int(x_min), int(x_max), step_size):
        pygame.draw.line(screen, (200, 200, 200), (x, y_min), (x, y_max), 1)

    for y in range(int(y_min), int(y_max), step_size):
        pygame.draw.line(screen, (200, 200, 200), (x_min, y), (x_max, y), 1)

def update_display(model, name, X, y, area):
    x_min, y_min, x_max, y_max = area

    model.fit(X, y)

    #cmap = ListedColormap(('red', 'blue'))

    xx = np.linspace(x_min, x_max, 100)
    yy = np.linspace(y_min, y_max, 100)
    x_mesh, y_mesh = np.meshgrid(xx, yy)
    z = model.predict(np.array([x_mesh.ravel(), y_mesh.ravel()]).T).reshape(x_mesh.shape)
    #print(z)
    
    screen.fill(WHITE)
    draw_grid(area)  

    for i in range(len(X)):
        color = RED if y[i] == 0 else BLUE
        pygame.draw.circle(screen, color, (int(X[i][0]), int(X[i][1])), 5)

    for i in range(len(xx)):
        for j in range(len(yy)):
            color = RED if z[j][i] == 0 else BLUE
            pygame.draw.circle(screen, color, (int(xx[i]), int(yy[j])), 1)

    red_points = np.count_nonzero(z == 0)
    #print(red_points)
    accuracy = red_points / len(z) 

    font = pygame.font.Font(None, 24)
    text = font.render(f"{name}, Red vs Blue = {accuracy}% : {100-accuracy}%", True, (0, 0, 0))
    screen.blit(text, (10, 520))

    pygame.display.flip()


running = True
points = []
colors = []
area = (0, 0, game_size[0], game_size[1])
draw_grid(area) 

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            grid_x = (x // step_size) * step_size
            grid_y = (y // step_size) * step_size
            if [grid_x, grid_y] not in points:
                points.append([grid_x, grid_y])
                colors.append(RED if len(points) % 2 == 1 else BLUE)
    
    screen.fill(WHITE)
    draw_grid(area) 
    draw_points(points, colors)  

    if len(points) >= 2:
        X = np.array(points)
        y = np.arange(len(points)) % 2
        model_game = KNeighborsClassifier(n_neighbors=1)
        area = (0, 0, game_size[0], game_size[1])
        update_display(model_game, "Alpha Version", X, y, area)
    
    pygame.display.update()


pygame.quit()
