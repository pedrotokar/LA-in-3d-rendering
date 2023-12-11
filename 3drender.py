import pygame
import numpy as np
import math

# Função para ler arquivo obj e retornar dicionário representando as coordenadas, faces e vetores normais
def read_obj(path):
    object_ = {
        "coordinates": np.empty((4, 1)),
        "faces": [],
        "center_point": np.asarray((0, 0, 0, 1)),
        "norms": np.empty((4, 1))
    }
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v"):
                line = line.replace("v ", "")
                x, y, z = line.split(" ")
                x = float(x)
                y = float(y)
                z = float(z)
                array = np.asarray((x, y, z, 1,)).reshape(4, 1)
                object_["coordinates"] = np.hstack((object_["coordinates"], array))
            elif line.startswith("f"):
                line = line.replace("f ", "")
                points = line.split(" ")
                points = tuple(map(int, points))
                object_["faces"].append(points)
    object_["coordinates"] = object_["coordinates"][:, 1:]
    for index, face in enumerate(object_["faces"]):
        v1 = object_["coordinates"][0:3, face[0] - 1] - object_["coordinates"][0:3, face[1] - 1]
        v2 = object_["coordinates"][0:3, face[2] - 1] - object_["coordinates"][0:3, face[1] - 1]
        normal = np.cross(v1, v2)
        object_["norms"] = np.hstack((object_["norms"], np.asarray((*normal[0:3], 0)).reshape((4, 1))))
    object_["norms"] = object_["norms"][:, 1:]
    return object_

# Funções que retornam matrizes de rotação
def rotate_x(angle):
    angle = math.radians(angle)
    return np.asarray(
        ((1, 0              , 0               , 0),
         (0, math.cos(angle), -math.sin(angle), 0),
         (0, math.sin(angle), math.cos(angle) , 0),
         (0, 0              , 0               , 1))
    )

def rotate_y(angle):
    angle = math.radians(angle)
    return np.asarray(
        ((math.cos(angle) , 0, math.sin(angle), 0),
         (0               , 1, 0              , 0),
         (-math.sin(angle), 0, math.cos(angle), 0),
         (0               , 0, 0              , 1))
    )

def rotate_z(angle):
    angle = math.radians(angle)
    return np.asarray(
        ((math.cos(angle) , math.sin(angle), 0, 0),
         (-math.sin(angle), math.cos(angle), 0, 0),
         (0               , 0              , 1, 0),
         (0               , 0              , 0, 1))
    )

def rotate_around(vector, angle):
    if all(vector == np.asarray((0, 1, 0, 0))):
        return rotate_y(angle)

    vector = vector[0:3]
    up = np.asarray((0, 1, 0))
    right = np.cross(up, vector.reshape(3))
    up = np.cross(vector, right)

    up = up/np.linalg.norm(up)
    right = right/np.linalg.norm(right)
    vector = vector/np.linalg.norm(vector)

    m_matrix = np.vstack((vector.reshape(1, 3), right.reshape(1, 3), up.reshape(1, 3), np.zeros((1, 3))))
    m_matrix = np.hstack((m_matrix, np.asarray((0, 0, 0, 1)).reshape(4, 1)))
    return m_matrix.T @ rotate_x(angle) @ m_matrix

# Função que retorna matriz de translação
def translate(x_offset, y_offset, z_offset, w = None):
    return np.asarray(
        ((1, 0, 0, x_offset),
         (0, 1, 0, y_offset),
         (0, 0, 1, z_offset),
         (0, 0, 0, 1       ))
    )

# Função que retona matriz de escalonamento
def scale(x_scale, y_scale, z_scale, w = None):
    return np.asarray(
        ((x_scale, 0      , 0      , 0),
         (0      , y_scale, 0      , 0),
         (0      , 0      , z_scale, 0),
         (0      , 0      , 0      , 1))
    )

WIDTH = 1280 * 1.5
HEIGHT = 720 * 1.5

# Informações da posição da câmera
camera = {
    "position": np.asarray((100, 100, 100, 1), dtype = np.float64),
    "right": np.asarray((1, 0, 0, 0), dtype = np.float64),
    "up": np.asarray((0, 1, 0, 0), dtype = np.float64),
    "look_at": np.asarray((0, 0, 1, 0), dtype = np.float64),
}

# Informações sobre a construção do view frustum
projection_parameters = {
    "distance": 1000,
    "distance_2": 1000,
    "dimensions": (WIDTH, HEIGHT)
}

# Função que coloca a câmera na origem e remove faces não visiveis
def apply_camera_position(world, camera):
    rotated_matrix = np.hstack(
        (camera["right"].reshape(4, 1),
         camera["up"].reshape(4, 1),
         camera["look_at"].reshape(4, 1),
         np.asarray((0, 0, 0, 1)).reshape(4, 1))
        )
    view_matrix = rotated_matrix.T @ translate(*list(-camera["position"]))
    for object_ in world:
        object_["faces"] = object_["faces"].copy()
        for index, face in enumerate(object_["faces"].copy()):
            normal = object_["norms"][0:3, index]
            vision = object_["coordinates"][0:3, face[1] - 1] - camera["position"][0:3]
            if (normal @ vision) <= 0:
                object_["faces"].remove(face)
        object_["coordinates"] = view_matrix @ object_["coordinates"]
    return world

# Função que transforma o frustum em um cubo, e depois translaciona e escala para encaixar na tela
def apply_perspective_projection(world, projection_parameters):
    corner_1 = (-projection_parameters["dimensions"][0]/2, -projection_parameters["dimensions"][1]/2, -projection_parameters["distance"])
    corner_2 = (projection_parameters["dimensions"][0]/2, projection_parameters["dimensions"][1]/2, -projection_parameters["distance"])
    l = corner_1[0]
    b = corner_1[1]
    r = corner_2[0]
    t = corner_2[1]
    n = -projection_parameters["distance"]
    f = -projection_parameters["distance_2"] - projection_parameters["distance"]
    perspective_matrix = np.asarray(
        ((((2*n)/(r - l)), 0              , -((r + l)/(r - l)) , 0                 ),
         (0              , ((2*n)/(t - b)), -((t + b)/(t - b)) , 0                 ),
         (0              , 0              , ((f + n)/(f - n))  , -((2*f*n)/(f - n))),
         (0              , 0              , 1                  , 0                 ))
    )

    for object_ in world:
        object_["coordinates"] = perspective_matrix @ object_["coordinates"]
        for i in range(object_["coordinates"].shape[1]):
            object_["coordinates"][:, i] = object_["coordinates"][:, i]/object_["coordinates"][3, i]
        object_["coordinates"] = translate(WIDTH/2, HEIGHT/2, 0) @ scale(WIDTH/2, -HEIGHT/2, 1) @ object_["coordinates"] # Especificamente ajusta o cubo 2x2x2 para a tela
    return world


# Carrega alguns objetos e coloca em pontos do mundo
piramid_object = read_obj("piramide.obj")
piramid_object["coordinates"] = scale(20, 20, 20) @ translate(5, -5, -14) @ piramid_object["coordinates"]
piramid_object["center_point"] = scale(35, 35, 35) @ translate(5, -5, -14) @ piramid_object["center_point"]

cube_object = read_obj("cubo.obj")
cube_object["coordinates"] = scale(25, 25, 25) @ translate(0, 6, -16) @ cube_object["coordinates"]
cube_object["center_point"] = scale(25, 25, 25) @ translate(0, 6, -16) @ cube_object["center_point"]

teapot = read_obj("teapot.obj")
teapot["coordinates"] = scale(30, 30, 30) @ translate(5, 4, -5) @ teapot["coordinates"]
teapot["center_point"] = scale(30, 30, 30) @ translate(5, 4, -5) @ teapot["center_point"]

axis = read_obj("axis.obj")
axis["coordinates"] = scale(-5, 5, 5) @ axis["coordinates"]
axis["norms"] = scale(-5, 5, 5) @ axis["norms"]

cessna = read_obj("cessna.obj")
cessna["coordinates"] = scale(10, 10, 10) @ translate(-15, 10, -20) @ rotate_y(90) @ cessna["coordinates"]
cessna["norms"] = rotate_y(90) @ cessna["norms"]
cessna["center_point"] = scale(30, 30, 30) @ translate(5, 4, -5) @ cessna["center_point"]

# Linhas para iniciar o pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
dt = 0
font = pygame.font.SysFont('monospaced', 24)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Em todo frame, aplica transformações de rotação para alguns objetos
    cube_object["coordinates"] = translate(*(list(cube_object["center_point"])[0:3])) @ rotate_x(0.5) @ rotate_y(1) @ translate(*(list(cube_object["center_point"]*-1)[0:3])) @ cube_object["coordinates"]
    cube_object["norms"] = rotate_x(0.5) @ rotate_y(1) @ cube_object["norms"]
    teapot["coordinates"] = translate(*(list(teapot["center_point"])[0:3])) @ rotate_around(np.asarray((3, 4, -2, 0)), 1) @ translate(*(list(teapot["center_point"]*-1)[0:3])) @ teapot["coordinates"]
    teapot["norms"] = rotate_around(np.asarray((3, 4, -2, 0)), 1) @ teapot["norms"]

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        camera["position"] += -camera["look_at"] * 4
        CAMERA_MOD = True
    if keys[pygame.K_s]:
        camera["position"] += +camera["look_at"] * 4
        CAMERA_MOD = True
    if keys[pygame.K_a]:
        camera["position"] += -camera["right"] * 4
        CAMERA_MOD = True
    if keys[pygame.K_d]:
        camera["position"] += camera["right"] * 4
        CAMERA_MOD = True
    if keys[pygame.K_UP]:
        camera["position"] += camera["up"] * 4
        CAMERA_MOD = True
    if keys[pygame.K_DOWN]:
        camera["position"] += -camera["up"] * 4
        CAMERA_MOD = True
    if keys[pygame.K_LEFT]:
        camera["look_at"] = rotate_around(camera["up"], 2) @ camera["look_at"]
        camera["right"] = rotate_around(camera["up"], 2) @ camera["right"]
        CAMERA_MOD = True
    if keys[pygame.K_RIGHT]:
        camera["look_at"] = rotate_around(camera["up"], -2) @ camera["look_at"]
        camera["right"] = rotate_around(camera["up"], -2) @ camera["right"]
        CAMERA_MOD = True
    if keys[pygame.K_n]:
        camera["look_at"] = rotate_around(camera["right"], 2) @ camera["look_at"]
        camera["up"] = rotate_around(camera["right"], 2) @ camera["up"]
        CAMERA_MOD = True
    if keys[pygame.K_m]:
        camera["look_at"] = rotate_around(camera["right"], -2) @ camera["look_at"]
        camera["up"] = rotate_around(camera["right"], -2) @ camera["up"]
        CAMERA_MOD = True
    if keys[pygame.K_j]:
        camera["right"] = rotate_around(camera["look_at"], -2) @ camera["right"]
        camera["up"] = rotate_around(camera["look_at"], -2) @ camera["up"]
        CAMERA_MOD = True
    if keys[pygame.K_k]:
        camera["right"] = rotate_around(camera["look_at"], 1) @ camera["right"]
        camera["up"] = rotate_around(camera["look_at"], 1) @ camera["up"]
    if keys[pygame.K_u]:
        projection_parameters["distance"] += 5
    if keys[pygame.K_i]:
        projection_parameters["distance"] -= 5

    # Aplicação dos posicionamentos de câmera e da perspectiva
    world = [axis.copy(), teapot.copy(), cube_object.copy(), piramid_object.copy(), cessna.copy()]
    modified_world = apply_camera_position(world, camera)
    modified_world = apply_perspective_projection(modified_world, projection_parameters)

    # Desenha os polígonos
    screen.fill("white")
    for object_ in modified_world:
        def map_range(x, in_min, in_max, out_min, out_max):
            return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min
        for index, face in enumerate(object_["faces"]):
            for i in range(len(face) - 1):
                pygame.draw.line(screen, "black", object_["coordinates"][0:2, face[i] - 1], object_["coordinates"][0:2, face[i + 1] - 1])
            pygame.draw.line(screen, "black", object_["coordinates"][0:2, face[i + 1] - 1], object_["coordinates"][0:2, face[0] - 1])

    pygame.display.flip()
    dt = clock.tick(30) / 1000
