import pygame
import pymunk
import pymunk.pygame_util
import math
import torch

import main

pygame.init()

WIDTH, HEIGHT = 1000, 800
FORCE_MAGNITUDE = 2000
MAX_VELOCITY = 100

window = pygame.display.set_mode((WIDTH, HEIGHT))

def draw(window, space, draw_options):
    window.fill("white")
    space.debug_draw(draw_options)
    pygame.display.update()


def create_boundaries(space, width, height):

    rects = [
        [(width/2, height - 10), (width, 20)],
        [(width / 2, 10), (width, 20)],
        [(10, 0), (20, height / 2)],
        [(10, height * 7/4), (20, height * 2)],
        [(width - 10, 0), (20, height / 2)],
        [(width - 10, height * 7 / 4), (20, height * 2)],

        [(-10, height / 2 - 20), (0, height)],
        [(width+10, height / 2 - 20), (0, height)]
    ]

    for pos, size in rects:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.8
        space.add(body, shape)


def create_circle(space, radius, mass, position, color):
    body = pymunk.Body()
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.mass = mass
    shape.color = color
    shape.elasticity = 0.95
    space.add(body, shape)
    return shape


def run(window, model1, model2, model3, model4, width, height):
    run = True
    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps

    space = pymunk.Space()
    space.gravity = (0, 0)

    agent1 = create_circle(space, 30, 10, (300, 300), (255, 0, 0, 100))
    agent2 = create_circle(space, 30, 10, (300, 400), (255, 0, 0, 100))

    agent3 = create_circle(space, 30, 10, (WIDTH - 300, 300), (255, 125, 180, 100))
    agent4 = create_circle(space, 30, 10, (WIDTH - 300, 400), (255, 125, 180, 100))

    soccer_ball = create_circle(space, 15, 4, (500, 400), (0, 0, 0, 100))
    create_boundaries(space, width, height)

    draw_options = pymunk.pygame_util.DrawOptions(window)


    while run:

        input_vector = []

        input_vector.extend([(agent1.body.position.x / WIDTH) * 2 - 1, (agent1.body.position.y / HEIGHT) * 2 - 1])
        input_vector.extend(
            [(agent1.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent1.body.velocity.y / MAX_VELOCITY) * 2 - 1])

        input_vector.extend([(agent2.body.position.x / WIDTH) * 2 - 1, (agent2.body.position.y / HEIGHT) * 2 - 1])
        input_vector.extend(
            [(agent2.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent2.body.velocity.y / MAX_VELOCITY) * 2 - 1])

        input_vector.extend([(agent3.body.position.x / WIDTH) * 2 - 1, (agent3.body.position.y / HEIGHT) * 2 - 1])
        input_vector.extend(
            [(agent3.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent3.body.velocity.y / MAX_VELOCITY) * 2 - 1])

        input_vector.extend([(agent4.body.position.x / WIDTH) * 2 - 1, (agent4.body.position.y / HEIGHT) * 2 - 1])
        input_vector.extend(
            [(agent4.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent4.body.velocity.y / MAX_VELOCITY) * 2 - 1])

        input_vector.extend(
            [(soccer_ball.body.position.x / WIDTH) * 2 - 1, (soccer_ball.body.position.y / HEIGHT) * 2 - 1])
        input_vector.extend([(soccer_ball.body.velocity.x / MAX_VELOCITY) * 2 - 1,
                             (soccer_ball.body.velocity.y / MAX_VELOCITY) * 2 - 1])

        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

        y1 = model1.forward(input_tensor)
        y2 = model2.forward(input_tensor)
        y3 = model3.forward(input_tensor)
        y4 = model4.forward(input_tensor)

        agent1_force_x = FORCE_MAGNITUDE * y1[0] * math.cos(y1[1])
        agent1_force_y = FORCE_MAGNITUDE * y1[0] * math.sin(y1[1])
        agent1.body.apply_impulse_at_local_point((agent1_force_x, agent1_force_y), (0, 0))

        agent2_force_x = FORCE_MAGNITUDE * y2[0] * math.cos(y2[1])
        agent2_force_y = FORCE_MAGNITUDE * y2[0] * math.sin(y2[1])
        agent2.body.apply_impulse_at_local_point((agent2_force_x, agent2_force_y), (0, 0))

        agent3_force_x = FORCE_MAGNITUDE * y3[0] * math.cos(y3[1])
        agent3_force_y = FORCE_MAGNITUDE * y3[0] * math.sin(y3[1])
        agent3.body.apply_impulse_at_local_point((agent3_force_x, agent3_force_y), (0, 0))

        agent4_force_x = FORCE_MAGNITUDE * y4[0] * math.cos(y4[1])
        agent4_force_y = FORCE_MAGNITUDE * y4[0] * math.sin(y4[1])
        agent4.body.apply_impulse_at_local_point((agent4_force_x, agent4_force_y), (0, 0))

        agent1.body.velocity = agent1.body.velocity * .99
        agent2.body.velocity = agent2.body.velocity * .99

        agent3.body.velocity = agent3.body.velocity * .99
        agent4.body.velocity = agent4.body.velocity * .99

        soccer_ball.body.velocity = soccer_ball.body.velocity * .995

        if soccer_ball.body.position[0] < 5 or soccer_ball.body.position[0] > WIDTH - 5:
            print("Goal scored")
            run = False
            break

        draw(window, space, draw_options)
        space.step(dt)
        clock.tick(fps)

    pygame.quit()

model1 = main.NueralNetwork()
model2 = main.NueralNetwork()
model3 = main.NueralNetwork()
model4 = main.NueralNetwork()

run(window, model1, model2, model3, model4, WIDTH, HEIGHT)