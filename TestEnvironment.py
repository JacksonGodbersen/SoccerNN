import pygame
import pymunk
import pymunk.pygame_util
import math
import torch

import PolicyNetwork
import main

pygame.init()

WIDTH, HEIGHT = 1000, 800
FORCE_MAGNITUDE = 500
MAX_VELOCITY = 1000

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

    circle = create_circle(space, 15, 10, (300, 300), (255, 0, 0, 100))
    agent2 = create_circle(space, 15, 10, (300, 400), (255, 0, 0, 100))

    agent3 = create_circle(space, 15, 10, (WIDTH - 300, 300), (255, 125, 180, 100))
    agent4 = create_circle(space, 15, 10, (WIDTH - 300, 400), (255, 125, 180, 100))

    soccer_ball = create_circle(space, 7.5, 4, (500, 400), (0, 0, 0, 100))
    create_boundaries(space, width, height)

    draw_options = pymunk.pygame_util.DrawOptions(window)


    while run:

        v1 = [(circle.body.position.x / WIDTH) * 2 - 1, (circle.body.position.y / HEIGHT) * 2 - 1, (circle.body.velocity.x / MAX_VELOCITY) * 2 - 1, (circle.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v2 = [(agent2.body.position.x / WIDTH) * 2 - 1, (agent2.body.position.y / HEIGHT) * 2 - 1, (agent2.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent2.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v3 = [(agent3.body.position.x / WIDTH) * 2 - 1, (agent3.body.position.y / HEIGHT) * 2 - 1, (agent3.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent3.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v4 = [(agent4.body.position.x / WIDTH) * 2 - 1, (agent4.body.position.y / HEIGHT) * 2 - 1, (agent4.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent4.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v5 = [(soccer_ball.body.position.x / WIDTH) * 2 - 1, (soccer_ball.body.position.y / HEIGHT) * 2 - 1, (soccer_ball.body.velocity.x / MAX_VELOCITY) * 2 - 1, (soccer_ball.body.velocity.y / MAX_VELOCITY) * 2 - 1]

        v1_flipped = [(-circle.body.position.x / WIDTH) * 2 - 1, (circle.body.position.y / HEIGHT) * 2 - 1,
              (-circle.body.velocity.x / MAX_VELOCITY) * 2 - 1, (circle.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v2_flipped = [(-agent2.body.position.x / WIDTH) * 2 - 1, (agent2.body.position.y / HEIGHT) * 2 - 1,
              (-agent2.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent2.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v3_flipped = [(-agent3.body.position.x / WIDTH) * 2 - 1, (agent3.body.position.y / HEIGHT) * 2 - 1,
              (-agent3.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent3.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v4_flipped = [(agent4.body.position.x / WIDTH) * 2 - 1, (agent4.body.position.y / HEIGHT) * 2 - 1,
              (-agent4.body.velocity.x / MAX_VELOCITY) * 2 - 1, (agent4.body.velocity.y / MAX_VELOCITY) * 2 - 1]
        v5_flipped = [(-soccer_ball.body.position.x / WIDTH) * 2 - 1, (soccer_ball.body.position.y / HEIGHT) * 2 - 1,
              (-soccer_ball.body.velocity.x / MAX_VELOCITY) * 2 - 1,
              (soccer_ball.body.velocity.y / MAX_VELOCITY) * 2 - 1]

        in1 = v1 + v2 + v3 + v4 + v5
        in2 = v2 + v1 + v3 + v4 + v5
        in3 = v3_flipped + v4_flipped + v1_flipped + v2_flipped + v5_flipped
        in4 = v4_flipped + v3_flipped + v1_flipped + v2_flipped + v5_flipped

        input_tensor1 = 1 * torch.tensor(in1, dtype=torch.float32).unsqueeze(0)
        input_tensor2 = 1 * torch.tensor(in2, dtype=torch.float32).unsqueeze(0)
        input_tensor3 = -1 * torch.tensor(in3, dtype=torch.float32).unsqueeze(0)
        input_tensor4 = -1 * torch.tensor(in4, dtype=torch.float32).unsqueeze(0)

        print(v1)

        y1 = model1.forward(input_tensor1)
        y2 = model2.forward(input_tensor2)
        y3 = model3.forward(input_tensor3)
        y4 = model4.forward(input_tensor4)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame.KEYDOWN:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        circle.body.apply_impulse_at_local_point((0, -FORCE_MAGNITUDE), (0, 0))
                    elif event.key == pygame.K_s:
                        circle.body.apply_impulse_at_local_point((0, FORCE_MAGNITUDE), (0, 0))
                    elif event.key == pygame.K_a:
                        circle.body.apply_impulse_at_local_point((-FORCE_MAGNITUDE, 0), (0, 0))
                    elif event.key == pygame.K_d:
                        circle.body.apply_impulse_at_local_point((FORCE_MAGNITUDE, 0), (0, 0))

        agent2_force_x = FORCE_MAGNITUDE * y2[0] * math.cos(y2[1])
        agent2_force_y = FORCE_MAGNITUDE * y2[0] * math.sin(y2[1])
        agent2.body.apply_impulse_at_local_point((agent2_force_x, agent2_force_y), (0, 0))

        agent3_force_x = FORCE_MAGNITUDE * y3[0] * math.cos(y3[1])
        agent3_force_y = FORCE_MAGNITUDE * y3[0] * math.sin(y3[1])
        agent3.body.apply_impulse_at_local_point((-agent3_force_x, agent3_force_y), (0, 0))

        agent4_force_x = FORCE_MAGNITUDE * y4[0] * math.cos(y4[1])
        agent4_force_y = FORCE_MAGNITUDE * y4[0] * math.sin(y4[1])
        agent4.body.apply_impulse_at_local_point((-agent4_force_x, agent4_force_y), (0, 0))

        circle.body.velocity = circle.body.velocity * .99
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

model1 = PolicyNetwork.NueralNetwork()
model2 = PolicyNetwork.NueralNetwork()
model3 = PolicyNetwork.NueralNetwork()
model4 = PolicyNetwork.NueralNetwork()

run(window, model1, model2, model3, model4, WIDTH, HEIGHT)