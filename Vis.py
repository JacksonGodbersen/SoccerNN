import pygame
import pymunk
import pymunk.pygame_util
import math

pygame.init()

WIDTH, HEIGHT = 1000, 800
FORCE_MAGNITUDE = 2000

window = pygame.display.set_mode((WIDTH, HEIGHT))

def draw(window, space, draw_options):
    window.fill("white")
    space.debug_draw(draw_options)
    pygame.display.update()


def create_boundaries(space, width, height):
    goal_size = 100
    # rects = [
    #     [(width / 2, height - 10), (width, 20)],
    #     [(width / 2, 10), (width, 20)],
    #     [(10, height / 2), (20, height)],
    #     [(width - 10, height / 2), (20, height)]
    # ]

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


def run(window, width, height):
    run = True
    clock = pygame.time.Clock()
    fps = 60
    dt = 1 / fps

    space = pymunk.Space()
    space.gravity = (0, 0)

    circle = create_circle(space, 30, 10, (300, 300), (255, 0, 0, 100))
    agent2 = create_circle(space, 30, 10, (300, 400), (255, 0, 0, 100))

    agent3 = create_circle(space, 30, 10, (WIDTH - 300, 300), (255, 125, 180, 100))
    agent4 = create_circle(space, 30, 10, (WIDTH - 300, 400), (255, 125, 180, 100))

    soccer_ball = create_circle(space, 15, 4, (500, 400), (0, 0, 0, 100))
    create_boundaries(space, width, height)

    draw_options = pymunk.pygame_util.DrawOptions(window)


    while run:
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

if __name__ == "__main__":
    run(window, WIDTH, HEIGHT)