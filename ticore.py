# !/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import taichi as ti


ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32)

# ================================================common================================================
res = (960, 540)
# res = (1920, 1080)
t = 0.0
fps = 60.0
dt = 1.0 / fps
pixels = ti.Vector.field(n=4, dtype=float, shape=res)  # write color [rgba]
pixels_dis = ti.Vector.field(n=3, dtype=float, shape=res)  # display [rgb]
clear_color = ti.Vector([0.12, 0.12, 0.12, 1.0])

# ================================================star================================================
fov = 120
tan_half_fov = math.tan(fov / 360 * math.pi)
z_near, z_far, grid_size = 200, 3200, 120
N = (res[0]//grid_size, res[1]//grid_size, (z_far - z_near) // grid_size)
pos = ti.Vector.field(n=3, dtype=float, shape=N)
color = ti.Vector.field(n=3, dtype=float, shape=N)
vel = ti.Vector.field(n=3, dtype=float, shape=())


@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.func
def rand4():
    return ti.Vector([ti.random(), ti.random(), ti.random(), 1.0])


@ti.func
def clamp(x, low=0.0, high=1.0):
    ret = x
    if x > high:
        ret = high
    elif x < low:
        ret = low
    return ret


@ti.func
def sign(x):
    if x > 0:
        x = 1
    elif x < 0.0:
        x = -1.0
    return x


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def mix(x, y, a):
    # x * (1.0 - a) + y * a
    return x * (1.0 - a) + y * a


@ti.func
def Union(d1, d2):
    return min(d1, d2)


@ti.func
def Subtraction(d1, d2):
    # return max(-d1, d2) # maybe has erro.
    return max(d1, -d2)


@ti.func
def Intersection(d1, d2):
    return max(d1, d2)


@ti.func
def SmoothUnion(d1, d2, k):
    h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return mix(d2, d1, h) - k * h * (1.0 - h)


@ti.func
def SmoothSubstraction(d1, d2, k):
    h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return mix(d2, -d1, h) - k * h * (1.0 - h)


@ti.func
def SmoothIntersection(d1, d2, k):
    h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return mix(d2, d1, h) - k * h * (1.0 - h)
# ==================================================================


@ti.func
def combine(x, y):
    # combine RGBA color. y : high level, x : low level
    col = ti.Vector([0.0, 0.0, 0.0, 0.0])
    r = y[0] * y[3] + x[0] * x[3] * (1.0 - y[3])
    g = y[1] * y[3] + x[1] * x[3] * (1.0 - y[3])
    b = y[2] * y[3] + x[2] * x[3] * (1.0 - y[3])
    a = 1.0 - (1.0 - y[3]) * (1.0 - x[3])
    r = r / a
    g = g / a
    b = b / a
    r = clamp(r)
    g = clamp(g)
    b = clamp(b)
    a = clamp(a)
    col = ti.Vector([r, g, b, a])
    return col


@ti.func
def sdf_star(uv, rf, rad):
    # generate distance field of 2D-stars
    k1 = ti.Vector([0.809016994375, -0.587785252292])
    k2 = ti.Vector([-k1[0], k1[1]])
    uv[0] = ti.abs(uv[0])
    uv -= 2.0 * ti.max(k1.dot(uv), 0.0) * k1
    uv -= 2.0 * ti.max(k2.dot(uv), 0.0) * k2
    uv[0] = ti.abs(uv[0])
    uv[1] -= rad

    ba = rf * ti.Vector([-k1[1], k1[0]]) - ti.Vector([0, 1])
    h = uv.dot(ba) / ba.dot(ba)
    h = ti.max(h, rad)
    h = ti.min(h, 0.0)

    v = uv[1] * ba[0] - uv[0] * ba[1]
    v = sign(v)

    dist = (uv - ba * h).norm() * v  # diatance field of 2D-stars
    return dist


@ti.func
def sdf_sphere(uv, rad):
    return uv.norm() - rad


@ti.func
def sdf_box(pos, size):
    d = ti.abs(pos)-size
    dist = ti.max(d, 0.0).norm()
    dist = dist + ti.min(ti.max(d.x, d.y), 0.0)
    return dist


@ti.func
def sdf_arc(pos, sc, ra, rb):
    # pos.x = ti.abs(pos.x)
    dist = 0.0
    if sc.y * pos.x > sc.x * pos.y:
        dist = (pos - sc * ra).norm()
    else:
        dist = ti.abs(pos.norm() - ra) - rb
    return dist


@ti.func
def sdf_line(pos, ang):
    n = ti.floor(ang / (2.0 * math.pi))
    ang = ang - 2.0 * math.pi * n
    k = ti.tan(ang)
    dist = (k * pos[0] - pos[1]) / ti.sqrt(k * k + 1)
    if ang > 0.5 * math.pi and ang < 1.5 * math.pi:
        dist = - dist
    return dist


@ti.func
def sdf_taichi(pos, radius, ang, padding=0.01, inner_radius=0.2):
    sub_offset = ti.Vector([ti.cos(ang), ti.sin(ang)]) * radius * 0.5
    d = sdf_sphere(pos, radius)
    d1 = sdf_sphere(pos - sub_offset, radius * 0.5)
    d0 = sdf_sphere(pos + sub_offset, radius * 0.5)
    dl = sdf_line(pos, ang)

    dist0 = max(min(max(d, dl - padding), d0), -d1) + padding
    dist1 = max(min(max(d, -dl - padding), d1), -d0) + padding
    dist = min(dist0, dist1)
    dist = max(dist, -(d0 + inner_radius))
    dist = max(dist, -(d1 + inner_radius))
    return dist


@ti.func
def blink_col():
    pass


inner_col = ti.Vector([0.25, 0.85, 1.0])
outer_col = ti.Vector([0.65, 0.35, 0.76])


@ti.func
def render_grad(dist):
    # ref: https://github.com/ElonKou/biulab/blob/master/shaders/shadertoy/sdf2d1.fs
    col = ti.Vector([0.0, 0.0, 0.0])
    if (dist < 0.0):
        col = inner_col
    else:
        col = outer_col
    col = col * (1.0 - ti.exp(-6.0 * ti.abs(dist)))
    col = col * (0.8 + 0.2 * ti.cos(dist * 350))
    c = smoothstep(0.0, 0.01, ti.abs(dist))
    col = mix(col, ti.Vector([1.0, 1.0, 1.0]), 1.0 - c)

    ret = ti.Vector([col[0], col[1], col[2], 1.0])
    return ret


@ti.func
def render_blink(dist, t):
    col = ti.Vector([0.0, 0.0, 0.0, 0.0])
    if dist < 0.0:
        col = ti.Vector([0.7 + ti.sin(t * 8.03) * 0.3, 0.9 * ti.sin(dist * 2 + 1.0), 0.7, 1.0])
    return col
