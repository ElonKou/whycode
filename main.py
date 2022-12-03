# !/bin/env python3
# -*- coding: utf-8 -*-

from ticore import *


@ti.data_oriented
class Scene():
    def __init__(self) -> None:
        self.scene_name = "default"
        self.dura = 5.0
        self.st = 0.0
        self.ed = self.st + self.dura
        self.multi_samples = True  # [2x2] samples per pixel
        self.draw_l = 0
        self.draw_r = res[0]
        self.draw_b = 0
        self.draw_t = res[1]

    def LoadScene(self):
        print("Load Scene")

    def Render(self):
        # print(self.scene_name)
        self.RenderCore(t - self.st, self.draw_l, self.draw_r, self.draw_b, self.draw_t)

    @ti.kernel
    def RenderCore(self, t: float, left: int, right: int, bot: int, top: int):
        # for I in ti.grouped(ti.ndrange(res[0], res[1])):
        for I in ti.grouped(ti.ndrange((left, right), (bot, top))):
            width = right - left
            height = top - bot
            # uv = ti.Vector([(I[0]*2.0 - res[0]) / res[0], (I[1]*2.0 - res[1]) / res[1]])  # [-1.0, -1.0]    -> [1.0, 1.0]
            # uv = ti.Vector([(I[0]*2.0 - res[0]) / res[1], (I[1]*2.0 - res[1]) / res[1]])  # [-some, -1.0]   -> [some, 1.0]
            uv = ti.Vector([(I[0] - width/2.0) / height, (I[1] - height/2.0) / height])      # [-some/2, -0.5] -> [some/2, 0.5]

            samples = 2

            if self.multi_samples:
                total_col = ti.Vector([0.0, 0.0, 0.0, 0.0])
                offset = ti.Vector([(2.0 * width / height) / height, 2.0 / height]) / 2
                # offset = ti.Vector([(1.0 * res[0] / res[1]) / res[1], 1.0 / res[1]]) / 2
                for i in range(samples):  # heigth
                    for j in range(samples):  # width
                        off = ti.Vector([offset[0] * j,  offset[1] * i])
                        dist = self.GetSDF(uv + off, I, t)
                        col = self.GetColor(uv+off, I, dist, t)
                        if col.norm() > 0.0:
                            col = combine(pixels[I], col)
                        total_col = total_col + col
                total_col = total_col / samples / samples
                if total_col.norm() > 0.0:
                    pixels[I] = total_col
            else:
                dist = self.GetSDF(uv, I, t)
                col = self.GetColor(uv, I, dist, t)
                if col.norm() > 0.0:
                    pixels[I] = combine(pixels[I], col)

    @ ti.func
    def GetSDF(self, uv, I, t):
        dist = sdf_sphere(uv, 0.5)
        return dist

    @ ti.func
    def GetColor(self, uv, I, dist, t):
        if dist < 0.0:
            col = ti.Vector([0.06, 0.23, 0.85, 1.0])
        else:
            col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        return col

    @ ti.func
    def SetColor(self, uv, I, dist, t, col):
        pass

    @ ti.kernel
    def Clear(self):
        pass


@ ti.data_oriented
class FirstScene(Scene):
    def __init__(self) -> None:
        super(FirstScene, self).__init__()
        self.scene_name = "First"
        self.multi_samples = False

    @ ti.func
    def GetSDF(self, uv, I, t):
        uv = fract(uv * 2.0) - ti.Vector([0.5, 0.5])
        dist = sdf_sphere(uv, 0.5) + ti.sin(t) * 0.1
        # dist = sdf_sphere(uv, 0.5)

        # pos = ti.Vector([0.0, 0.0])
        # size = ti.Vector([0.1, 0.2])
        # dist = sdf_box(uv + pos, size)

        # tb = 3.14 * 0.9
        # sc = ti.Vector([ti.sin(tb), ti.cos(tb)])
        # dist = sdf_arc(uv, sc, 0.4, 0.05)

        # ang = t + ti.sin(t) * 3 + 1.5
        # radius = 0.4 + ti.sin(t * 2.1) * 0.1 + 0.1
        # padding = 0.01 + ti.sin(t * 2.3) * 0.01 + 0.01
        # inner_radius = 0.16 + ti.sin(t * 1.32) * 0.08 + 0.08
        # dist = sdf_taichi(uv, radius, ang, padding, inner_radius)
        return dist

    @ ti.func
    def GetColor(self, uv, I, dist, t):
        # col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        # if dist < 0.0:
        #     col = ti.Vector([0.78, 0.45, 0.46, 1.0])

        dist = dist * ti.sin(t) * 1.45
        col = render_grad(dist)
        col.x = mix(col.x, 1.0, ti.sin(t * 2.3) * 0.9)
        col.z = mix(col.z, 1.0, ti.cos(t * 3.1) * 0.9)
        col.y = mix(col.y, 0.0, ti.sin(t * 0.67) * 0.3)

        # col = render_blink(dist, t)
        # col = render_grad(dist)
        return col

    @ ti.func
    def SetColor(self, uv, I, dist, t, col):
        pass

    @ti.kernel
    def Clear(self):
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = clear_color


@ ti.data_oriented
class SecondScene(Scene):
    def __init__(self) -> None:
        super(SecondScene, self).__init__()
        self.scene_name = "Second"

    def Render(self):
        self.RenderCore(t - self.st, self.draw_l, self.draw_r, self.draw_b, self.draw_t)

    @ ti.kernel
    def RenderCore(self, t: float, left: int, right: int, bot: int, top: int):
        col = ti.Vector([ti.sin(t), ti.cos(t), ti.sin(t), 0.0]) * 0.5 + ti.Vector([0.5, 0.5, 0.5, 1.0])
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = col


# @ti.data_oriented
# class ThirdScene(Scene):
#     def __init__(self) -> None:
#         super(ThirdScene, self).__init__()
#         self.scene_name = "ThirdScene"

#     def Render(self):
#         self.RenderCore(t, self.draw_l, self.draw_r, self.draw_b, self.draw_t)

#     @ti.kernel
#     def RenderCore(self, t: float, left: int, right: int, bot: int, top: int):
#         self.DoSome(t)

#     @ti.func
#     def DoSome(t):
#         if t > 5.0:
#             t = t - 5.0
#             print(t)


@ ti.data_oriented
class BGScene(Scene):
    def __init__(self) -> None:
        super(BGScene, self).__init__()
        self.scene_name = "BGScene"
        self.border = -0.3

    @ ti.func
    def GetSDF(self, uv, I, t):
        dist = 0.0
        if uv[1] < self.border:
            dist = -0.1
        return dist

    @ ti.func
    def GetColor(self, uv, I, dist, t):
        col = ti.Vector(0.0, 0.0, 0.0, 0.0)
        if dist < 0.0:
            col = ti.Vector([0.23, 0.34, 0.12, 0.8])
        return col


@ ti.data_oriented
class StarScene(Scene):
    def __init__(self) -> None:
        super(StarScene, self).__init__()
        self.rad = 0.5
        self.rf = 0.6
        self.multi_samples = False
        self.star_init()

    def Render(self):
        vel[None].z = -1000 * (1.2 + math.cos(t * math.pi))
        self.step()
        self.paint(t)
        # self.RenderCore(t - self.st, self.draw_l, self.draw_r, self.draw_b, self.draw_t)

    @ti.func
    def RenderCoreStar(self, t, left, right, bot, top, colr, colg, colb, cola):
        for I in ti.grouped(ti.ndrange((left, right), (bot, top))):
            width = right - left
            height = top - bot
            pos = I - ti.Vector([left, bot]) - ti.Vector([width, height]) * 0.5
            uv = pos / height
            samples = 2
            if self.multi_samples:
                total_col = ti.Vector([0.0, 0.0, 0.0, 0.0])
                offset = ti.Vector([(2.0 * width / height) / height, 2.0 / height]) / 2
                for i in range(samples):  # heigth
                    for j in range(samples):  # width
                        off = ti.Vector([offset[0] * j,  offset[1] * i])
                        dist = self.GetSDF(uv + off, I, t)
                        col = ti.Vector([colr, colg, colb, cola])
                        if dist > 0.0:
                            col = ti.Vector([0.0, 0.0, 0.0, 0.0])
                        if col.norm() > 0.0:
                            col = combine(pixels[I], col)
                        total_col = total_col + col
                total_col = total_col / samples / samples
                if total_col.norm() > 0.0:
                    pixels[I] = total_col
            else:
                dist = self.GetSDF(uv, I, t)
                col = ti.Vector([colr, colg, colb, cola])
                if dist > 0.0:
                    col = ti.Vector([0.0, 0.0, 0.0, 0.0])
                if col.norm() > 0.0:
                    pixels[I] = combine(pixels[I], col)

    @ ti.func
    def GetSDF(self, uv, I, t):
        dist = sdf_star(uv, self.rf, self.rad)
        return dist

    # @ ti.func
    # def GetColor(self, uv, I, dist, t, col):
    #     col = ti.Vector([0.0, 0.0, 0.0, 0.0])
    #     if dist < 0.0:
    #         col = ti.Vector([0.7 + ti.sin(t * 8.03) * 0.3, 0.9 * ti.sin(dist * 2 + 1.0), 0.7, 1.0])
    #     return col

    @ ti.func
    def draw_star(self, c_pos, radius, t, col):
        left = max(int(c_pos.x - radius - 1.0), 0)
        right = min(int(c_pos.x + radius + 1.0), res[0])
        bot = max(int(c_pos.y - radius - 1.0), 0)
        top = min(int(c_pos.y + radius + 1.0), res[1])
        self.RenderCoreStar(t - self.st, left, right, bot, top, col[0], col[1], col[2], col[3])

    @ ti.kernel
    def paint(self, t: float):
        # self.draw_star(ti.Vector([res[0]//2, res[1]//2]), 100.0, t)

        for I in ti.grouped(pos):
            rad = 10.0 * (1.0 - pos[I].z / z_far)**2
            cur_p = pos[I]
            p = self.project(cur_p)

            col = color[I] * (1.0 - pos[I].z / z_far + 0.1) ** 0.5
            dist = ((pos[I].x / res[0] - 0.5)**2 + (pos[I].y / res[1] - 0.5)**2)**0.2  # distance from pos to center.
            col = col * (1.0 - dist)
            col_ret = ti.Vector([col[0], col[1], col[2], 1.0])

            self.draw_star(p, rad, t, col_ret)

    @ ti.kernel
    def Clear(self):
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = ti.Vector([0.83, 0.83, 0.83, 1.0]) * pixels[I]

    @ti.kernel
    def step():
        for I in ti.grouped(pos):
            pos[I] += vel[None] * dt
            if pos[I].z < z_near:
                pos[I].z += z_far - z_near
                pos[I].x = vel[None].z * ti.cos(pos[I].x)

    @ti.kernel
    def star_init():
        for I in ti.grouped(pos):
            pos[I] = (I + rand3()) * grid_size
            pos[I].z += z_near
            color[I] = rand3() + ti.Vector([1.0, 2.0, 3.0])

    @ti.func
    def project(pos_3d):
        center = ti.Vector(res) / 2
        w = tan_half_fov * pos_3d.z
        res_pos = (ti.Vector([pos_3d.x, pos_3d.y]) - center) / w
        screen_pos = res_pos * res[1] + center
        return screen_pos

    @ ti.kernel
    def Clear(self):
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = clear_color


@ ti.data_oriented
class Movie():
    def __init__(self) -> None:
        self.frames = []
        self.frames.append(FirstScene())
        # self.frames.append(SecondScene())
        # self.frames.append(ThirdScene())
        # self.frames.append(StarScene())
        # self.bg_frame = BGScene()
        sum_t = 0.0  # modified frame dura.
        for frame in self.frames:
            frame.st = frame.st + sum_t
            frame.ed = frame.ed + sum_t
            sum_t = sum_t + frame.dura
        self.gui = ti.GUI("why coding?", res, fast_gui=True)

    @ ti.kernel
    def Init(self):
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = clear_color

    @ ti.kernel
    def Copy(self):
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            col = pixels[I]
            pixels_dis[I] = ti.Vector([col[0], col[1], col[2]]) * col[3]

    def Play(self):
        global t
        print("Start")
        cur_frame = 0
        frame = self.frames[cur_frame]
        print(frame.scene_name, frame.st, frame.ed)
        while self.gui.running:
            if self.gui.get_event(ti.GUI.ESCAPE):
                self.gui.running = False
            t = t + dt
            if t >= frame.ed and cur_frame < (len(mov.frames)-1):
                cur_frame = cur_frame + 1
                frame = self.frames[cur_frame]
                print(frame.scene_name, frame.st, frame.ed)
            frame.Render()
            # self.bg_frame.Render()
            mov.Copy()
            self.gui.set_image(pixels_dis)
            self.gui.show()
            frame.Clear()


if __name__ == "__main__":
    mov = Movie()
    mov.Init()
    mov.Play()
