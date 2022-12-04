# !/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022.12.04
# @Author  : elonkou
# @File    : main.py
# Contains all Scenes / Movie for render.

from ticore import *
import _thread


@ti.data_oriented
class Scene():
    def __init__(self) -> None:
        self.scene_name = "default"
        self.dura = 15.0              # [seconds]
        self.st = 0.0                 # start time
        self.ed = 0.0                 # end time
        self.multi_samples = True     # if use multi-sample, render will sample [2x2] points per pixel.
        self.draw_l = 0
        self.draw_r = res[0]
        self.draw_b = 0
        self.draw_t = res[1]

    def LoadScene(self):
        print("Load Scene")

    def Render(self):
        self.RenderCore(t - self.st, self.draw_l, self.draw_r, self.draw_b, self.draw_t)

    @ti.kernel
    def RenderCore(self, t: float, left: int, right: int, bot: int, top: int):
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
                        col = self.GetColor(uv + off, I, dist, t)
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
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = clear_color


@ti.data_oriented
class CircleScene(Scene):
    def __init__(self) -> None:
        super(CircleScene, self).__init__()
        self.scene_name = "CircleScene"

    @ti.func
    def GetSDF(self, uv, I, t):
        if t > 5.0 and t < 15.0:
            cnt = perid_time(t-5.0, 4, 2)
            if cnt > 1.0:
                uv = fract(uv * cnt) - ti.Vector([0.5, 0.5])
        dist = sdf_sphere(uv, 0.5) + ti.sin(t * 0.5) * 0.1
        return dist

    @ti.func
    def GetColor(self, uv, I, dist, t):
        col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if dist < 0.0:
            col = render_scale(dist, t * 3.0)
        return col


@ ti.data_oriented
class TaiChiScene(Scene):
    def __init__(self) -> None:
        super(TaiChiScene, self).__init__()
        self.scene_name = "TaiChiScene"
        self.multi_samples = False
        self.dura = 17.0

    @ ti.func
    def GetSDF(self, uv, I, t):
        rotate_speed = 12.2
        ang = t + ti.sin(t) * 3 + 1.5
        radius = 0.4 + ti.sin(t * rotate_speed) * 0.1 + 0.1
        padding = 0.02 + ti.sin(t * 2.3) * 0.02 + 0.02
        inner_radius = 0.16 + ti.sin(t * 2.32) * 0.08 + 0.08

        step = int(ti.floor(t / 3.0))
        step = step % 6
        dist = 0.0
        dist_taichi = sdf_taichi(uv, radius, ang, padding, inner_radius, step)
        dist_sphere = sdf_sphere(uv, radius)
        if t < 12.0:
            dist = dist_taichi
        elif t >= 12.0 and t < 15.0:
            dist = mix(dist_taichi, dist_sphere, (t - 12.0) * (1.0 / 3.0))
        else:
            dist = dist_sphere
        return dist

    @ ti.func
    def GetColor(self, uv, I, dist, t):
        col = render_scale(dist, t)
        return col

    @ ti.func
    def SetColor(self, uv, I, dist, t, col):
        pass

    @ti.kernel
    def Clear(self):
        for I in ti.grouped(ti.ndrange(res[0], res[1])):
            pixels[I] = clear_color


@ ti.data_oriented
class LineScene(Scene):
    def __init__(self) -> None:
        super(LineScene, self).__init__()
        self.scene_name = "LineScene"
        self.dura = 21.0

    @ti.func
    def GetSDF(self, uv, I, t):
        dist = 10.0
        dist_sphere = sdf_sphere(uv, 0.5)
        dist_star = sdf_star(uv, 0.6, 0.5)

        if t < 12.0:
            cnt = int(perid_time(t, 5, 2) + 1)
            for l in range(cnt):
                d = ti.abs(sdf_line(uv, t + math.pi * 1.0 / cnt * l)) - 0.05
                dist = ti.min(dist, d)
                if l >= 3:
                    d = sdf_sphere(uv, 0.5 + ti.sin(t * 4.5) * 0.1 + 0.1)
                    dist = ti.max(dist, d)
                if l >= 4:
                    d = sdf_sphere(uv, 0.5 - ti.sin(t * 4.5) * 0.2 + 0.2)
                    dist = ti.min(dist, d)
            if t < 2.0:
                dist = mix(dist, dist_sphere, convert_0to1_smooth(t - 2.0, 2.0))
        elif t < 15.0:
            dist = mix(dist_sphere, dist_star, (t-12.0) * (1.0 / 3.0))
        elif t < 18.0:
            dist = ti.max(dist_star, -(dist_sphere + 0.3 + ti. sin(t) * 0.1))
        else:
            dist = dist_star

        return dist

    @ti.func
    def GetColor(self, uv, I, dist, t):
        col = render_scale(dist, t)
        return col


@ ti.data_oriented
class StarScene(Scene):
    def __init__(self) -> None:
        super(StarScene, self).__init__()
        self.rad = 0.5
        self.rf = 0.6
        self.multi_samples = False
        self.dura = 20
        self.star_init()

    def Render(self):
        self.RenderCore(t - self.st, self.draw_l, self.draw_r, self.draw_b, self.draw_t)
        vel[None].z = -1000 * (1.2 + math.cos(t * math.pi))
        self.step()
        self.paint(t)

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
        dist_star = sdf_star(uv, self.rf, self.rad)
        dist_sphere = sdf_sphere(uv, self.rad)
        dist = 0.0
        if t < 5.0:
            dist = dist_star
        elif t >= 5.0 and t < 13.0:
            dist = mix(dist_star, dist_sphere, ti.sin((t - 5.0) * math.pi * 0.5) * 0.5 + 0.5)
        else:
            dist = dist_star
        return dist

    @ ti.func
    def GetColor(self, uv, I, dist, t):
        col = render_scale(dist, t)
        return col

    @ ti.func
    def draw_star(self, c_pos, radius, t, col):
        left = ti.max(int(c_pos.x - radius - 1.0), 0)
        right = ti.min(int(c_pos.x + radius + 1.0), res[0])
        bot = ti.max(int(c_pos.y - radius - 1.0), 0)
        top = ti.min(int(c_pos.y + radius + 1.0), res[1])
        self.RenderCoreStar(t - self.st, left, right, bot, top, col[0], col[1], col[2], col[3])

    @ ti.kernel
    def paint(self, t: float):
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
    def step(self):
        for I in ti.grouped(pos):
            pos[I] += vel[None] * dt
            if pos[I].z < z_near:
                pos[I].z += z_far - z_near
                pos[I].x = vel[None].z * ti.cos(pos[I].x)

    @ti.kernel
    def star_init(self):
        for I in ti.grouped(pos):
            pos[I] = (I + rand3()) * grid_size
            pos[I].z += z_near
            color[I] = rand3() + ti.Vector([1.0, 2.0, 3.0])

    @ti.func
    def project(self, pos_3d):
        center = ti.Vector(res) / 2
        w = tan_half_fov * pos_3d.z
        res_pos = (ti.Vector([pos_3d.x, pos_3d.y]) - center) / w
        screen_pos = res_pos * res[1] + center
        return screen_pos


@ ti.data_oriented
class TreeScene(Scene):
    def __init__(self) -> None:
        super(TreeScene, self).__init__()
        self.scene_name = "TreeScene"

    @ ti.func
    def GetSDF(self, uv, I, t):
        dist_star = sdf_star(uv, 0.6, 0.5)
        rad = 0.5
        ang = -60 / 180.0 * math.pi
        p0 = ti.Vector([0.0, rad])
        p1 = ti.Vector([rad * ti.cos(ang), rad * ti.sin(ang)])
        p2 = ti.Vector([-rad * ti.cos(ang), rad * ti.sin(ang)])
        dist_tri1 = sdf_tri(uv, p0, p1, p2)
        p0 = ti.Vector([0.0, 0.45])
        p1 = ti.Vector([0.18, 0.18])
        p2 = ti.Vector([-0.18, 0.18])
        dist_tri2 = sdf_tri(uv, p0, p1, p2)
        dist_tree = sdf_tree(uv)

        dist = 0.0
        if t < 1.0:
            dist = dist_star
        elif t >= 1.0 and t < 5.0:
            dist = mix(dist_star, dist_tri1, convert_0to1_smooth(t - 1.0, 4.0))
        elif t >= 5.0 and t < 7.0:
            dist = dist_tri1
        elif t >= 7.0 and t < 12.0:
            dist = mix(dist_tri1, dist_tree, convert_0to1_smooth(t - 7.0, 5.0))
        else:
            dist = dist_tree
        return dist

    @ ti.func
    def GetColor(self, uv, I, dist, t):
        col = render_scale(dist, t)
        return col


@ti.data_oriented
class FulidScene(Scene):
    def __init__(self) -> None:
        super(FulidScene, self).__init__()
        self.dura = 30

    @ti.func
    def GetSDF(self, uv, I, t):
        dist_tree = sdf_tree(uv)
        dist_sphere1 = sdf_sphere(uv - ti.Vector([0.0, 0.1]), 0.4)
        dist_sphere2 = sdf_sphere(uv + ti.Vector([0.0, 0.1]), 0.2)
        dist_egg = sdf_egg(uv + ti.Vector([0.0, 0.25]), 0.20, 0.09)
        dist_heart = sdf_heart(uv * 2.0 + ti.Vector([0.0, 0.5]))

        dist = 0.0
        combine_time = 3.0

        # NOTE : I am not familiar with taichi variable, is list or vector like "x = [...]"?
        if t >= 0.0 and t < 2.0:
            dist = dist_tree
        elif t >= 2.0 and t < 5.0:
            dist = mix(dist_tree, dist_sphere1, convert_0to1_smooth(t - 2.0, combine_time))
        elif t >= 5.0 and t < 7.0:
            dist = dist_sphere1
        elif t >= 7.0 and t < 10.0:
            dist = mix(dist_sphere1, dist_egg, convert_0to1_smooth(t - 7.0, combine_time))
        elif t >= 10.0 and t < 12.0:
            dist = dist_egg
        elif t >= 12.0 and t < 15.0:
            dist = mix(dist_egg, dist_heart, convert_0to1_smooth(t - 12.0, combine_time))
        elif t >= 15.0 and t < 17.0:
            dist = dist_heart
        elif t >= 17.0 and t < 20.0:
            dist = mix(dist_heart, dist_sphere2, convert_0to1_smooth(t - 17.0, combine_time))
        elif t >= 20.0 and t < 22.0:
            dist = dist_sphere2
        elif t >= 22.0 and t < 25.0:
            dist = mix(dist_sphere2, dist_tree, convert_0to1_smooth(t - 22.0, combine_time))
        else:
            # cnt = tooth((t - 25.0) * 0.2) * 4.0 + 1.0
            # uv = fract(uv * cnt) - ti.Vector([0.5, 0.5])
            # dist = sdf_tree(uv)
            dist = dist_tree

        return dist

    @ti.func
    def GetColor(self, uv, I, dist, t):
        col = render_scale(dist, t)
        return col


@ti.data_oriented
class MultiCircleScene(Scene):
    def __init__(self) -> None:
        super(MultiCircleScene, self).__init__()
        self.dura = 25.0

    @ti.func
    def GetSDF(self, uv, I, t):
        dist = 100.0
        dis_circle = 100.0

        dis_tree = sdf_tree(uv)
        for i in range(10):
            rad = fract(i * 0.1 + t * 0.12)
            d = sdf_multi_circle(uv, rad)
            dis_circle = ti.min(dis_circle, d)
        dist_flower = sdf_flower(uv, t)
        dis_sph = sdf_sphere(uv, 0.5)

        if t < 2.0:
            dist = dis_tree
        elif t >= 2.0 and t < 5.0:
            dist = mix(dis_tree, dis_circle, convert_0to1_smooth(t - 2.0, 3.0))
        elif t >= 5.0 and t < 10.0:
            dist = dis_circle
        elif t >= 10.0 and t < 15.0:
            dist = mix(dis_circle, dist_flower, convert_0to1_smooth(t - 10.0, 5.0))
        elif t >= 15.0 and t < 20.0:
            dist = dist_flower
        elif t >= 20.0 and t < 25.0:
            dist = mix(dist_flower, dis_sph, convert_0to1_smooth(t - 20.0, 5.0))
        else:
            dist = dis_sph
        return dist

    @ti.func
    def GetColor(self, uv, I, dist, t):
        col2 = render_scale(dist, t)
        col3 = render_black(dist, uv)
        col = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if t < 2.0:
            col = col2
        elif t >= 2.0 and t < 5.0:
            col = mix(col2, col3, convert_0to1_smooth(t - 2.0, 3.0))
        elif t >= 5.0 and t < 10.0:
            col = col3
        elif t >= 10.0 and t < 15.0:
            col = mix(col3, col2, convert_0to1_smooth(t - 10.0, 5.0))
        elif t >= 15.0:
            col = col2
        return col


@ti.data_oriented
class AnimalScene(Scene):
    def __init__(self) -> None:
        super(AnimalScene, self).__init__()

    @ti.func
    def GetSDF(self, uv, I, t):
        dist_sph = sdf_sphere(uv, 0.5)
        dist_animal = sdf_animal(uv)
        dist = 0.0
        if t < 3.0:
            dist = dist_sph
        elif t >= 3.0 and t < 11.0:
            dist = mix(dist_sph, dist_animal, convert_whole_smooth(t - 3.0, 8.0 / 3 * 2))
        else:
            dist = dist_animal
        return dist

    @ti.func
    def GetColor(self, uv, I, dist, t):
        col = render_scale(dist, t)
        return col


@ ti.data_oriented
class Movie():
    def __init__(self) -> None:
        # add all frames
        self.frames = []
        self.frames.append(CircleScene())
        self.frames.append(TaiChiScene())
        self.frames.append(LineScene())
        self.frames.append(StarScene())
        self.frames.append(TreeScene())
        self.frames.append(FulidScene())
        self.frames.append(MultiCircleScene())
        self.frames.append(AnimalScene())

        self.sum_t = 0.0  # modified frame dura.
        self.play_music = False
        self.play_music = True

        # load music
        self.music_files = [
            "./music/" + "dubstep-drum-solo-140bpm-by-prettysleepy-art-15454.mp3",
            "./music/" + "Dynamic-good-electronic-music.mp3",
            "./music/" + "electronic-drum-loop-by-prettysleepy-art-12918.mp3"
        ]
        self.music = []
        for mf in self.music_files:
            song = AudioSegment.from_mp3(mf)
            self.music.append(song)

        # modified start and end time
        for frame in self.frames:
            frame.st = frame.st + self.sum_t
            frame.ed = frame.st + frame.dura
            self.sum_t = self.sum_t + frame.dura
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
        print("Start movie")
        cur_frame = 0
        frame = self.frames[cur_frame]
        print("Current frame : ", frame.scene_name, frame.st, frame.ed)

        if self.play_music:
            _thread.start_new_thread(PlayMusic, ("Music-1", self.music[1][:173000]))

        while self.gui.running:
            if self.gui.get_event(ti.GUI.ESCAPE):
                self.gui.running = False
            t = t + dt
            # print(t)
            if t >= frame.ed and cur_frame < (len(mov.frames)-1):
                cur_frame = cur_frame + 1
                frame = self.frames[cur_frame]
                print("Current frame : ", frame.scene_name, frame.st, frame.ed)
            frame.Render()
            mov.Copy()
            self.gui.set_image(pixels_dis)
            self.gui.show()
            frame.Clear()


if __name__ == "__main__":
    mov = Movie()
    mov.Init()
    mov.Play()
