# whycode
`whycoding?`是2022太极黑客马拉松中`五彩斑斓的黑` 团队的作品，主要是使用并行编程语言太极来实现一些动态变换的视觉特效，并辅助音乐伴奏。项目主要是使用隐式表达的SDF(Signed Distance Field)描述场景，利用taichi高效的并行能力在GPU上执行多重采样计算，有符号的距离函数的返回值用于各种形状的变换处理。

[中文](README.zh-CN.md)， [English](README.md)

## 安装和使用
```bash
# 安装 ffplay [安装ffmpeg即可]
# ubuntu
sudo apt-get install ffmpeg
# mac
brew insatll ffmpeg

# 下载代码
git clone https://github.com/ElonKou/whycode

# 按住那个python库
cd whycode
pip install -r requirements.txt

# 运行
python3 main.py
```

## 效果展示图
![赛博飞鼠](images/img05.png)

动态效果
![齿轮](images/gif_images/1gear.gif)
![太极图案](images/gif_images/2taichi_rot.gif)
![星星](images/gif_images/3star.gif)
![多层星星](images/gif_images/4star_multi.gif)
![星星变树](images/gif_images/5star2tree.gif)
![树变波浪](images/gif_images/6tree2wave.gif)
![水滴变心](images/gif_images/7waterdrop2heart.gif)
![波浪变花朵](images/gif_images/8wave2flower.gif)
![圆圈变笋笋](images/gif_images/9circle2mascot.gif)

## 项目介绍🎎
**团队名**：五彩斑斓的黑
**项目名**：why coding?
**参赛方向**：应用方向
**项目类别**：互动艺术

## 项目细节
**项目简介**： 可以交互的、酷炫的程序动画短片，利用使用shader toy类似的方式渲染动画。
**期望效果**： 实现多个shader SDF场景的随着时间不断的切换的效果，并且有音乐作为伴奏。

## 技术方案：
**整体技术**：采用fragment shader渲染的方式实现
**模块划分**：
- **shader模块✔︎**：使用taichi实现片段着色器。
- **场景模块✔︎**：维护各种场景的信息，包含时间等，在不同的时间段调用不同shader。
- **基本组件模块✔︎**：使用SDF的几何形状表示物理。
- **特效组件模块✔︎**：实现不同的变换函数用于shader特效的控制。
- **场景转场特效组件✔︎**：实现场景的切换特效
- **音乐控制组件✔︎**：实现特效音乐

## 技术风险：
**风险一**：内容太多做不完（解决方案：使用参数化的设计场景）`（果然是风险，已经遇到这个风险😄）`
**风险二**：难度太高（解决方案：使用简单几何形状去表达复杂的物体）