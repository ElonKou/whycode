# whycode
`whycoding?` It is the work of the "Colorful Black" team in the TaiChi Hackathon 2022. It mainly uses the parallel programming language Taichi to achieve some dynamic transformation visual effects and assist in music accompaniment. The project mainly describes the scene with SDF (Signed Distance Field), which is implicitly expressed. It uses Taichi's efficient parallelism to perform multiple sampling calculations on the GPU. The return value of the SDF is used for transformation processing of various shapes.

[中文](README.zh-CN.md)， [English](README.md)


## Install and Use
```bash
# install ffplay [ffmpeg]
# ubuntu
sudo apt-get install ffmpeg
# mac
brew insatll ffmpeg

# clone code
git clone https://github.com/ElonKou/whycode

# install environment.
cd whycode
pip install -r requirements.txt

# run
python3 main.py
```

## Results
![](images/img05.png)

动态效果
![gear](images/gif_images/1gear.gif)
![taichi](images/gif_images/2taichi_rot.gif)
![star](images/gif_images/3star.gif)
![multi-stars](images/gif_images/4star_multi.gif)
![star2tree](images/gif_images/5star2tree.gif)
![tree2wave](images/gif_images/6tree2wave.gif)
![drop2geart](images/gif_images/7waterdrop2heart.gif)
![wave2folwer](images/gif_images/8wave2flower.gif)
![circle2mascot](images/gif_images/9circle2mascot.gif)

## Project introduction 🎎 
**Team name**: colorful black 
**project name**: why coding? 
**Entry direction**: application direction 
**Project category**: interactive art

## Project details 
**Project introduction**: Interactive and cool program animation clips can be used to render animation in a similar way using shader toy. 
**Expected effect**: to achieve the effect of continuous switching of multiple shader SDF scenes over time, with music as accompaniment.

## Technical solution: 
overall technology: module division is realized by fragment shader rendering: 
- **shader module ✔**: Use taichi to implement clip shaders
- **Scene module ✔**: Maintain the information of various scenarios, including time, and call different shaders in different time periods
- **Basic component module ✔**: The geometry of SDF is used to represent physics
- **Special effect component module ✔**: Realize different transformation functions for the control of shader special effects
- **Scene transition special effect components ✔**: Realizing scene switching special effects
- **music control component ✔**: Realize special effect music
