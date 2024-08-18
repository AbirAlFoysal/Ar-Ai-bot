import os
def setup_environment():
    if not os.path.exists('Thin-Plate-Spline-Motion-Model'):
        os.system('git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git')
    os.chdir('Thin-Plate-Spline-Motion-Model')

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
        os.system('pip3 install wldhx.yadisk-direct')
        os.system('curl -L $(yadisk-direct https://disk.yandex.com/d/i08z-kCuDGLuYA) -o checkpoints/vox.pth.tar')
        # Uncomment the following lines if you want to use other datasets
        # os.system('curl -L $(yadisk-direct https://disk.yandex.com/d/vk5dirE6KNvEXQ) -o checkpoints/taichi.pth.tar')
        # os.system('curl -L $(yadisk-direct https://disk.yandex.com/d/IVtro0k2MVHSvQ) -o checkpoints/mgif.pth.tar')
        # os.system('curl -L $(yadisk-direct https://disk.yandex.com/d/B3ipFzpmkB1HIA) -o checkpoints/ted.pth.tar')

    os.system('pip install imageio_ffmpeg')
# os.chdir('Thin-Plate-Spline-Motion-Model')

setup_environment()
