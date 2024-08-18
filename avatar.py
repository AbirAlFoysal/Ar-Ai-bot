import os
import sys
import torch
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings

warnings.filterwarnings("ignore")

def process_video(source_image_path, driving_video_path, output_video_path, config_path, checkpoint_path, device, predict_mode='relative', find_best_frame=False):
    pixel = 256
    source_image = imageio.imread(source_image_path)
    source_image = resize(source_image, (pixel, pixel))[..., :3]

    sys.path.append(os.path.abspath('Thin-Plate-Spline-Motion-Model'))
    from demo import load_checkpoints, make_animation

    reader = imageio.get_reader(driving_video_path)
    fps = reader.get_meta_data()['fps']
    driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in reader]
    reader.close()

    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    
    if predict_mode == 'relative' and find_best_frame:
        from demo import find_best_frame as _find
        i = _find(source_image, driving_video, device.type == 'cpu')
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i + 1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)
        predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device=device, mode=predict_mode)

    # Save the video
    imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    return output_video_path

def main(source_image_path, driving_video_path='driving_ns.mp4'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_video_path = 'generated.mp4'
    config_path = 'Thin-Plate-Spline-Motion-Model/config/vox-256.yaml'
    checkpoint_path ='vox.pth.tar'
    predict_mode = 'relative'
    find_best_frame = False
    
    output_path = process_video(source_image_path, driving_video_path, output_video_path, config_path, checkpoint_path, device, predict_mode, find_best_frame)
    print(f"Output video saved at: {output_path}")

# main('submitted_image.png')
