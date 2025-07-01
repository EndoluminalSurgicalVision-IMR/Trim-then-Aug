from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2

from .lowcam_mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        fx = 816.8598
        fy =  814.8223
        cx =  308.2864
        cy =  158.3971
        skew =  0.2072

        self.K = np.array([[fx / 320, skew / 320, cx / 320, 0],
                           [0, fy / 256 , cy / 256, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        # print('side is ', side)
        f_str = "{:01d}{}".format(frame_index, self.img_ext)
        # image_path = os.path.join(
        #     self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        if side == 'l':
            image_path = os.path.join(self.data_path, folder, 'Frames', f_str)
        elif side == 'r':
            image_path = os.path.join(self.data_path, folder,  'Frames', f_str)

        # if side == 'l':
        #     image_path = os.path.join(self.data_path, f_str)
        # elif side == 'r':
        #     image_path = os.path.join(self.data_path, f_str)




        # print('image path is ', image_path)


        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


