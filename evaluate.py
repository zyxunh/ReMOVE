import os
import time
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from unhcv.common import visual_mask, write_im
from unhcv.common.image import putText, concat_differ_size
from unhcv.common.utils import find_path, attach_home_root, ProgressBarTqdm, dict2strs, obj_dump, obj_load
from unhcv.projects.diffusion.inpainting.evaluation.evaluation_model import init_inpainting_eval_dataset

from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor
from crop import find_smallest_bounding_square, draw_bb
import argparse

import warnings

warnings.filterwarnings("ignore")

class Metric:
    def __init__(self, crop=1):
        assert crop in [0, 1]
        crop = crop == 1
        sam = sam_model_registry["vit_h"](checkpoint=find_path("model/sam/sam_vit_h_4b8939.pth")).cuda()
        self.predictor = SamPredictor(sam)
        self.crop = crop
        self.remove_score_sum = 0
        self.num = 0

    def metric(self, image, result, inpainting_mask):
        if self.crop:
            binary_image = np.array(inpainting_mask)
            # if args.draw:
            #     draw_bb(args.mask_path)
            x, y, size = find_smallest_bounding_square(binary_image)

            input_img = np.array(result.convert("RGB"))[y:y + size, x:x + size]

            mask_fg = np.array(
                Image.fromarray(np.array(inpainting_mask.convert("L"))[y:y + size, x:x + size]).resize(
                    (64, 64))).reshape((1, 1, 64, 64)) // 255
            mask_bg = 1 - mask_fg

        else:
            input_img = np.array(result.convert("RGB"))

            mask_fg = np.array(inpainting_mask.resize((64, 64))).reshape((1, 1, 64, 64)) // 255
            mask_bg = 1 - mask_fg

        embeddings = self.predictor.get_aggregate_features(input_img, [mask_fg, mask_bg])

        remove_score = cosine_similarity(embeddings[0], embeddings[1]).item()
        if np.isnan(remove_score):
            remove_score = 0
            print('nan')
        self.remove_score_sum += remove_score
        self.num += 1

        return dict(remove_score=remove_score)

    def summary(self):
        return dict(mean_remove_score=self.remove_score_sum / self.num * 100)

def get_score(predictor, args):
    if args.crop:
        binary_image = np.array(Image.open(args.mask_path))
        if args.draw:
            draw_bb(args.mask_path)
        x, y, size = find_smallest_bounding_square(binary_image)

        input_img = np.array(Image.open(args.image_path).convert("RGB"))[y:y + size, x:x + size]

        mask_fg = np.array(
            Image.fromarray(np.array(Image.open(args.mask_path).convert("L"))[y:y + size, x:x + size]).resize(
                (64, 64))).reshape((1, 1, 64, 64)) // 255
        mask_bg = 1 - mask_fg

    else:
        input_img = np.array(Image.open(args.image_path).convert("RGB"))

        mask_fg = np.array(Image.open(args.mask_path).resize((64, 64))).reshape((1, 1, 64, 64)) // 255
        mask_bg = 1 - mask_fg

    embeddings = predictor.get_aggregate_features(input_img, [mask_fg, mask_bg])

    remove_score = cosine_similarity(embeddings[0], embeddings[1]).item()

    return remove_score




def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str)
    parser.add_argument(
        '--show_path', type=str)
    parser.add_argument(
        '--data_indexes_path', type=str, default=None)
    parser.add_argument('--inter_ratio_thres', type=float, default=0.8)
    parser.add_argument(
        '--show', action="store_true")
    parser.add_argument(
        '--crop', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_parser()
    model_result_path = attach_home_root(args.result_path)
    show_root = attach_home_root(args.show_path)
    # remove_dir(show_root)
    from unhcv.projects.diffusion.inpainting.evaluation.dataset import InpaintEvalutionDatasetRead
    # removing_metric = RemovingMetric(inter_ratio_thres=args.inter_ratio_thres)
    clip_metric = Metric(crop=args.crop)

    dataset = init_inpainting_eval_dataset(data_indexes_path=args.data_indexes_path, preprocess=False)
    # dataset = InpaintEvalutionDatasetRead()
    dataset_num = len(dataset)
    progress_bar = ProgressBarTqdm(dataset_num)
    for i_data, data in enumerate(dataset):
        if i_data != 4:
            pass
        model_result = obj_load(os.path.join(model_result_path, f"{i_data}.jpg"))
        model_result = model_result.resize(data['inpainting_mask'].size, resample=Image.BICUBIC)
        # data['inpainting_mask'] = data['inpainting_mask'].resize(model_result.size, resample=Image.NEAREST)
        metric_dict = clip_metric.metric(data['image'], model_result, data['inpainting_mask'])
        obj_dump(os.path.join(show_root, f"{i_data}.yml"), metric_dict)
        progress_bar.update()
        if not args.show:
            continue

        # visual
        model_result_np = np.array(model_result)[..., ::-1]
        image = np.array(data['image'].convert('RGB'))[..., ::-1]
        inpainting_mask_show = visual_mask(image, np.array(data['inpainting_mask']))[-1]
        show_text = putText(None, show_texts=dict2strs(metric_dict),
                            img_size=(image.shape[1], image.shape[0]))
        shows = [model_result_np, inpainting_mask_show, show_text]
        shows = concat_differ_size(shows, axis=1)

        write_im(os.path.join(show_root, f"{i_data}.jpg"), shows)

    metric_inform = clip_metric.summary()
    # print(f'bad_img_num: {bad_img_num}, bad_case_num: {bad_case_num / dataset_num}, bad_case_inter_ratio_to_inpainting: {bad_case_inter_ratio_to_inpainting / dataset_num}')
    print(metric_inform)
    obj_dump(os.path.join(show_root, 'all_metric', "all_metric.yml"), metric_inform)