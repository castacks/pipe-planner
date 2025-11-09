#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


def calculate_iou_binary(predicted, ground_truth):
    # Threshold the predicted result to convert to binary mask
    predicted_binary = (predicted > 0.5).astype(np.uint8)
    
    # Ensure ground_truth is a binary mask, if not threshold similarly
    ground_truth_binary = (ground_truth > 0.5).astype(np.uint8) if ground_truth.max() > 1 else ground_truth
    
    # Calculate intersection and union
    intersection = np.logical_and(predicted_binary, ground_truth_binary).sum()
    union = np.logical_or(predicted_binary, ground_truth_binary).sum()
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0  # Avoid division by zero
    
    return intersection, union, iou


@hydra.main(config_path='../configs/prediction', config_name='map.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        # Set up different models 
        model_list = []
        for model_path in predict_config.model.paths:
            train_config_path = os.path.join(model_path, 'config.yaml')
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))
            
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            out_ext = predict_config.get('out_ext', '.png')

            checkpoint_path = os.path.join(model_path, 
                                        'models', 
                                        predict_config.model.checkpoint)
            model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
            model.freeze()
            if not predict_config.get('refine', False):
                model.to(device)
            model_list.append(model)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        
        # Key structure
        # metrics_dict_allmodels_allimages = {
        #     'model1': {
        #         'image1': {
        #             'iou_all': {'intersection': 0, 'union': 0, 'iou': 0},...}
        metrics_dict_allmodels_allimages = {}
        # initalize with model name 
        for model_name in predict_config.model.display_names:
            metrics_dict_allmodels_allimages[model_name] = {}
        
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.obs_img_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                (os.path.splitext(mask_fname[len(predict_config.indir):])[0]).split('/')[-1] + out_ext
            )
            
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            # import pdb; pdb.set_trace()
            disp_pred_result = []
            for model_i, model in enumerate(model_list):
                if predict_config.get('refine', False):
                    assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                    # image unpadding is taken care of in the refiner, so that output image
                    # is same size as the input image
                    cur_res = refine_predict(batch, model, **predict_config.refiner)
                    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = model(batch)                    
                        cur_gt = batch['image'][0].permute(1, 2, 0).detach().cpu().numpy()
                        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]
                # assert batch size is 1 for inference 
                assert len(cur_res.shape) == 3, f"Expected 3 dimensions, got {cur_res.dim()}"
                
                
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cur_gt = np.clip(cur_gt * 255, 0, 255).astype('uint8')
                cur_gt = cv2.cvtColor(cur_gt, cv2.COLOR_RGB2BGR)
                
                # Get mask overlaid on the input image
                mask = batch['mask'][0].detach().cpu().numpy()
                mask = np.clip(mask * 255, 0, 255).astype('uint8')
                red_mask = np.zeros_like(cur_gt)
                red_mask[:, :, 2] = mask  # Assuming mask is already binary and 2D
                # In areas where there is no observation, lower the other channels by a factor (e.g., 0.5)
                overlayed_img = np.copy(cur_gt)
                overlayed_img[mask[0] > 0] = (overlayed_img[mask[0] > 0] * 0.5).astype('uint8')
                overlayed_img[:, :, 2] = np.clip(overlayed_img[:, :, 0] + 0.5 * red_mask[:, :, 2], 0, 255)
                
                # Get current input image (with unobserved masked out)
                gt_masked = np.copy(cur_gt)
                gt_masked[mask[0] > 0] = 122

                # Calculate IOU metrics
                metrics_dict = {}
                # Initialize with iou_all, iou_masked, iou_observed keys 
                metrics_dict['iou_all'] = {}
                metrics_dict['iou_masked'] ={}
                metrics_dict['iou_observed'] = {}
                mask_onechan = mask[0]
                mask_expanded = np.repeat(np.expand_dims(mask_onechan, axis=-1), 3, axis=-1)
                metrics_dict['iou_all']['intersection'], metrics_dict['iou_all']['union'], metrics_dict['iou_all']['iou'] = calculate_iou_binary(cur_res, cur_gt) # IOU of whole map
                # IOU of masked region 
                metrics_dict['iou_masked']['intersection'], metrics_dict['iou_masked']['union'], metrics_dict['iou_masked']['iou'] = calculate_iou_binary(cur_res * mask_expanded, cur_gt * mask_expanded)
                # import pdb; pdb.set_trace()
                metrics_dict['iou_observed']['intersection'], metrics_dict['iou_observed']['union'], metrics_dict['iou_observed']['iou'] = calculate_iou_binary(cur_res * (255 - mask_expanded), cur_gt * (255 - mask_expanded))
                
            
                # display output (cur_input and cur_res stacked vertically)
                # add text to the output image (cur_res -> 'Result'), (cur_input -> 'Input')
                fontScale = 0.5
                fontThickness = 1
                model_name = predict_config.model.display_names[model_i]
                cur_res = cv2.putText(cur_res, '{}: iou {:.2f} masked {:.2f} observed {:.2f}'.format(\
                    model_name, metrics_dict['iou_all']['iou'], \
                        metrics_dict['iou_masked']['iou'], metrics_dict['iou_observed']['iou']), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), fontThickness, cv2.LINE_AA)
                cur_input = cv2.putText(overlayed_img, 'Current Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), fontThickness, cv2.LINE_AA)
                gt_masked = cv2.putText(gt_masked, 'GT (with mask)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), fontThickness, cv2.LINE_AA)
                # print("cur_res.shape", cur_res.shape)
                # print("gt_masked.shape", gt_masked.shape)
                disp_pred_result.append(cur_res)
            
                # Update metrics dict 
                metrics_dict_allmodels_allimages[model_name][mask_fname] = metrics_dict
            
                
            print(f"Writing to {cur_out_fname}")
            # print("cur_input.shape", cur_input.shape)
            # print("gt_masked.shape", gt_masked.shape)
            # print("pred_result[0].shape", pred_result[0].shape)
            # print("pred_result[1].shape", pred_result[1].shape)
            output = np.vstack((cur_input, gt_masked, np.vstack(disp_pred_result)))
            # import pdb; pdb.set_trace()
            cv2.imwrite(cur_out_fname, output)
            
        # Save metrics dict to file
        metrics_out_fname = os.path.join(predict_config.outdir, 'metrics_dict_allmodels_allimages.yaml')
        with open(metrics_out_fname, 'w') as f:
            yaml.dump(metrics_dict_allmodels_allimages, f)
        
        # Get average metrics
        # After processing all images for all models, calculate average IoUs from accumulated intersection and union
        for model_name, model_metrics in metrics_dict_allmodels_allimages.items():
            for metric_type in ['iou_all', 'iou_masked', 'iou_observed']:  # Add more metric types as needed
                # total_iou = 0
                total_intersection = 0
                total_union = 0
                count = 0
                for image_metrics in model_metrics.values():
                    if metric_type in image_metrics:
                        total_intersection += image_metrics[metric_type]['intersection']
                        total_union += image_metrics[metric_type]['union']
                        # print(f"Intersection: {image_metrics[metric_type]['intersection']}, Union: {image_metrics[metric_type]['union']}")
                        count += 1
                
                avg_iou = total_intersection / total_union if total_union != 0 else 0
                print(f"Average {metric_type} IoU for {model_name}: {avg_iou}")

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
