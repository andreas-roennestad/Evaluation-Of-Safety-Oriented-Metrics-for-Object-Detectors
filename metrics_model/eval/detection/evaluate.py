# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
import math
import numpy as np
import torch

from typing import Tuple, Dict, Any, List
from pyquaternion import Quaternion

from nuscenes.map_expansion.map_api import NuScenesMap
from planning_centric_metrics import calculate_pkl

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_ap_crit, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, summary_plot_crit, class_pr_curve, class_pr_curve_crit, class_tp_curve, dist_pr_curve, visualize_sample, visualize_sample_crit, visualize_sample_crit_r, visualize_sample_crit_d, visualize_sample_crit_t, visualize_sample_debug_1 
from nuscenes.eval.detection.utils import json_to_csv
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import boxes_to_sensor

model_name="None",
MAX_DISTANCE_OBJ=0.0,
MAX_DISTANCE_INTERSECT=0.0,
MAX_TIME_INTERSECT=0.0
recall_type="NONE"

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 MAX_DISTANCE_OBJ : float =100.0,
                 MAX_DISTANCE_INTERSECT : float =101.0,
                 MAX_TIME_INTERSECT_OBJ : float =102.0
                ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        DetectionBox.MAX_DISTANCE_OBJ  = MAX_DISTANCE_OBJ
        DetectionBox.MAX_DISTANCE_INTERSECT=MAX_DISTANCE_INTERSECT
        DetectionBox.MAX_TIME_INTERSECT_OBJ=MAX_TIME_INTERSECT_OBJ

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(nusc, self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        print("STARTING EVALUATION in evaluate(self)")
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')

        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, 
                                self.cfg.dist_fcn_callable, dist_th, path=self.output_dir,
                                model_name=self.model_name, 
                                MAX_DISTANCE_OBJ=self.MAX_DISTANCE_OBJ, 
                                MAX_DISTANCE_INTERSECT=self.MAX_DISTANCE_INTERSECT,
                                MAX_TIME_INTERSECT=self.MAX_TIME_INTERSECT,
                                recall_type=self.recall_type)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        f = open(self.output_dir+"/AP_SUMMARY.txt", "a")
        f.write("Model;MAX_DISTANCE_OBJ;MAX_DISTANCE_INTERSECT;MAX_TIME_INTERSECT;class_name;dist_th;ap;ap_crit\n")
        
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
                ap_crit=calc_ap_crit(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap_crit(class_name, dist_th, ap_crit)
                f.write(str(self.model_name)+
                        ";"+str(self.MAX_DISTANCE_OBJ)+
                        ";"+str(self.MAX_DISTANCE_INTERSECT)+
                        ";"+str(self.MAX_TIME_INTERSECT)+
                        ";"+str(class_name)+
                        ";"+str(dist_th)+
                        ";"+str(ap)+
                        ";"+str(ap_crit)+"\n")
                
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)
        f.close()


        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        def savepath_crit(name):
            return os.path.join(self.plot_dir, name + '_crit.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        summary_plot_crit(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
            dist_th_tp=self.cfg.dist_th_tp, savepath=savepath_crit('summary'))

        
        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_pr_curve_crit(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_crit_pr'))
            print(metrics)
            print(md_list)
            
            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))


    
        
    def calc_sample_crit(self, sample_token: str, save_path: str, verbose: bool = False, conf_th_sample: float = 0.15):
        # Get boxes.
        # choose specific samples(!)
        # Get boxes corresponding to sample
        if verbose:
            print("sample token gt boxes len: {}".format(len(self.gt_boxes.serialize()[sample_token])))
            print("sample token pred boxes len: {}".format(len(self.pred_boxes.serialize()[sample_token])))
        
        boxes_gt = EvalBoxes()
        boxes_pred = EvalBoxes()
        
        boxes_gt.add_boxes(sample_token, self.gt_boxes.boxes[sample_token])
        boxes_pred.add_boxes(sample_token, self.pred_boxes.boxes[sample_token])

        # ADD FPs, remove boxes for FNs, or add orientation errors here. 
        
        # Accumulate metric data for specific sample
        metric_data_list = DetectionMetricDataList()

        boxes_pred = self.filter_boxes_confidence(pred_boxes=boxes_pred, conf_th=conf_th_sample) # Filter BBs on confidence before PKL evaluation
        # FILTER TO ONLY INCLUDE CLASSES EVALUATED WITH OCM
        boxes_gt, boxes_pred = self.filter_boxes_class(gt_boxes=boxes_gt, pred_boxes=boxes_pred)

        #### IMPORTANT: WE ONLY USE DIST_TH = 2 FOR SINGLE SAMPLES ####
        dist_ths = [2.0]


        for class_name in self.cfg.class_names:
            for dist_th in dist_ths:
                md = accumulate(boxes_gt, boxes_pred, class_name, 
                                self.cfg.dist_fcn_callable, dist_th, path=save_path,
                                model_name=self.model_name, 
                                MAX_DISTANCE_OBJ=self.MAX_DISTANCE_OBJ, 
                                MAX_DISTANCE_INTERSECT=self.MAX_DISTANCE_INTERSECT,
                                MAX_TIME_INTERSECT=self.MAX_TIME_INTERSECT,
                                recall_type=self.recall_type, verbose=False, single_sample=True, conf_th_sample=conf_th_sample)
                metric_data_list.set(class_name, dist_th, md)


        # Calculate metrics from the data.
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)


        f = open(save_path+"/AP_summary.txt", "a")
        f.write("Model;class_name;dist_th;AP;AP_crit\n")

        # Compute APs.
        for class_name in self.cfg.class_names:
            for dist_th in dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
                ap_crit=calc_ap_crit(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap_crit(class_name, dist_th, ap_crit)
                ## Create summary in txt file ##
                # PARAMETERS MAX_DIST(...) are constant for these experiments and have ..
                # been evaluated at different values in Ceccarelli & Montecchi (2022)
                # See algo.py write to confusion_matrix.txt for lower level metrics
                f.write(str(self.model_name)+";"+
                        str(class_name)+";"+
                        str(dist_th)+";"+
                        str(ap)+";"+
                        str(ap_crit)+
                        "\n")

        f.close()
        # Mean_ap and mean_ap_crit are written to metrics_summary.json

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

        #AP_summary['mean_ap'] = metrics.mean_ap
        #AP_summary['mean_ap_crit'] = metrics.mean_ap_crit


        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()

        with open(os.path.join(save_path, 'metrics_summary.json'.format(sample_token)), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
    
        # CALCULATE PKLS
        

        #print(torch.cuda.device_count())
        #print(torch.cuda.is_available())
        gpuid = -1
        device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
                else torch.device('cpu')
        #print('using device: {0}'.format(device))

        map_folder = '/cluster/work/andronn/MasterThesis/MASTER/mmdetection3d/data/nuscenes/maps/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                        map_name=map_name) for map_name in [
                            "singapore-hollandvillage",
                            "singapore-queenstown",
                            "boston-seaport",
                            "singapore-onenorth",
                        ]}
        nworkers = 2

        
        pkl = calculate_pkl(boxes_gt, boxes_pred,
                                [sample_token], self.nusc,
                                nusc_maps, device,
                                nworkers=nworkers, bsz=16,
                                plot_kextremes=0,
                                verbose=True)
        with open(os.path.join(save_path,'pkl_results.json'), 'w') as fp:
            json.dump(pkl, fp)


        print("Saved metric data for sample {}".format(sample_token))
        



    def safety_metric_evaluation(self, sample_tokens: List[str], add_falses: bool = False, random_token_predictions: bool = False, 
                                    conf_th_sample: float = 0.15) -> None:
        """ collection of relevant samples
         and metric data for safety-oriented metrics.
         :param sample_tokens: list of sample tokens to evaluate.
         :param add_falses: save in alternate folder for storing results modified with FPs and FNs .
         :param inherited from main function """
    

        # Create necessary directories
        samples_directory = os.path.join(self.output_dir, 'METRIC_SAMPLES_MODIFIED' if add_falses else 'METRIC_SAMPLES')
        if random_token_predictions==True: samples_directory = os.path.join(self.output_dir, 'METRIC_SAMPLES_RANDOM')
        if not os.path.isdir(samples_directory):
                os.mkdir(samples_directory)

        for sample_token in sample_tokens:
            ##
            sample_dir = os.path.join(samples_directory, str(sample_token))
            if not os.path.isdir(sample_dir):
                os.mkdir(sample_dir)
                print("Made sample directory: {}\n".format(sample_dir))

            # Collect relevant samples with necessary annotations
            sample = self.nusc.get('sample', sample_token)
            
            
            ## Save lidar birds eye (with crit vals)
            if not random_token_predictions:
                visualize_sample(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                eval_range=max(self.cfg.class_range.values()),
                                savepath=os.path.join(sample_dir, 'LIDAR.png'.format(sample_token)),
                                verbose=False)
                visualize_sample_crit(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                eval_range=max(self.cfg.class_range.values()),
                                savepath=os.path.join(sample_dir, 'LIDAR_CRIT.png'.format(sample_token)),
                                verbose=False)
                visualize_sample_crit_r(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                eval_range=max(self.cfg.class_range.values()),
                                savepath=os.path.join(sample_dir, 'LIDAR_CRIT_R.png'.format(sample_token)),
                                verbose=False)
                visualize_sample_crit_t(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                eval_range=max(self.cfg.class_range.values()),
                                savepath=os.path.join(sample_dir, 'LIDAR_CRIT_T.png'.format(sample_token)),
                                verbose=False)
                visualize_sample_crit_d(self.nusc,
                                sample_token,
                                self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                # Don't render test GT.
                                self.pred_boxes,
                                eval_range=max(self.cfg.class_range.values()),
                                savepath=os.path.join(sample_dir, 'LIDAR_CRIT_D.png'.format(sample_token)),
                                verbose=False)
            
            ## Save sample specific metric data
            self.calc_sample_crit(sample_token=sample_token, save_path=sample_dir, conf_th_sample=conf_th_sample)
            
            ## Save images with annotations
            if not random_token_predictions:
                self.nusc.render_sample(sample_token, out_path=os.path.join(sample_dir, 'SENSOR_ANN_VIZ.png'.format(sample_token)), verbose=False)
            
            print("Sample {} data saved.\n".format(sample_token))
        

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True, 
             model_name="None",
             MAX_DISTANCE_OBJ=0.0,
             MAX_DISTANCE_INTERSECT=0.0,
             MAX_TIME_INTERSECT=0.0,
             recall_type="NONE",
             save_metrics_samples=False,
             single_sample_tokens=None,
             modified_predictions=False,
             random_token_predictions=False,
             conf_th_sample=0.15) -> Dict[str, Any]:

        self.model_name=model_name
        self.MAX_DISTANCE_OBJ=MAX_DISTANCE_OBJ
        self.MAX_DISTANCE_INTERSECT=MAX_DISTANCE_INTERSECT
        self.MAX_TIME_INTERSECT=MAX_TIME_INTERSECT
        self.recall_type=recall_type
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :param save_metrics_samples(bool) whether to save safety metrics related data for individual samples 
        :param single_sample_tokens: list of sample tokens for individual samples to be evaluated
        :param random_token_predictions: true if generating metric data for large amounts of single samples, used in plotting
        :param conf_th_sample: parameter passed down through safety_evaluation_metrics and calc_crit_sample as threshold for single samples.
        :return: A dict that stores the high-level metrics and meta data.
        """
        print("STARTING EVALUATION in main (self)")

        # ** HERE ** #
        if save_metrics_samples == True:
            pre_saved_samples = single_sample_tokens

            # Optionally add noise, remove or add BBs to test sensitivity of models to errors, noise, FPs, FNs (only for specific selected samples)
            self.safety_metric_evaluation(sample_tokens = pre_saved_samples, 
                                            add_falses=modified_predictions, 
                                            random_token_predictions=random_token_predictions,
                                            conf_th_sample=conf_th_sample)
                                    


        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            #random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Create images for debug_1 (ground truth only) images
            example_dir = os.path.join(self.output_dir, 'examples_gt_only')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            """for sample_token in sample_tokens:
                visualize_sample_debug_1(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))"""
                
            # Visualize samples without crit
            example_dir = os.path.join(self.output_dir, 'examples_clean')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_crit(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples with crit
            example_dir = os.path.join(self.output_dir, 'examples_crit')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_crit(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples with crit_r
            for sample_token in sample_tokens:
                visualize_sample_crit_r(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))
                
            # Visualize samples with crit_d
            for sample_token in sample_tokens:
                visualize_sample_crit_d(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))


            # Visualize samples with crit_t
            for sample_token in sample_tokens:
                visualize_sample_crit_t(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        if not save_metrics_samples:
            metrics, metric_data_list = self.evaluate()
        else:
            return

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
#        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
#       for tp_name, tp_val in metrics_summary['tp_errors'].items():
#           print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
#        print('NDS: %.4f' % (metrics_summary['nd_score']))
#        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
#        print()
#        print('Per-class results:')
#        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']

#        for class_name in class_aps.keys():
#            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#                  % (class_name, class_aps[class_name],
#                     class_tps[class_name]['trans_err'],
#                     class_tps[class_name]['scale_err'],
#                     class_tps[class_name]['orient_err'],
#                     class_tps[class_name]['vel_err'],
#                     class_tps[class_name]['attr_err']))

        return metrics_summary

    def filter_boxes_confidence(self, pred_boxes, conf_th: float = 0.15):
        """
        Filter Pred boxes on a confidence threshold. 
        :param pred_boxes: boxes to filter.
        :param conf_th: confidence threshold.
        """
        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                          box.detection_score >= conf_th]

        return pred_boxes

    def filter_boxes_class(self, gt_boxes, pred_boxes, classes: List[str] = ["car"]):
        """
        Filter GT and Pred boxes to only include a set of object classes. 
        :param pred_boxes: boxes to filter.
        :param classes: list of object class names.
        """
        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                          box.detection_name in classes]

        for ind, sample_token in enumerate(gt_boxes.sample_tokens):
            gt_boxes.boxes[sample_token] = [box for box in gt_boxes[sample_token] if
                                          box.detection_name in classes]

        return gt_boxes, pred_boxes


    def add_FP(self, sample_token: str, pos: Tuple[float, float], size: Tuple[float, float, float], match_ego_speed = False):
        """
        Add a FP prediction in coordinates in coordinates coords wrt. ego reference frame
        :param sample_token: Sample to operate on.
        :param pos: 2D coordinates to place FP in with format Tuple[x, y] (z same as ego).
        :param size: FP size.
        """
        print("Adding FP at position {0} relative to ego at sample {1}".format(pos, sample_token))     

        # Get ego reference
        sample = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_translation = pose_record['translation']
        ego_rotation = pose_record['rotation']
        ego_speed = self.pred_boxes[sample_token][0].ego_speed
        # Create FP box
        fp_box = DetectionBox(
            sample_token=sample_token,
            translation=ego_translation,
            rotation=ego_rotation,
            size=size,
            detection_score=0.99,
            num_pts=25,
            attribute_name='vehicle.moving' if match_ego_speed==True else 'vehicle.stopped',
            velocity=(ego_speed[0],ego_speed[1]) if match_ego_speed==True else (0,0),
            nusc=self.nusc,
        )
        
        # Create Box instance.
        box = Box(center=fp_box.translation, size=fp_box.size, orientation=Quaternion(fp_box.rotation), velocity=(fp_box.velocity[0],fp_box.velocity[1],0),
                    name=fp_box.detection_name, crit=fp_box.crit, crit_t=fp_box.crit_t, crit_r=fp_box.crit_r, crit_d=fp_box.crit_d)
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        # Place FP at pos coordinates (note: x and y coordinates are in opposite positions in ego frame: trans=(y,x,z))
        box.center[0]+=pos[1]
        box.center[1]-=pos[0] # positive x-axis to right

        # Transform to global ref frame
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))

        # Re-initialize Detection-Box to match new translation (to get correct criticalities)
        fp_box_m = DetectionBox(
            sample_token=sample_token,
            translation=box.center,
            rotation=ego_rotation,
            size=size,
            detection_score=0.99,
            num_pts=25,
            attribute_name='vehicle.moving' if match_ego_speed==True else 'vehicle.stopped',
            velocity=(ego_speed[0],ego_speed[1]) if match_ego_speed==True else (0,0),
            nusc=self.nusc,
        )
        
        # add box
        self.pred_boxes.add_boxes(sample_token, [fp_box_m])

    def add_FN(self, sample_token: str, dist: float = 10, remove_all: bool = True, classes: List[str] = ['car']):
        """
        Add FN to sample by removing BB that is less than dist from ego (dist as a measure of importance)
        :param sample_token: sample token.
        :param dist: distance within which predictions are removed.
        :param remove_all: remove all BBs that fit criteria dist(ego, pred)<10 (or first one detected).
        :param object classes to consider when removing BBs, e.g. ['car', 'truck']
        """
        
        print("Removing {0} within {1}m from ego at sample {2}".format("all BBs" if remove_all else "first found BB", dist, sample_token))
        def bbs_dist(trans_1, trans_2): 
            """ Return Euclidean dist between BB centers """
            return math.sqrt((trans_2[0]-trans_1[0])**2+(trans_2[1]-trans_1[1])**2+(trans_2[2]-trans_1[2])**2)

        sample = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_translation = pose_record['translation']
        it = 0
        shortest_dist = 999999.9
        shortest_dist_idx = None

        while it < len(self.pred_boxes[sample_token]):
            # Remove correctly predicted boxes within dist
            distance_pred = bbs_dist(ego_translation, self.pred_boxes[sample_token][it].translation)
            if distance_pred < dist and self.pred_boxes[sample_token][it].detection_name in classes:
                # Create Box instance for pred_box and transform to ego frame.
                pred_box = self.pred_boxes[sample_token][it]
                min_dist = 999999.9

                # Check if box is match (if not, cant be FN)
                for gt_idx, gt_box in enumerate(self.gt_boxes[pred_box.sample_token]):
                    # Find closest match among ground truth boxes
                    if gt_box.detection_name in classes:
                        this_distance = self.cfg.dist_fcn_callable(gt_box, pred_box)
                        if this_distance < min_dist:
                            min_dist = this_distance

                # If the closest match is close enough according to threshold we have a match!
                is_match = min_dist < 2.0 # dist_th 2.0
                if not is_match: 
                    it+=1
                    continue
                
                # Transform to ego ref frame and check location
                box = Box(center=pred_box.translation, size=pred_box.size, orientation=Quaternion(pred_box.rotation), velocity=(pred_box.velocity[0], pred_box.velocity[1],0))
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                # if y-value of box.center is positive, predicted box is in front of ego (and is a candidate for removal)
                """if not box.center[0] > 1.0:
                    # 1.0 so that vehicles right next to ego with very similar y-val are excluded
                    it += 1
                    continue"""
                #del self.pred_boxes[sample_token][it]
                #if not remove_all: break
                if remove_all:
                    del self.pred_boxes[sample_token][it]
                    continue

                if distance_pred < shortest_dist:
                    shortest_dist = distance_pred
                    shortest_dist_idx = it
                
                
            it += 1
        if not remove_all and shortest_dist_idx:
            del self.pred_boxes[sample_token][shortest_dist_idx]

class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    print("STARTING EVALUATION in NuScene Eval -- should not be used")

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
