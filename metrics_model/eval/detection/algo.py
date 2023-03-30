# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from typing import Callable
import numpy as np
import math
import json
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False,
               single_sample: bool = False, 
               path="None",
               model_name="None",
               MAX_DISTANCE_OBJ=0.0,
               MAX_DISTANCE_INTERSECT=0.0,
               MAX_TIME_INTERSECT=0.0,
               recall_type="NONE") -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    """#count positives with GT values for crit, crit_r, crit_t, crit_d 
    gt_crit=0.0
    gt_crit_r=0.0
    gt_crit_t=0.0
    gt_crit_d=0.0
    for gt_box in gt_boxes.all:
        gt_crit=gt_crit + gt_box.crit
        gt_crit_r=gt_crit_r + gt_box.crit_r
        gt_crit_d=gt_crit_d + gt_box.crit_d
        gt_crit_t=gt_crit_t + gt_box.crit_t"""


    conf_thresh_sample = 0.15
    # Count the positives.###
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])

    npos_crit1=[]
    for gt_box in gt_boxes.all:
        if (gt_box.detection_name == class_name):
            npos_crit1.append(gt_box.crit)

    npos_crit=0.0
    for i in npos_crit1:
        if(i>0.0): #TODO: il problema è nei NaN in data_classes, dovuti alle divisioni per 0. Qualcosa da aggiustare nelle formule.
            npos_crit=npos_crit+i

    

    ####
    if verbose:
        print("number values:  {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))
        print("Total value of GT crit npos_crit:  {}  of class {} out of {} total across {} samples.".
              format(npos_crit, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0: # if viewing individual samples this will generate warning for all classes not present in scene
        print("ALGO.PY WARNING : NO GT PREDICTIONS, CLASS: {}".format(class_name))
        return DetectionMetricData.no_predictions()
    if npos_crit == 0:
        print("ALGO.PY WARNING : NO CRIT GT PREDICTIONS, CLASS: {}".format(class_name))
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.
    tp_pred_crit = []  # Accumulator of true positives crit. predicted
    fp_pred_crit = []  # Accumulator of false positives crit. predicted
    tp_gt_crit = []  # Accumulator of true positives crit. ground truth

    # For single sample analyzing (snapshot at specific confidence) 
    tp_s = []  # Accumulator of true positives.
    fp_s = []  # Accumulator of false positives.
    tp_s_crit = []  # Accumulator of true positives crit. predicted above conf_th
    fp_s_crit = []  # Accumulator of false positives crit. predicted above conf_th
    tp_gts_crit = []  # Accumulator of true positives crit. ground truth above conf_th

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': [],
                 }

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            tp_pred_crit.append(pred_box.crit)
            
            
            fp.append(0)
            fp_pred_crit.append(0)

            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            tp_gt_crit.append(gt_box_match.crit)

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
            
            
            if single_sample and pred_box.detection_score > conf_thresh_sample: 
                # Boxes are sorted so all rest are above conf_th (to get low level metrics for conf_th)
                tp_s.append(1)
                tp_s_crit.append(pred_box.crit)
                
                fp_s.append(0)
                fp_s_crit.append(0)
                tp_gts_crit.append(gt_box_match.crit)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            tp_pred_crit.append(0)
            tp_gt_crit.append(0)

            fp.append(1)
            fp_pred_crit.append(pred_box.crit)
            conf.append(pred_box.detection_score)

            if single_sample and pred_box.detection_score > conf_thresh_sample:
                tp_s.append(0)
                tp_s_crit.append(0)
                tp_gts_crit.append(0)

                fp_s.append(1)
                fp_s_crit.append(pred_box.crit)


    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    tp_pred_crit_app=tp_pred_crit #per utilizzarle dopo
    tp_gt_crit_app=tp_gt_crit
    fp_pred_crit_app=fp_pred_crit
    
    # Accumulate.
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    tp_pred_crit=np.cumsum(tp_pred_crit).astype(np.float)
    tp_gt_crit=np.cumsum(tp_gt_crit).astype(np.float)     #TODO: il problema è nei NaN in data_classes, dovuti alle divisioni per 0. Qualcosa da aggiustare nelle formule.
    fp_pred_crit=np.cumsum(fp_pred_crit).astype(np.float)
    conf = np.array(conf)
    

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    
    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    
    if(recall_type=="GT AL NUMERATORE"):
        prec_crit= tp_pred_crit/(fp_pred_crit + tp_pred_crit) #TODO: ha dato 1 warning su runtime value not valid???
        rec_crit= tp_gt_crit/(npos_crit)
    elif(recall_type=="PRED AL NUMERATORE"):
        rec_crit= tp_pred_crit/(npos_crit)
        prec_crit= tp_gt_crit/(fp_pred_crit + tp_pred_crit) #TODO: ha dato 1 warning su runtime value not valid???
        for i in range(len(rec_crit)):
            if (rec_crit[i]>1.0):
                rec_crit[i]=1.0
        for i in range(len(prec_crit)):
            if (prec_crit[i]>1.0):
                prec_crit[i]=1.0
    else:
        print("BIG MISTAKE IN ALGO.PY ON COMPUTING RECALL_CRIT")
        sys.exit(0)  


    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    rec_crit_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.

    prec = np.interp(rec_interp, rec, prec, right=0)
    prec_crit = np.interp(rec_crit_interp, rec_crit, prec_crit, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp
    rec_crit = rec_crit_interp
    
    if not single_sample:
        f = open(path+"/confusion_matrix.txt", "a")
        f.write("Model;MAX_DISTANCE_OBJ;MAX_DISTANCE_INTERSECT;MAX_TIME_INTERSECT;class_name;dist_th;max_tp;max_fp;min_fn;max_tp_pred_crit;max_tp_gt_crit;max_fp_pred_crit;min_fn_gt_crit\n")
        f.write(str(model_name)+
                ";"+str(MAX_DISTANCE_OBJ)+
                ";"+str(MAX_DISTANCE_INTERSECT)+
                ";"+str(MAX_TIME_INTERSECT)+
                ";"+str(class_name)+
                ";"+str(dist_th)+
                ";"+str(np.amax(tp))+
                ";"+str(np.amax(fp))+
                ";"+str(np.sum(npos)-np.amax(tp))+
                ";"+str(np.amax(tp_pred_crit))+
                ";"+str(np.amax(tp_gt_crit))+
                ";"+str(np.amax(fp_pred_crit))+
                ";"+ str(np.sum(npos_crit)-np.amax(tp_pred_crit))+
                "\n")
        f.close()
    else:
        # Precision and recall values for sample computed
        # IMPORTANT: This is a "snapshot" of low-level metrics (Precision, Recall, TPs, (...)) at the SAME confidence value that the metric to be compared with is evaluated
        tp_s = np.sum(tp_s).astype(np.float)
        fp_s = np.sum(fp_s).astype(np.float)
        tp_s_crit = np.sum(tp_s_crit).astype(np.float)
        fp_s_crit = np.sum(fp_s_crit).astype(np.float)
        tp_gts_crit = np.sum(tp_gts_crit).astype(np.float)
        prec_s = tp_s / (fp_s + tp_s)
        rec_s = tp_s / float(npos)
        #prec_s_crit = tp_s_crit / (fp_s_crit + tp_s_crit)
        #rec_s_crit = tp_s_crit / float(npos_crit)


        if(recall_type=="GT AL NUMERATORE"):
            prec_s_crit= tp_s_crit/(fp_s_crit + tp_s_crit) 
            rec_s_crit= tp_gts_crit/(npos_crit)
        elif(recall_type=="PRED AL NUMERATORE"):
            rec_s_crit= tp_s_crit/(npos_crit)
            prec_s_crit= tp_gts_crit/(fp_s_crit + tp_s_crit) 
            if (rec_s_crit>1.0):
                rec_s_crit=1.0
            if (prec_s_crit>1.0):
                prec_s_crit=1.0
            f = open(path+"/threshold_metrics.txt", "a")
            f.write("Model;Class;dist_th;conf_th;TPs;FPs;FNs;TPs_pred_crit;TPs_gt_crit;FPs_pred_crit;FNs_gt_crit;Prec;Rec;Prec_crit;Rec_crit\n")
            f.write(str(model_name)+
                    ";"+str(class_name)+
                    ";"+str(dist_th)+
                    ";"+str(conf_thresh_sample)+
                    ";"+str(tp_s)+
                    ";"+str(fp_s)+
                    ";"+str(npos-tp_s)+
                    ";"+str(tp_s_crit)+
                    ";"+str(tp_gts_crit)+
                    ";"+str(fp_s_crit)+
                    ";"+ str(npos_crit-tp_s_crit)+   
                    ";"+ str(prec_s)+
                    ";"+ str(rec_s)+
                    ";"+ str(prec_s_crit)+
                    ";"+ str(rec_s_crit) +
                    "\n")
            f.close()

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]




    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               recall_crit=rec_crit,
                               precision=prec,
                               precision_crit=prec_crit,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])

def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)

def calc_ap_crit(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision_crit)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)

def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
