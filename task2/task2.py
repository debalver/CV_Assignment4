import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Get the coordinates for intersection
    x_left = max(prediction_box[0], gt_box[0])
    x_right = min(prediction_box[2], gt_box[2])
    y_bottom = max(prediction_box[1], gt_box[1])
    y_top = min(prediction_box[3], gt_box[3])
    # Compute intersection
    width = max(0, x_right-x_left)
    height = max(0,y_top-y_bottom)
    intersection = width*height
    # Compute union
    width_p = prediction_box[2] - prediction_box[0]
    height_p = prediction_box[3] - prediction_box[1]
    width_gt = gt_box[2] - gt_box[0]
    height_gt = gt_box[3] - gt_box[1]
    total_area = width_gt*height_gt + width_p*height_p
    union = total_area-intersection

    iou = intersection/union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp+num_fp == 0:
        return 1
    return num_tp/(num_tp+num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp+num_fn == 0:
        return 0
    return num_tp/(num_tp+num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    matches = [None for i in range(len(gt_boxes)*len(prediction_boxes))]
    index = 0
    # Iterate over all the ground truth and predicted boxes, get their intersection over union and store it in a tuple
    for i, gt in enumerate(gt_boxes):
        for j, predicted in enumerate(prediction_boxes):
            iou = calculate_iou(predicted, gt)
            matches[index] = (i, j, iou)
            index += 1

    # Sort all matches on IoU in descending order
    matches.sort(key=lambda tup: tup[2], reverse=True)
    # Remove the matches that are not superior to the threshold
    matches = list(filter(lambda x: x[2] > iou_threshold, matches))

    # Extract the index of the boxes that have been matched ONLY ONCE with the best iou
    assigned_gt = []
    assigned_predicted = []
    to_keep = []
    for i, val in enumerate(matches):
        if not (val[0] in assigned_gt or val[1] in assigned_predicted):
            assigned_gt.append(val[0])
            assigned_predicted.append(val[1])
            to_keep.append(i)

    # Create the lists that contain the matched bxoes
    matched_predicted = np.zeros((len(to_keep),4))
    matched_gt = np.zeros((len(to_keep),4))
    # Get the index of the boxes in the original lists passed as arguments
    # matches[index][0] get the index of the ground truth bounding box in the original list "gt_boxes"
    # matches[index][1] get the index of the predicted bounding box in the original list "prediction_boxes"
    for i, index in enumerate(to_keep):
        matched_gt[i] = gt_boxes[matches[index][0]]
        matched_predicted[i] = prediction_boxes[matches[index][1]]
    
    return matched_predicted, matched_gt


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # The number of False Negative fn is the number of unmatched ground truth boxes
    # The number of False Positive fp is the number of unmatched predicted boxes
    # The number of True Positive tp is the number of matched predicted boxes
    matched_predicted, matched_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    fn = len(gt_boxes) - len(matched_gt)
    fp = len(prediction_boxes) - len(matched_predicted)
    tp = len(matched_predicted)
    return {"true_pos" : tp, "false_pos" : fp, "false_neg" : fn}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp = 0
    fp = 0
    fn = 0
    # Iterate through the list of images and count all the tp, fp and fn
    for i in range(len(all_prediction_boxes)):
        res = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp += res['true_pos']
        fp += res['false_pos']
        fn += res['false_neg']
    precision = calculate_precision(tp, fp, fn)
    recall = calculate_recall(tp, fp, fn)
    return precision, recall


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = []
    recalls = []
    all_filtered_pred = all_prediction_boxes.copy()
    # Iterate over the confidence levels
    for confidence_threshold in confidence_thresholds:
        # Filter the prediction that do not reach the confidence threshold and iterate over the images
        for index, image in enumerate(all_prediction_boxes):
            # Use boolean mask to filter
            filtered_pred = image[confidence_scores[index] > confidence_threshold]
            all_filtered_pred[index] = filtered_pred
        # Calculate the precision and recall for the filtered list of predictions
        precision, recall = calculate_precision_recall_all_images(all_filtered_pred, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.title("Precision vs Recall curve")
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    interpolated_precisions = []
    # If there is no recall value for recall=1.0, then the precision if set to 0
    if not (1.0 in recalls):
        recalls = np.append(recalls, 1.0)
        precisions = np.append(precisions, 0)
    # To get the average precision we need to find the precision values for each recall levels
    # The precision for each level is the maximum precision to the right of the recall value
    for level in recall_levels:
        # Remove the recall values which are smaller than the recall level, by setting them to a negative value
        precisions[recalls < level] = np.NINF
        # Get the max precision for a given recall level
        max_precision = np.amax(precisions)
        interpolated_precisions.append(max_precision)

    average_precision = np.mean(interpolated_precisions)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
