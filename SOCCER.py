import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import warnings
from typing import Optional, Sequence
from torch.nn import KLDivLoss
from icecream import ic 
from PIL import Image, ImageDraw
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import KLDivLoss
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.layers import smooth_l1_loss
import torchvision.transforms.functional as TF
warnings.filterwarnings("ignore", category=DeprecationWarning)



#-----------------------------------------------------
#   We provide some visualization tools
#-----------------------------------------------------


#-----------------------------------------------------
#   Print original images
#-----------------------------------------------------

def visualize(image, iteration, task = ''):
    if iteration % 1000  == 0:    
        image = torchvision.transforms.functional.to_pil_image(image[0])
        image.save('./show_images/visualization——{}_{}.png'.format(task,iteration))
        return image

def show_img(img,iteration):
    if iteration % 1000 == 0 :
        image = img.detach().cpu().numpy() 
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)  
        plt.imshow(image)
        plt.savefig('./show_images/image_S_{}.png'.format(iteration))

#-----------------------------------------------------
#   Print the detection results
#-----------------------------------------------------

def visualize_detection(image, result, iteration, task=''):
    if iteration % 1 == 0:
        # Convert the image to a PIL Image object
        image = TF.to_pil_image(image[0])

        draw_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(draw_image)

        #-------------------------
        # Foggy Cityscapes
        #-------------------------
        class_colors = {
            1: "red",       # person
            2: "yellow",    # rider
            3: "blue",      # car 
            4: "orange",    # truck
            5: "purple",    # bus
            6: "pink",      # train (BDD100K is not)
            7: "cyan",      # motorcycle
            8: "magenta"    # bicycle
        }
        #-------------------------
        #SIM10K
        #-------------------------
        # class_colors = {
        #     1: "blue",       # person
        # }
        #-------------------------
        # BDD100K
        #-------------------------
        # class_colors = {
        #     1: "red",       # person
        #     2: "yellow",    # rider
        #     3: "blue",      # car 
        #     4: "orange",    # truck
        #     5: "purple",    # bus
        #     6: "cyan",      # motorcycle
        #     7: "magenta"    # bicycle
        # }

        class_colors = {
            1: "blue",       # person
        }

        for i, box in enumerate(result[0].bbox):
            
            score = result[0].get_field("scores")[i]

            # for better visualization, we use the confidence threshold of 0.5
            if score > 0.5:
            
                x1, y1, x2, y2 = box.tolist()

                # Convert the tensor label to a Python integer
                label = int(result[0].get_field("labels")[i])

                # Draw bounding box with the color corresponding to the class
                draw.rectangle((x1, y1, x2, y2), outline=class_colors[label], width=6, fill=None)

        # Save the result image
        result_image = Image.alpha_composite(image.convert("RGBA"), draw_image)
        result_image.save('./show_images/visualization——{}_{}.png'.format(task,iteration))
        return result_image

 
#-----------------------------------------------------------------------
#   This is a supplementary experiment to generate Figure S1
# ---We want to explore how many objects are completely masked out---
#-----------------------------------------------------------------------

def visualize_detection_GT_mask(image, labels, iteration, input_mask, task=''):

        image = torchvision.transforms.functional.to_pil_image(image[0])
        draw_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(draw_image)

        for i, box in enumerate(labels[0].bbox):

            # Get the box coordinates
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if input_mask[:, :,y1:y2+1, x1:x2+1].sum() < 1 :
                draw.rectangle((x1, y1, x2, y2), outline="red", width=6)
            else:
                draw.rectangle((x1, y1, x2, y2), outline="green", width=6)

        result_image = Image.alpha_composite(image.convert("RGBA"), draw_image)
        result_image.save('./show_images/visualization——{}_{}.png'.format(task,iteration))
        return result_image               

#-----------------------------------------------------
# Print the ground truth
#-----------------------------------------------------

def visualize_detection_GT(image, result, iteration, task=''):
    if iteration % 1 == 0:

        image = TF.to_pil_image(image[0])
        draw_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(draw_image)
        #-------------------------
        # Foggy Cityscapes
        #-------------------------
        class_colors = {
            1: "red",       # person
            2: "yellow",    # rider
            3: "blue",      # car 
            4: "orange",    # truck
            5: "purple",    # bus
            6: "pink",      # train (BDD100K is not)
            7: "cyan",      # motorcycle
            8: "magenta"    # bicycle
        }
        #-------------------------
        #SIM10K
        #-------------------------
        # class_colors = {
        #     1: "blue",       # person
        # }
        #-------------------------
        # BDD100K
        #-------------------------
        # class_colors = {
        #     1: "red",       # person
        #     2: "yellow",    # rider
        #     3: "blue",      # car 
        #     4: "orange",    # truck
        #     5: "purple",    # bus
        #     6: "cyan",      # motorcycle
        #     7: "magenta"    # bicycle
        # }

        for i, box in enumerate(result[0].bbox):
            x1, y1, x2, y2 = box.tolist()

            label = int(result[0].get_field("labels")[i])

            draw.rectangle((x1, y1, x2, y2), outline=class_colors[label], width=6, fill=None)

        result_image = Image.alpha_composite(image.convert("RGBA"), draw_image)
        result_image.save('./show_images/visualization_{}_{}.png'.format(task, iteration))
        return result_image
        
#-----------------------------------------------------
# Inter-CCR
#-----------------------------------------------------
def consis_Loss( ROI_result_1, ROI_result_2, mode = 'none', loss_name = ''):

    IOU_THRESHOLD = 0.75
    CLA_CONSIS = []
    BOX_CONSIS = []
    BOX_ALL_CONSIS = []
    BOX_MSE_CONSIS = []
    BOX_SMOOTH_CONSIS = []
    BOX_REG_DIOU = []


    for ROI_1, ROI_2 in zip(ROI_result_1, ROI_result_2):
        pred_box_num_1 = ROI_1.bbox.size(0)
        pred_box_num_2 = ROI_2.bbox.size(0)
        # if mode == 'Tea_consis_Stu':
        if mode == 'same_model_consis':
            if pred_box_num_1 > pred_box_num_2:
                ROI_1, ROI_2 =  ROI_2, ROI_1

        IOU_matrix = boxlist_iou(ROI_1,ROI_2)    
        if IOU_matrix.numel() == 0:
                if IOU_matrix.shape[0] == 0:   
                    ic('The number of rows is 0 then ROI 1 does not detect the objects')
                    return {}
                else:
                    ic('The number of columns is 0 then ROI 2 does not detect the objects')
                    return {}
        matched_vals, matches = IOU_matrix.max(dim=0)
        no_selected_index = matched_vals < IOU_THRESHOLD
        selected_index = matched_vals >= IOU_THRESHOLD
        matches[no_selected_index] = -1 
        matches = matches[matches != -1]
        matches_list = matches.tolist()
        match_ROI_1 = ROI_1[matches_list]  
        match_ROI_2 = ROI_2[selected_index]        
        if len(match_ROI_1) ==0 and len(match_ROI_2)==0:
            return {}
        class_consis_loss = Class_Consistency_Losses().losses( match_ROI_1, match_ROI_2, mode = mode)
        #eq.(7)
        CLA_CONSIS.append(class_consis_loss)
        #eq.(8)
        matchec_index = IOU_matrix >= IOU_THRESHOLD
        box_regression_smooth_l1_loss = box_reg_smooth_l1_consis(ROI_1,ROI_2,matchec_index)
        BOX_SMOOTH_CONSIS.append(box_regression_smooth_l1_loss)

        #--------------------------------------------------------------------
        #  We also provide several Inter-CCR consistency regression losses
        #--------------------------------------------------------------------

        #--------------------
        # Matched IoU loss
        #--------------------
        # box_regression_consis_loss = 1 - matched_vals[selected_index].mean()
        # BOX_CONSIS.append(box_regression_consis_loss)
        #--------------------
        # Global IoU loss
        #--------------------
        # box_regression_consis_loss_entirety = box_reg_consis(ROI_1,ROI_2)
        # BOX_ALL_CONSIS.append(box_regression_consis_loss_entirety)
        #--------------------
        # MSE reg loss
        #--------------------
        # matchec_index = IOU_matrix >= IOU_THRESHOLD
        # box_regression_MSE_loss = box_reg_MSE_consis(ROI_1,ROI_2,matchec_index)
        # BOX_MSE_CONSIS.append(box_regression_MSE_loss)
        #--------------------
        # DIoU reg loss
        #--------------------
        # bbox_1 = ROI_1.bbox.detach()
        # bbox_2 = ROI_2.bbox.detach()
        # box_reg_diou_loss = diou(bbox_1[matches_list], bbox_2[selected_index])
        # BOX_REG_DIOU.append(box_reg_diou_loss)
    loss = {}
    loss['loss_CLA_consis{}'.format(loss_name)]  = sum(CLA_CONSIS)/len(CLA_CONSIS)

    loss['loss_SMOOTH_L1_consis{}'.format(loss_name)] = sum(BOX_SMOOTH_CONSIS)/len(BOX_SMOOTH_CONSIS)
    
    # loss['loss_BOX_consis{}'.format(loss_name)] = sum(BOX_CONSIS)/len(BOX_CONSIS)

    # loss['loss_BOX_ALL_consis{}'.format(loss_name)] = sum(BOX_ALL_CONSIS)/len(BOX_ALL_CONSIS)

    # loss['loss_BOX_MSE_consis{}'.format(loss_name)] = sum(BOX_MSE_CONSIS)/len(BOX_MSE_CONSIS)

    # loss['loss_BOX_DIOU_consis{}'.format(loss_name)] = sum(BOX_REG_DIOU)/len(BOX_REG_DIOU)

    

    return loss

#-----------------------------------------------------
#   eq.(7)
#-----------------------------------------------------

class Class_Consistency_Losses:
    def __init__(self):
        self.kldivloss = KLDivLoss(reduction="none", log_target=False)

    def losses(self, teacher_roi, student_roi, mode = 'none'):

        class_scores_student = []
        class_scores_teacher = []
        class_label_student = []
        class_label_teacher = []


        class_scores_student.append(student_roi.get_field('probs').detach()) 
        class_scores_teacher.append(teacher_roi.get_field('probs').detach()) 
        class_label_student.append(student_roi.get_field('labels'))
        class_label_teacher.append(teacher_roi.get_field('labels'))
        class_scores_student=torch.cat(class_scores_student,axis=0)
        class_scores_teacher=torch.cat(class_scores_teacher,axis=0)
        class_label_student=torch.cat(class_label_student,axis=0)
        class_label_teacher=torch.cat(class_label_teacher,axis=0)

        if mode == 'Tea_consis_Stu':
            ic("for the future work")
        if mode == 'same_model_consis':
            softmax_weight_student = torch.softmax(class_scores_student.detach(), dim=1)
            softmax_weight_teacher = torch.softmax(class_scores_teacher.detach(), dim=1)
            weight_student = torch.max(softmax_weight_student, dim=1)[0]
            weight_teacher = torch.max(softmax_weight_teacher, dim=1)[0]
            
            classification_loss_a = F.cross_entropy(class_scores_teacher, class_label_student)
            classification_loss_b = F.cross_entropy(class_scores_student, class_label_teacher)
            cross_entropy_loss_a = torch.mean(weight_student * classification_loss_a)
            cross_entropy_loss_b = torch.mean(weight_teacher * classification_loss_b)
            cross_entropy_loss = torch.div(cross_entropy_loss_a + cross_entropy_loss_b, 2)

        return cross_entropy_loss

#-----------------------------------------------------
# eq.(8)
# Different from the paper report, we applied Huber loss to the ROI regression loss of Intra-CCR： /home/cuiyiming/SOCCER/maskrcnn_benchmark/modeling/roi_heads/box_head/loss.py
# You can also replace both with Huber loss, but we found that the replacement works best only in Intra-CCR
#-----------------------------------------------------

def box_reg_smooth_l1_consis(ROI_1,ROI_2,matchec_index):
    ROI_1_indices, ROI_2_indices = torch.where(matchec_index)

    regression_targets_1 = ROI_1.get_field('regression_targets').detach()
    regression_targets_2 = ROI_2.get_field('regression_targets').detach()


    box_loss = smooth_l1_loss(
            regression_targets_1[ROI_1_indices],
            regression_targets_2[ROI_2_indices],
            size_average=True,
            beta=1,
        )

    return box_loss

def box_reg_consis(ROI_1,ROI_2):
    W, H = ROI_1.size
    zero_matrix1 = np.zeros((H, W), dtype=np.float)
    for box in ROI_1.bbox:
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        zero_matrix1[y1:y2+1, x1:x2+1] = 1  
    zero_matrix2 = np.zeros((H, W), dtype=np.float)
    for box in ROI_2.bbox:
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        zero_matrix2[y1:y2+1, x1:x2+1] = 1

    intersection = np.logical_and(zero_matrix1, zero_matrix2).sum()
    union = np.logical_or(zero_matrix1, zero_matrix2).sum()
    box_regression_loss_entirety = 1 - intersection / union

    return box_regression_loss_entirety

def box_reg_MSE_consis(ROI_1,ROI_2,matchec_index):
    ROI_1_indices, ROI_2_indices = torch.where(matchec_index)
    reg_loss = 0
    regression_targets_1 = ROI_1.get_field('regression_targets').detach()
    regression_targets_2 = ROI_2.get_field('regression_targets').detach()

    for n, (i, j) in enumerate(zip(ROI_1_indices, ROI_2_indices)):
        reg_loss += ((regression_targets_1[i] - regression_targets_2[j]) ** 2).mean()

    box_reg_loss = (reg_loss / (n + 1)) 
    return box_reg_loss


def diou(bboxes1, bboxes2):

    # Extract coordinates from [x1, y1, x2, y2] format
    x1_1, y1_1, x2_1, y2_1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    # Calculate intersection coordinates and area
    inter_l = torch.max(x1_1, x1_2)
    inter_r = torch.min(x2_1, x2_2)
    inter_t = torch.max(y1_1, y1_2)
    inter_b = torch.min(y2_1, y2_2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    # Calculate diagonal coordinates and squared diagonal length
    c_l = torch.min(x1_1, x1_2)
    c_r = torch.max(x2_1, x2_2)
    c_t = torch.min(y1_1, y1_2)
    c_b = torch.max(y2_1, y2_2)
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    # Calculate IoU (Intersection over Union) and distance factor
    inter_diag = (((x1_2 +x2_2) / 2) - ((x1_1 + x2_1) / 2)) ** 2 + (((y1_2 +y2_2) / 2) - ((y1_1 + y2_1) / 2)) ** 2 
    union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    dious = iou - u

    # Clamp DIoU values to the range [-1, 1]
    dious = torch.clamp(dious, min=-1.0, max=1.0)


    # Calculate and return the sum of 1 - DIoU
    return torch.mean(1 - dious)

def IOU_pred_GT( preds, gts):

    W, H = gts[0].size
    zero_matrix_pred = np.zeros((H, W), dtype=np.float)
    zero_matrix1_gt  = np.zeros((H, W), dtype=np.float)
    for pred in preds[0].bbox:
        x1, y1, x2, y2 = pred.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        zero_matrix_pred[y1:y2+1, x1:x2+1] = 1  

    for gt in gts[0].bbox:
        xx1, yy1, xx2, yy2 = gt.tolist()
        xx1, yy1, xx2, yy2 = int(xx1), int(yy1), int(xx2), int(yy2)     
        zero_matrix1_gt[yy1:yy2+1, xx1:xx2+1] = 1  
    intersection = np.logical_and(zero_matrix_pred, zero_matrix1_gt).sum()
    union = np.logical_or(zero_matrix_pred, zero_matrix1_gt).sum()
    iou = intersection / union

    return iou


def random_crop( img, crop_size, target_pseudo_labels):
    if len(target_pseudo_labels) > 0:
        img=img.tensors.float()[0]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()  

        height, width = img.shape[-2:]
        crop_width = int(crop_size)
        crop_height = int(crop_size)

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        pseudo_label = filter_and_adjust_bboxes( crop_size, top, bottom, right, left, target_pseudo_labels)
        if len(pseudo_label) > 0:
            cropped_img = img[..., top:bottom, left:right]  

            if isinstance(cropped_img, np.ndarray):
                cropped_img = torch.from_numpy(cropped_img)  
                cropped_img = cropped_img.unsqueeze(0)

            return cropped_img, pseudo_label
        else:
            return [], []


def filter_and_adjust_bboxes(crop_size, top, bottom, right, left, target_pseudo_labels):
    from maskrcnn_benchmark.structures.bounding_box import BoxList
    
    new_pseudo_labels_list = []
    
    for bbox in target_pseudo_labels:
        pseudo_bboxes = bbox.bbox.detach()
        pseudo_bboxes111 = bbox.bbox.detach()
        labels = bbox.get_field('labels').detach()
        scores = bbox.get_field('scores').detach()
        probs = bbox.get_field('probs').detach()
        
        filtered_idx = ((left < pseudo_bboxes[:, 2]) & (right > pseudo_bboxes[:, 0] )) & ((top < pseudo_bboxes[:, 3]) & (bottom > pseudo_bboxes[:, 1]))
        pseudo_bboxes[:, 0] = torch.max(pseudo_bboxes[:, 0], torch.tensor(left, device=pseudo_bboxes.device))
        pseudo_bboxes[:, 1] = torch.max(pseudo_bboxes[:, 1], torch.tensor(top, device=pseudo_bboxes.device))
        pseudo_bboxes[:, 2] = torch.min(pseudo_bboxes[:, 2], torch.tensor(right, device=pseudo_bboxes.device))
        pseudo_bboxes[:, 3] = torch.min(pseudo_bboxes[:, 3], torch.tensor(bottom, device=pseudo_bboxes.device))
        
        filtered_bboxes = pseudo_bboxes[filtered_idx]
        filtered_labels = labels[filtered_idx]
        filtered_scores = scores[filtered_idx]
        filtered_probs  = probs[filtered_idx]

        if len(filtered_bboxes) > 0:
            new_bbox_list = BoxList(filtered_bboxes, (crop_size,crop_size), mode=bbox.mode)
            domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
            new_bbox_list.add_field("labels", filtered_labels)
            new_bbox_list.add_field("is_source", domain_labels)
            new_bbox_list.add_field("probs", filtered_probs)
            new_bbox_list.add_field("scores", filtered_scores)
            new_pseudo_labels_list.append(new_bbox_list)
        
    return new_pseudo_labels_list


#------------------------------------------------------------------------------------------------------------------------
#   This is a supplementary experiment: we also explore addressing unfair penalization at the label level: 
#                                       removing completely masked objects from the pseudo-label.
#------------------------------------------------------------------------------------------------------------------------

def adjust_label(label,mask):
    from maskrcnn_benchmark.structures.bounding_box import BoxList
    pseudo_labels_list = []
    selection = []
    for idx, bbox_l in enumerate(label):
        pred_bboxes = bbox_l.bbox.detach()
        labels = bbox_l.get_field('labels').detach()
        scores = bbox_l.get_field('scores').detach()
        probs  = bbox_l.get_field('probs').detach() 

        for box in pred_bboxes:
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if mask[:, :,y1:y2+1, x1:x2+1].sum() < 1 :
                selection.append(False)
            else:
                selection.append(True)
        selection = torch.tensor(selection)
        ic(selection)
        filtered_bboxes = pred_bboxes[selection]
        filtered_labels = labels[selection]
        filtered_scores = scores[selection]
        new_bbox_list = BoxList(filtered_bboxes, bbox_l.size, mode=bbox_l.mode)
        new_bbox_list.add_field("labels", filtered_labels)
        domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
        new_bbox_list.add_field("is_source", domain_labels)
        new_bbox_list.add_field("probs", probs[selection])
        new_bbox_list.add_field("scores", filtered_scores)

        if len(new_bbox_list)>0:
            pseudo_labels_list.append(new_bbox_list)

    return pseudo_labels_list
