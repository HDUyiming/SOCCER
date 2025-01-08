# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from icecream import ic 
import torch
import torch.distributed as dist
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from SOCCER import consis_Loss,visualize_detection,IOU_pred_GT,visualize_detection_GT_mask
from maskrcnn_benchmark.structures.bounding_box import BoxList
import copy

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

#--------------------
#     Source-only
#--------------------
def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

#--------------------
#     baseline
#--------------------
def do_da_train(
    model,
    source_data_loader,
    target_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()
        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

#--------------------
#       SOCCER
#--------------------

def do_soccer_da_train(
    model, model_teacher,
    source_data_loader,
    target_data_loader,
    SCM,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg,
    checkpointer_teacher,
    test
):
    from maskrcnn_benchmark.structures.image_list import ImageList
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    logger.info("---SOCCER---")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_teacher.eval()
    start_training_time = time.time()
    end = time.time()

    #------------------------------------------------------------
    #    Load the source and target domain data
    #------------------------------------------------------------
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        source_images = source_images.to(device)
        target_images = target_images.to(device)
        images = source_images+target_images
        targets = [target.to(device) for target in list(source_targets+target_targets)]

        #------------------------------------------------------------
        #    Generate stochastic complementary masked target images
        #------------------------------------------------------------

        masked_target_images, mirror_target_images = SCM(target_images.tensors.clone().detach())

        #---------------------------------------------------------------------
        #    Teacher model is updated by the student model through the EMA
        #---------------------------------------------------------------------

        model_teacher.update_weights(model, iteration)

        #-----------------------------------------------------------------------------
        #    Teacher model outputs the predictions for original target images
        #-----------------------------------------------------------------------------

        target_output = model_teacher(target_images)

        #-----------------------------------------------------------------------------
        #    Predictions of the teacher model  are filtered to generate pseudo-labels
        #-----------------------------------------------------------------------------

        target_pseudo_labels, pseudo_inds = filter_pred_to_pseudo_label(target_output, threshold=cfg.MODEL.PSEUDO_LABEL_THRESHOLD)
        
        #-----------------------------------------------------------------------------
        #    Supervised training...
        #-----------------------------------------------------------------------------
        
        record_dict = model(images, targets) 

        #-----------------------------------------------------------------------------
        #    We also provide visualization tools to observe the detection results 
        #----------------------------------------------------------------------------- 

        # visualize_detection(masked_target_images, target_output, iteration,task = 'mask_pred')
        # visualize_detection_GT_mask(target_images.tensors, target_targets, iteration,task = 'target_MASK_GT')
        # visualize_detection_GT(target_images.tensors, target_targets, iteration,task = 'target_img_GT')

        #-----------------------------------------------------------------------------
        #    Unsupervised training...
        #-----------------------------------------------------------------------------

        if len(target_pseudo_labels)>0:

            #-----------------------------------------------------------------------------
            #    Target masked data processing stage
            #-----------------------------------------------------------------------------

            masked_images = ImageList(masked_target_images[pseudo_inds], target_images.image_sizes)
            mirred_images = ImageList(mirror_target_images[pseudo_inds], target_images.image_sizes)

            #-----------------------------------------------------------------------------
            #    Target masked data loading stage
            #-----------------------------------------------------------------------------

            masked_loss_dict, masked_ROI = model(masked_images, target_pseudo_labels, SOCCER_mode = 'Stu_consis', use_pseudo_labeling_weight=cfg.MODEL.PSEUDO_LABEL_WEIGHT, with_DA_ON=False)
            mirror_loss_dict, mirror_ROI = model(mirred_images, target_pseudo_labels, SOCCER_mode = 'Stu_consis', use_pseudo_labeling_weight=cfg.MODEL.PSEUDO_LABEL_WEIGHT, with_DA_ON=False)
            
            #-----------------------------------------------------------------------------
            #    Inter-CCR (L_cls + L_reg)
            #-----------------------------------------------------------------------------            
            
            cons_loss_tea = consis_Loss( masked_ROI, mirror_ROI, mode = 'same_model_consis')
            record_dict.update(cons_loss_tea)
            
            #-----------------------------------------------------------------------------
            #    Intra-CCR (L_cls + L_reg)
            #-----------------------------------------------------------------------------

            weight_mask = masked_ROI[0].bbox.size(0) / (masked_ROI[0].bbox.size(0) + mirror_ROI[0].bbox.size(0))
            weight_mirror = mirror_ROI[0].bbox.size(0) / (masked_ROI[0].bbox.size(0) + mirror_ROI[0].bbox.size(0))
            pseudo_loss = {}
            for key in masked_loss_dict.keys():
                pseudo_loss[key + "_mami"] = masked_loss_dict[key] * weight_mask + mirror_loss_dict[key] * weight_mirror
            record_dict.update(pseudo_loss)        

        #-----------------------------------------------------------------------------
        #    weight losses (For simplicity, we set all weights to 1)
        #-----------------------------------------------------------------------------
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == 'loss_classifier_mami' or key == 'loss_objectness_mami':
                    loss_dict[key] = record_dict[key] * 1
                elif key == 'loss_box_reg_mami' or key == 'loss_rpn_box_reg_mami':
                    loss_dict[key] = record_dict[key] * 1
                elif key == "loss_CLA_consis":
                    loss_dict[key] = record_dict[key] * 1
                elif key == "loss_SMOOTH_L1_consis":
                    loss_dict[key] = record_dict[key] * 1  
                else:  
                    loss_dict[key] = record_dict[key] * 1  

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        #------------------------------------------
        #    checkpoint save model weight
        #------------------------------------------

        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            model_copy = copy.deepcopy(model)
            test(cfg, model_copy, False)
        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
            checkpointer_teacher.save("model_final_teacher", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

#-----------------------------------------------------------------------------------------------------
#    Filter out the low-confidence predictions of the teacher model to generate pseudo-labels
#-----------------------------------------------------------------------------------------------------

def filter_pred_to_pseudo_label(target_output, threshold=0.8):
    pseudo_labels_list = []
    inds = []
    for idx, bbox_l in enumerate(target_output):
        pred_bboxes = bbox_l.bbox.detach()
        labels = bbox_l.get_field('labels').detach()
        scores = bbox_l.get_field('scores').detach()
        probs  = bbox_l.get_field('probs').detach() 
        filtered_idx = scores>=threshold
        filtered_bboxes = pred_bboxes[filtered_idx]
        filtered_labels = labels[filtered_idx]
        filtered_scores = scores[filtered_idx]
        new_bbox_list = BoxList(filtered_bboxes, bbox_l.size, mode=bbox_l.mode)
        new_bbox_list.add_field("labels", filtered_labels)
        domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
        new_bbox_list.add_field("is_source", domain_labels)
        new_bbox_list.add_field("probs", probs[filtered_idx])
        new_bbox_list.add_field("scores", filtered_scores)

        if len(new_bbox_list)>0:
            pseudo_labels_list.append(new_bbox_list)
            inds.append(idx)

    return pseudo_labels_list, inds


