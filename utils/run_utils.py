import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.preprocess import create_weight_mask

import numpy as np
from itertools import product

import os

from model.QuadNet import assign_nets_to_coords


def train_epoch(model, data_loader, loss, optimizer, model_device, loss_device):
    """
    Runs one patch based training epoch for RCVNet, CropNet3D and (single loss) MultiResVNet. The volumetric cropping is
    performed in the model and not in this function
    
    Args:
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.Dataloader): The dataloader for the evaluation
        loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
        model_device (torch.cuda.device): The device the model is stored on.
        loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                         needed.
    
    Returns:
         float: the total loss over the dataset
    """
    
    model.train()       # Set model to train mode
    loss.train()        # Set loss to train mode. Needed only to change the values returned by CombinedLoss
    loss_itr = 0

    # Adjusting end point of iteration for tqdm (progress bar)
    itr_end = len(data_loader.dataset)
    
    optimizer.zero_grad()   # Set optimizer zero grad initially
    
    with tqdm(total=itr_end) as pbar:
        
        # Iterate over the dataset
        for batch_id, (volume, aseg, weights) in enumerate(data_loader, 0):

            # Move volume and aseg to model and loss devices respectively and change aseg to LongTensor type for loss
            volume = volume.to(model_device)
            aseg = aseg.type(torch.cuda.LongTensor)
            aseg = aseg.to(loss_device)

            output = model(volume)          # Feedforward through model

            if len(weights.size()) != 1:
                weights = weights.to(loss_device)

            coords = model.coords           # Extract coordinates of extracted patch in model
            patch_size = model.patch_size   # Extract patch size from model
            
            # Extract aseg and weight patch from same coordinate
            aseg = aseg[..., coords[0]:coords[0] + patch_size[0],
                   coords[1]:coords[1] + patch_size[1],
                   coords[2]:coords[2] + patch_size[2]]

            weights = weights[..., coords[0]:coords[0] + patch_size[0],
                      coords[1]:coords[1] + patch_size[1],
                      coords[2]:coords[2] + patch_size[2]]

            output = output.to(loss_device)
            batch_loss, _, _ = loss(output, aseg, weights) # Calculate loss on loss device

            # Propagate loss backward and optimizer step to change weights
            batch_loss.backward()
            optimizer.step()

            optimizer.zero_grad()   # Reset optimizer grads

            loss_itr += batch_loss.item() # Add sample loss to accumulator variable
            pbar.update(1)

    return loss_itr / len(data_loader)


def train_epoch_multiloss(model, data_loader, loss, optimizer, model_device, loss_device, alpha=0.1):
    """
    Runs one patch based training epoch for (multi loss) MultiResVNet. There is a second loss function calculated on the
    lower resolution but at full volume that is added to the final loss.

    Args:
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.Dataloader): The dataloader for the evaluation
        loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
        model_device (torch.cuda.device): The device the model is stored on.
        loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                         needed.
        alpha (float): factor for low res loss

    Returns:
         float: the total loss over the dataset
    """

    model.train()     # Set model to train mode
    loss.train()      # Set loss to train mode. Needed only to change the values returned by CombinedLoss

    total_loss_itr = 0
    loss_main_itr_ce = 0
    loss_main_itr_dice = 0
    loss_main_itr = 0
    loss_aux_itr_ce = 0
    loss_aux_itr_dice = 0
    loss_aux_itr = 0
    print("Alpha:{}".format(alpha))

    # Adjusting end point of iteration
    itr_end = len(data_loader.dataset)
    optimizer.zero_grad()       # Reset optimizer grads

    with tqdm(total=itr_end) as pbar:
        for batch_id, (volume, aseg, weights) in enumerate(data_loader, 0):
            volume = volume.to(model_device)

            aseg = aseg.to(loss_device)

            # Majority voted downsampling
            aseg_2 = torch.nn.functional.pad(aseg.detach(), pad=(1, 1, 1, 1, 1, 1)).to(loss_device)
            aseg_2 = aseg_2.unfold(2, 4, 2).unfold(3, 4, 2).unfold(4, 4, 2).resize(1, 1, 128, 128, 128, 64)

            aseg_t = torch.zeros(
                (aseg_2.shape[0], aseg_2.shape[1], aseg_2.shape[2], aseg_2.shape[3], aseg_2.shape[4], 79),
                device=loss_device)
            src = torch.zeros((aseg_2.shape[0], aseg_2.shape[1], aseg_2.shape[2], aseg_2.shape[3], aseg_2.shape[4], 79),
                              dtype=torch.float, device=loss_device)
            src[..., :] = 1.0
            aseg_2 = aseg_t.scatter_add(dim=5, index=aseg_2.long(), src=src).argmax(5)


            aseg_2 = aseg_2.to(loss_device).type(torch.cuda.LongTensor)
            aseg = aseg.type(torch.cuda.LongTensor)

            if len(weights.size()) != 1:
                weights = weights.to(loss_device)

            # Feedforward through model to calculate main output (cropped, full res) and auxiliary output (full, lowres)
            output_aux, output_main = model(volume)

            coords = model.coords               # Extract coordinates of extracted patch in model
            patch_size = model.patch_size       # Extract patch size from model

            # Extract patch for main loss
            aseg_1 = aseg[..., coords[0]:coords[0] + patch_size[0],
                     coords[1]:coords[1] + patch_size[1],
                     coords[2]:coords[2] + patch_size[2]]

            weights_1 = weights[..., coords[0]:coords[0] + patch_size[0],
                        coords[1]:coords[1] + patch_size[1],
                        coords[2]:coords[2] + patch_size[2]]

            # Create 2nd weight mask for aux loss
            weights_2 = create_weight_mask(aseg_2.int().cpu().numpy()[0]).reshape(
                (1, patch_size[0], patch_size[1], patch_size[2]))
            weights_2 = torch.from_numpy(weights_2).float().to(loss_device)

            # Move both losses into loss device
            output_main = output_main.to(loss_device)
            output_aux = output_aux.to(loss_device)

            # Calculate main and aux losses individually
            batch_loss_main, main_dice, main_ce = loss(output_main, aseg_1, weights_1)
            batch_loss_aux, aux_dice, aux_ce = loss(output_aux, aseg_2, weights_2)

            # Add losses
            batch_loss = batch_loss_main + (alpha * batch_loss_aux)  # Loss = crop_loss + alpha * lowres_fv_loss
            batch_loss.backward()  # Calculate gradients

            optimizer.step()        # Optimize weights
            optimizer.zero_grad()   # Reset grads

            # Accumulate main and aux losses for logging/reporting
            loss_main_itr_ce += main_ce.item()
            loss_main_itr_dice += main_dice.item()
            loss_main_itr += batch_loss_main.item()

            loss_aux_itr_ce += aux_ce.item()
            loss_aux_itr_dice += aux_dice.item()
            loss_aux_itr += batch_loss_aux.item()

            total_loss_itr += batch_loss.item()
            pbar.update(1)

    print(f"""Epoch Avg Main Loss: {(loss_main_itr / len(data_loader),  
                                loss_main_itr_ce / len(data_loader), 
                                loss_main_itr_dice / len(data_loader))}""")
    print(f"""Epoch Avg Aux Loss: {(loss_aux_itr / len(data_loader), 
                                    loss_aux_itr_ce / len(data_loader), 
                                    loss_aux_itr_dice / len(data_loader))}""")

    return total_loss_itr / len(data_loader)


def train_epoch_quadnet(model, data_loader, loss, optimizer, model_device, device, coords_dict):
    """
        Runs one patch based training epoch for QuadNet. The volumetric cropping is performed in the model and
        not in this function. The main difference from 'train_epoch' function is that the coordinates for the RCVNet
        (or CropNet3D - not tested) is not generated randomly and extracted from coords_dict.

        Args:
            model (torch.nn.Module): The model to be evaluated
            data_loader (torch.utils.Dataloader): The dataloader for the evaluation
            loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
            model_device (torch.cuda.device): The device the model is stored on.
            loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                             needed.
            coords_dict (dict): Set of indices and coordinates. Each coordinate is for a particular quadnet model and is
                                localized to a region of the original volume

        Returns:
            float: the total loss over the dataset
    """

    model.train()
    loss.train()
    loss_itr = 0

    # Adjusting end point of iteration
    itr_end = len(data_loader.dataset)
    optimizer.zero_grad()
    with tqdm(total=itr_end) as pbar:
        for batch_id, (idx, volume, aseg, weights) in enumerate(data_loader, 0):
            volume = volume.to(model_device)

            # Instead of generating random patch coordinates in the model, the coordinates are extracted from the
            # coords_dict for the specific model
            coords = coords_dict[idx.item()]
            patch_size = model.patch_size

            aseg = aseg.type(torch.cuda.LongTensor)
            aseg = aseg.to(device)

            # model coordinates assigned instead of being generated
            model.coords = coords

            output = model(volume)

            # Everything past this point similar to train_epoch(..)
            if len(weights.size()) != 1:
                weights = weights.to(device)

            aseg = aseg[..., coords[0]:coords[0] + patch_size[0],
                   coords[1]:coords[1] + patch_size[1],
                   coords[2]:coords[2] + patch_size[2]]

            weights = weights[..., coords[0]:coords[0] + patch_size[0],
                      coords[1]:coords[1] + patch_size[1],
                      coords[2]:coords[2] + patch_size[2]]

            output = output.to(device)
            batch_loss, _, _ = loss(output, aseg, weights)

            batch_loss.backward()  # This just calculates the gradient and sums it to the existing one.

            optimizer.step()
            optimizer.zero_grad()

            loss_itr += batch_loss.item()
            pbar.update(1)
            # break

    return loss_itr / len(data_loader)


@torch.no_grad()
def valid_epoch(model, data_loader, loss, model_device, device):
    """
        Calculates the total dice loss and classwise dice score for full volume based evaluations. The loss function is
        designed to be an torch.nn.Module object. [CALCULATES A SOFT VERSION OF DICE SCORE. NOT MEANT FOR EVALUATION
        CALCULATIONS. SUITABLE FOR INTERMEDIATE VALIDATION DICE SCORES]

        Args:
            model (torch.nn.Module): The model to be evaluated
            data_loader (torch.utils.Dataloader): The dataloader for the evaluation
            loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
            model_device (torch.cuda.device): The device the model is stored on.
            loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                             needed.

        Returns:
             tuple(float, float): the classwise dice score, The total dice score
    """

    model.eval()
    loss.eval()  # In my loss functions, I inherit from nn.Module. Hence they all have training and non.training
    # modes. In non-training modes they just calculate classwise and total dice 'score'

    with tqdm(total=len(data_loader.dataset)) as pbar:

        total_loss = 0
        total_class_loss = 0

        # Standard full volume evaluation: volume, asegs loaded, output generated and loss calculated
        for batch_id, (volume, asegs, _) in enumerate(data_loader):
            volume = volume.to(model_device)
            asegs = asegs.type(torch.cuda.LongTensor)
            asegs = asegs.to(device)
            output = model(volume)
            output = output.to(device)
            output = F.softmax(output, dim=1)  # Need to do this here in this variant

            batch_class_wise_score, batch_score = loss(output, asegs, 1)

            if total_class_loss is None:
                total_class_loss = batch_class_wise_score.cpu().data.numpy()
            else:
                total_class_loss += batch_class_wise_score.cpu().data.numpy()
            total_loss += batch_score.item()
            pbar.update(1)

    return total_class_loss / len(data_loader), total_loss / len(data_loader)


@torch.no_grad()
def valid_epoch_quadnet(model, data_loader, loss, model_device, loss_device, log_dir, p_len=128, p_step=128,
                        inp_len=256, n_class=79, part_model_path={}):
    """
    Calculates the total dice score and classwise dice score for 3D patch-based evaluations on QuadNet. The loss
    function is designed to be an torch.nn.Module object. [CALCULATES A SOFT VERSION OF DICE SCORE. NOT MEANT FOR
    EVALUATION CALCULATIONS. SUITABLE FOR INTERMEDIATE VALIDATION DICE SCORES]

    Two pools are maintained - current (containing the model saved in the current epoch) and best (the last best version
    of the quad model). The aim is to evaluate the entire quadnet with only a single (current) model changed from the
    best pool - this is the model which is currently being trained.

    If the inclusion of the current model gives a better dice score than the current best dice score, it is replaced in
    the 'best' pool. (This does not happen in this function)

    Args:
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.Dataloader): The dataloader for the evaluation
        loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
        model_device (torch.cuda.device): The device the model is stored on.
        loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                         needed.
        log_dir (string): The absolute path to the directory where the path models are stored
        p_len (int): The side length of the 3D patch
        p_step (int): The stride/step of the 3D patch extraction. Defines the degree of overlap
        inp_len (int): The side length of the input volume
        n_class (int): The number of classes in the output segmentation
        part_model_path (dict): The path for the temporary (current) epoch file. Any quadnet models with a path in this
                    dictionary will be loaded from this path instead of their current "best" version.

    Returns:
         tuple(float, float): the classwise dice score, The total dice score
    """
    model.eval()
    loss.eval()  # In my loss functions, I inherit from nn.Module. Hence they all have training and non.training
    # modes. In non-training modes they just calculate classwise and total dice 'score'

    # n_patches = inp_len / ((p_len) if p_len >= p_step else p_len - p_step)

    patch_s = [p for p in range(0, inp_len - p_len + 1, p_step)]
    patch_list = list(product(patch_s, patch_s, patch_s))

    net_coords = assign_nets_to_coords(
        input_shape=(inp_len, inp_len, inp_len),
        patch_size=(p_len, p_len, p_len),
        coords_list=patch_list
    )

    # print(net_coords)

    with tqdm(total=len(data_loader.dataset)) as pbar:

        total_loss = 0
        total_class_loss = 0

        # QuadNet eval: volume, asegs loaded, the current quad model replaced in main pool and dice score calculated
        for batch_id, (volume, asegs, _) in enumerate(data_loader):

            out = torch.zeros((1, n_class, inp_len, inp_len, inp_len)).to(model_device)

            volume = volume.to(model_device)

            for idx, k in enumerate(net_coords.keys()):

                # If the index is present in part_model_path as a key then use that path instead of path from main pool
                if (idx + 1) in part_model_path.keys():
                    part_model_save_path = os.path.join(log_dir, 'tmp', f"quad_model_{idx + 1}")
                else:
                    part_model_save_path = os.path.join(log_dir, f"quad_model_{idx + 1}")
                model.load_state_dict(torch.load(part_model_save_path))

                quad_data = net_coords[k]
                model_coordinates = [i for _, i in quad_data]

                # If patch and model receptive fields overlap, obtain a segmentation output for the patch from model
                # and add to 'out'
                for x, y, z in model_coordinates:
                    outs_patch = model(volume[..., x:x + p_len, y:y + p_len, z:z + p_len])
                    out[..., x:x + p_len, y:y + p_len, z:z + p_len] += outs_patch

            del outs_patch
            out = out.to(loss_device)
            out = F.softmax(out, dim=1)  # Need to do this here in this variant

            asegs = asegs.type(torch.cuda.LongTensor)
            asegs = asegs.to(loss_device)

            # Calculate loss
            batch_class_wise_score, batch_score = loss(out, asegs, 1)
            # print(batch_class_wise_score, batch_score)

            del asegs, out

            if total_class_loss is None:
                total_class_loss = batch_class_wise_score.cpu().data.numpy()
            else:
                total_class_loss += batch_class_wise_score.cpu().data.numpy()
            total_loss += batch_score.item()
            pbar.update(1)

    return total_class_loss / len(data_loader), total_loss / len(data_loader)


@torch.no_grad()
def eval_epoch(model, data_loader, loss, model_device, loss_device):
    """
    Calculates the total dice score and classwise dice score for full volume based evaluations. The loss function is
    designed to be an torch.nn.Module object
    
    Args:
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.Dataloader): The dataloader for the evaluation
        loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
        model_device (torch.cuda.device): The device the model is stored on.
        loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                         needed.
    
    Returns:
         tuple(float, float): the classwise dice score, The total dice score
    """

    model.eval()
    loss.eval()
    with tqdm(total=len(data_loader.dataset)) as pbar:

        total_loss = 0
        total_class_loss = 0

        # Standard full volume evaluation: volume, asegs loaded, output generated and loss calculated
        for batch_id, (volume, asegs, _) in enumerate(data_loader):
            volume = volume.to(model_device)
            asegs = asegs.to(loss_device)
            asegs = asegs.type(torch.cuda.LongTensor)

            out = model(volume)
            encoded_out = out.detach() * 0
            # out = out.cpu().numpy()

            # Soft logits converted to hard values. NumPy used for Torch argmax instability
            out = out.cpu().argmax(dim=1, keepdim=True)
            encoded_out.scatter_(1, out.to(model_device), 1)

            del out
            encoded_out = encoded_out.to(loss_device)

            # Dice score calculated
            batch_class_wise_score, batch_score = loss(encoded_out, asegs, 1)

            if total_class_loss is None:
                total_class_loss = batch_class_wise_score.cpu().data.numpy()
            else:
                total_class_loss += batch_class_wise_score.cpu().data.numpy()
            total_loss += batch_score
            pbar.update(1)

    return total_class_loss / len(data_loader), total_loss / len(data_loader)


@torch.no_grad()
def eval_epoch_patched(model, data_loader, loss, model_device, loss_device, p_len=128, p_step=32,
                       inp_len=256, n_class=79, add_full_vol=False):
    """
    Calculates the total dice score and classwise dice score for 3D patch-based evaluations. The loss function is
    designed to be an torch.nn.Module object.
    
    Args:
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.Dataloader): The dataloader for the evaluation
        loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
        model_device (torch.cuda.device): The device the model is stored on.
        loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                         needed.
        p_len (int): The side length of the 3D patch
        p_step (int): The stride/step of the 3D patch extraction. Defines the degree of overlap
        inp_len (int): The side length of the input volume
        n_class (int): The number of classes in the output segmentation
        add_full_vol (bool): Set 'True' to add a full volume evaluation to the patch based evaluations 
    
    Returns:
         tuple(float, float): the classwise dice score, The total dice score
    """

    model.eval()
    loss.eval()

    print(p_len, p_step)
    # n_patches = inp_len/((p_len) if p_len>p_step else p_len-p_step)

    # List of patch coordinates generated
    patch_s = [p for p in range(0, inp_len - p_len + 1, p_step)]
    patch_list = list(product(patch_s, patch_s, patch_s))

    with tqdm(total=len(data_loader.dataset)) as pbar:

        total_loss = 0
        total_class_loss = 0

        # Overlapped patch eval: volume, asegs loaded. Sliding window used to generate patches, segmentations genereated
        # from which are accumulated and converted to hard scores. Following this, dice score is calculated
        for batch_id, (volume, asegs, _) in enumerate(data_loader):

            out = torch.zeros((1, n_class, inp_len, inp_len, inp_len)).to(model_device)

            volume = volume.to(model_device)

            # Full vol is added if flag is set
            if add_full_vol:
                out_vol = model(volume)
                out_vol = torch.nn.functional.softmax(out_vol, dim=1)
                out += out_vol

            # Patches extracted by coordinates, segmentations generated and accumulated
            for idx, (x, y, z) in enumerate(patch_list):
                outs_patch = model(volume[..., x:x + p_len, y:y + p_len, z:z + p_len])
                outs_patch = torch.nn.functional.softmax(outs_patch, dim=1)
                out[..., x:x + p_len, y:y + p_len, z:z + p_len] += outs_patch

            del volume

            # Soft scores converted to hard values. Numpy for instability of torch.argmax
            encoded_out = out.detach() * 0
            out = out.cpu().argmax(dim=1, keepdim=True)

            encoded_out.scatter_(1, out.to(model_device), 1)

            del out
            encoded_out = encoded_out.to(loss_device)
            asegs = asegs.type(torch.cuda.LongTensor)
            asegs = asegs.to(loss_device)

            # Loss calculated
            batch_class_wise_score, batch_score = loss(encoded_out, asegs, 1)
            del asegs

            if total_class_loss is None:
                total_class_loss = batch_class_wise_score.cpu().data.numpy()
            else:
                total_class_loss += batch_class_wise_score.cpu().data.numpy()
            total_loss += batch_score.item()
            pbar.update(1)

    return total_class_loss / len(data_loader), total_loss / len(data_loader)


@torch.no_grad()
def eval_epoch_quadnet(model, data_loader, loss, model_device, loss_device, log_dir, p_len=128, p_step=32,
                       inp_len=256, n_class=79, base_pretrained_path=None):
    """
    Calculates the total dice score and classwise dice score for 3D patch-based evaluations on QuadNet. The loss 
    function is designed to be an torch.nn.Module object.
    
    Args:
        model (torch.nn.Module): The model to be evaluated
        data_loader (torch.utils.Dataloader): The dataloader for the evaluation
        loss (torch.nn.Module): The loss function. Always use a CombinedLoss object.
        model_device (torch.cuda.device): The device the model is stored on.
        loss_device (torch.cuda.device): The device the loss calculation happens on. Keep same as model_device if not
                                         needed.
        log_dir (string): The absolute path to the directory where the path models are stored
        p_len (int): The side length of the 3D patch
        p_step (int): The stride/step of the 3D patch extraction. Defines the degree of overlap
        inp_len (int): The side length of the input volume
        n_class (int): The number of classes in the output segmentation
        base_pretrained_path (string): The absolute path to the base model which served as the pretrained model before
                                       QuadNet finetuning. Providing this will 'add' a full volume evaluation on the 
                                       base model to the QuadNet logits
    
    Returns:
         tuple(float, float): the classwise dice score, The total dice score
    """
    loss.eval()
    print("Patch Length = {}, Patch Step = {}".format(p_len, p_step))
    # n_patches = inp_len / ((p_len) if p_len >= p_step else p_len - p_step)

    patch_s = [p for p in range(0, inp_len - p_len + 1, p_step)]
    patch_list = list(product(patch_s, patch_s, patch_s))

    net_coords = assign_nets_to_coords(
        input_shape=(inp_len, inp_len, inp_len),
        patch_size=(p_len, p_len, p_len),
        coords_list=patch_list
    )

    # Similar to valid epoch except for the conversion to hard scores before dice score calculation
    with tqdm(total=len(data_loader.dataset)) as pbar:

        total_loss = 0
        total_class_loss = 0

        for batch_id, (volume, asegs, _) in enumerate(data_loader):
            model.eval()

            out = torch.zeros((1, n_class, inp_len, inp_len, inp_len)).to(model_device)

            volume = volume.to(model_device)

            if base_pretrained_path is not None:
                model.load_state_dict(torch.load(base_pretrained_path))
                out_vol = model(volume)
                out_vol = torch.nn.functional.softmax(out_vol, dim=1)
                out += out_vol

            for idx, k in enumerate(net_coords.keys()):

                part_model_save_path = os.path.join(log_dir, f"quad_model_{idx + 1}")
                model.load_state_dict(torch.load(part_model_save_path))

                quad_data = net_coords[k]
                model_coordinates = [i for _, i in quad_data]

                for x, y, z in model_coordinates:
                    outs_patch = model(volume[..., x:x + p_len, y:y + p_len, z:z + p_len])
                    outs_patch = torch.nn.functional.softmax(outs_patch, dim=1)
                    out[..., x:x + p_len, y:y + p_len, z:z + p_len] += outs_patch
                    del outs_patch

            del volume
            # Hard scores calculated
            encoded_out = out.detach() * 0
            out = out.cpu().argmax(dim=1, keepdim=True)
            encoded_out.scatter_(1, out.to(model_device), 1)

            del out
            encoded_out = encoded_out.to(loss_device)
            asegs = asegs.type(torch.cuda.LongTensor)
            asegs = asegs.to(loss_device)

            batch_class_wise_score, batch_score = loss(encoded_out, asegs, 1)

            del asegs

            if total_class_loss is None:
                total_class_loss = batch_class_wise_score.cpu().data.numpy()
            else:
                total_class_loss += batch_class_wise_score.cpu().data.numpy()
            total_loss += batch_score.item()
            pbar.update(1)

    return total_class_loss / len(data_loader), total_loss / len(data_loader)
