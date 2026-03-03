"""
SuperDeepFool Attack Implementation

Enhanced version of DeepFool with improved convergence and robustness.
"""
import copy
from typing import Optional

import numpy as np
import torch as torch
from torch import Tensor, nn
from torch.autograd import Variable


def superdeepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50, 
                  alpha=1.5, adaptive_overshoot=True):
    """
    SuperDeepFool: Enhanced DeepFool with adaptive step size and improved convergence.
    
    Args:
        image: Image tensor of size CxHxW
        net: network (input: images, output: logits before softmax)
        num_classes: number of classes to test against (default = 10)
        overshoot: initial overshoot parameter (default = 0.02)
        max_iter: maximum number of iterations (default = 50)
        alpha: step size multiplier for faster convergence (default = 1.5)
        adaptive_overshoot: whether to adaptively adjust overshoot (default = True)
        
    Returns:
        r_tot: minimal perturbation that fools the classifier
        loop_i: number of iterations required
        label: original predicted label
        k_i: new predicted label after attack
        pert_image: perturbed image
    """
    device = image.device

    # Forward pass to get initial prediction
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    current_overshoot = overshoot

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label

    # Track convergence for adaptive overshoot
    prev_pert = np.inf

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grad = None

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # Compute w_k and f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            # Enhanced perturbation calculation with numerical stability
            pert_k = abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-8)

            # Determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # Adaptive overshoot adjustment
        if adaptive_overshoot and loop_i > 0:
            if pert < prev_pert * 0.8:  # Good progress
                current_overshoot = min(current_overshoot * 1.1, 0.1)
            elif pert > prev_pert * 0.95:  # Slow progress
                current_overshoot = max(current_overshoot * 0.9, 0.001)
        
        prev_pert = pert

        # Compute r_i with enhanced step size (alpha multiplier)
        r_i = alpha * (pert + 1e-4) * w / (np.linalg.norm(w) + 1e-8)
        r_tot = np.float32(r_tot + r_i)

        # Apply perturbation
        pert_image = image + (1 + current_overshoot) * torch.from_numpy(r_tot).to(device)

        # Forward pass with perturbed image
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1 + current_overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image


def superdeepfool_attack(model: nn.Module,
                         inputs: Tensor,
                         labels: Tensor,
                         targets: Optional[Tensor] = None,
                         targeted: bool = False,
                         num_classes: int = 10,
                         overshoot: float = 0.02,
                         num_steps: int = 50,
                         alpha: float = 1.5,
                         adaptive_overshoot: bool = True,
                         **kwargs) -> Tensor:
    """
    SuperDeepFool attack wrapper for batch processing.
    
    Args:
        model: Target neural network
        inputs: Input images (batch)
        labels: True labels
        targets: Target labels (for targeted attacks, not used)
        targeted: Whether attack is targeted (not supported)
        num_classes: Number of classes to consider
        overshoot: Initial overshoot parameter
        num_steps: Maximum iterations per sample
        alpha: Step size multiplier
        adaptive_overshoot: Whether to use adaptive overshoot
        
    Returns:
        Adversarial examples
    """
    adv_inputs = []
    
    model.eval()
    
    for i in range(len(inputs)):
        input_single = inputs[i]
        
        # Run SuperDeepFool on single image
        r, iter_count, orig_label, new_label, adv_img = superdeepfool(
            input_single, 
            model,
            num_classes=num_classes,
            overshoot=overshoot,
            max_iter=num_steps,
            alpha=alpha,
            adaptive_overshoot=adaptive_overshoot
        )
        
        # Remove batch dimension if present (should be [C, H, W])
        while adv_img.dim() > 3:
            adv_img = adv_img.squeeze(0)
        
        # Append directly like deepfool does
        adv_inputs.append(adv_img)
    
    # Stack to create batch dimension [N, C, H, W]
    return torch.stack(adv_inputs)
