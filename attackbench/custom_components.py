from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
import os



def create_custom_attack(
    attack_function: callable,
    validate_inputs: bool = True,
    enforce_constraints: bool = True,
    attack_name: str = None
) -> callable:
    """
    Advanced wrapper for user's custom attacks.
    
    Args:
        attack_function: User's attack function
        validate_inputs: Whether to validate inputs/outputs
        enforce_constraints: Whether to enforce constraints (clamp 0-1, device, etc.)
        attack_name: Attack name for logging
        
    Returns:
        Attack function compatible with AttackBench
        
    Example:
        def my_custom_pgd(model, inputs, labels, eps=0.1, steps=10):
            # User's custom implementation
            adv_inputs = inputs.clone()
            for i in range(steps):
                adv_inputs.requires_grad_(True)
                outputs = model(adv_inputs)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_inputs)[0]
                adv_inputs = adv_inputs + eps/steps * grad.sign()
                adv_inputs = torch.clamp(adv_inputs, 0, 1)
            return adv_inputs
        
        custom_attack = create_custom_attack(my_custom_pgd, attack_name="MyPGD")
        
        results = attackbench.run_attack(
            model='Carmon2019Unlabeled',
            dataset='cifar10',
            attack=custom_attack,
            threat_model='linf'
        )
    """
    
    def wrapped_attack(model, inputs, labels, **kwargs):
        """Internal wrapper that handles custom attack execution"""
        
        # Input validation
        if validate_inputs:
            _validate_attack_inputs(model, inputs, labels)
        
        # Input preparation (type conversion, device, etc.)
        inputs, labels = _prepare_inputs(inputs, labels, model.device if hasattr(model, 'device') else None)
        
        # Attack logging if available
        attack_display_name = attack_name or getattr(attack_function, '__name__', 'custom_attack')
        
        try:
            # Execute user's attack
            adv_examples = attack_function(model, inputs, labels, **kwargs)
            
            # Post-processing and output validation
            adv_examples = _process_attack_output(
                adv_examples, inputs, 
                enforce_constraints=enforce_constraints,
                attack_name=attack_display_name
            )
            
            return adv_examples
            
        except Exception as e:
            raise RuntimeError(f"Error in custom attack '{attack_display_name}': {str(e)}") from e
    
    # Add metadata to wrapper
    wrapped_attack.__name__ = f"custom_{attack_name or 'attack'}"
    wrapped_attack.__custom_attack__ = True
    wrapped_attack.original_function = attack_function
    
    return wrapped_attack


def _validate_attack_inputs(model, inputs, labels):
    """Validate attack inputs"""
    if not hasattr(model, 'forward'):
        raise ValueError("Model must have a 'forward' method")
    
    if not torch.is_tensor(inputs):
        raise TypeError("Inputs must be a torch.Tensor")
        
    if not torch.is_tensor(labels):
        raise TypeError("Labels must be a torch.Tensor")
    
    if len(inputs) != len(labels):
        raise ValueError(f"Batch size mismatch: inputs={len(inputs)}, labels={len(labels)}")
    
    if inputs.dim() < 2:
        raise ValueError(f"Inputs must have at least 2 dimensions, got {inputs.dim()}")


def _prepare_inputs(inputs, labels, target_device=None):
    """Prepare inputs for attack (conversions, device, etc.)"""
    # Convert to tensor if necessary
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Move to correct device
    if target_device is not None:
        inputs = inputs.to(target_device)
        labels = labels.to(target_device)
    
    # Ensure inputs is float and labels is long
    inputs = inputs.float()
    labels = labels.long()
    
    return inputs, labels


def _process_attack_output(adv_examples, original_inputs, enforce_constraints=True, attack_name="custom"):
    """Post-process attack output"""
    # Convert to tensor if necessary
    if not isinstance(adv_examples, torch.Tensor):
        adv_examples = torch.tensor(adv_examples)
    
    # Check that shape is correct
    if adv_examples.shape != original_inputs.shape:
        raise ValueError(
            f"Attack '{attack_name}' output shape mismatch: "
            f"expected {original_inputs.shape}, got {adv_examples.shape}"
        )
    
    # Apply constraints if requested
    if enforce_constraints:
        # Clamp between 0 and 1 (assuming normalized images)
        adv_examples = torch.clamp(adv_examples, 0.0, 1.0)
        
        # Ensure same device and dtype as inputs
        adv_examples = adv_examples.to(original_inputs.device, original_inputs.dtype)
    
    return adv_examples


# Helper function for more complex attacks
def create_iterative_attack(
    gradient_step_fn: callable,
    num_steps: int = 10,
    eps: float = 8/255,
    step_size: float = None,
    random_start: bool = True,
    attack_name: str = None
) -> callable:
    """
    Helper to create custom iterative attacks.
    
    Args:
        gradient_step_fn: Function that computes a single gradient step
                         Signature: fn(model, inputs, labels, **kwargs) -> gradient
        num_steps: Number of iterations
        eps: Adversarial budget
        step_size: Step size (default: eps/num_steps)
        random_start: Whether to start from random point
        attack_name: Attack name
        
    Example:
        def my_gradient_step(model, inputs, labels, **kwargs):
            inputs.requires_grad_(True)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, inputs)[0]
            return grad.sign()  # FGSM-style gradient
        
        my_attack = create_iterative_attack(
            gradient_step_fn=my_gradient_step,
            num_steps=20,
            eps=8/255,
            attack_name="MyIterativeFGSM"
        )
    """
    
    if step_size is None:
        step_size = eps / num_steps
    
    def iterative_attack(model, inputs, labels, **kwargs):
        """Generic implementation of iterative attack"""
        
        # EXTRACT NORM FROM KWARGS (passed by run_attack)
        norm = kwargs.pop('norm', 'linf')  # Default linf if not specified
        
        adv_inputs = inputs.clone().detach()
        
        # Random start based on norm
        if random_start:
            adv_inputs = _apply_random_start(adv_inputs, eps, norm)
        
        # Attack iterations
        for step in range(num_steps):
            # Compute gradient using user's function
            gradient = gradient_step_fn(model, adv_inputs.clone(), labels, **kwargs)
            
            # Apply step
            adv_inputs = adv_inputs + step_size * gradient
            
            # PROJECTION BASED ON NORM FROM KWARGS
            adv_inputs = _project_to_norm_ball(adv_inputs, inputs, eps, norm)
        
        return adv_inputs
    
    return create_custom_attack(iterative_attack, attack_name=attack_name)


def _apply_random_start(inputs, eps, norm):
    """Apply random start based on norm"""
    if norm == 'linf':
        noise = torch.empty_like(inputs).uniform_(-eps, eps)
    elif norm == 'l2':
        noise = torch.randn_like(inputs)
        noise = noise / torch.norm(noise.flatten(1), p=2, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        noise = noise * eps * torch.rand(inputs.shape[0], 1, 1, 1, device=inputs.device)
    else:
        noise = torch.randn_like(inputs) * (eps / 10)  # Conservative for other norms
    
    return torch.clamp(inputs + noise, 0, 1)


def _project_to_norm_ball(adv_inputs, original_inputs, eps, norm):
    """Project adversarial inputs to the budget specified by norm"""
    delta = adv_inputs - original_inputs
    
    if norm == 'linf':
        # L-infinity: clamp each component
        delta = torch.clamp(delta, -eps, eps)
    
    elif norm == 'l2':
        # L2: normalize if necessary
        batch_size = delta.shape[0]
        delta_flat = delta.flatten(1)  # [B, H*W*C]
        l2_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)  # [B, 1]
        
        # If norm > eps, normalize
        mask = l2_norm > eps
        delta_flat[mask.squeeze()] = delta_flat[mask.squeeze()] / l2_norm[mask] * eps
        delta = delta_flat.view_as(delta)
    
    elif norm == 'l1':
        # L1: more complex projection
        batch_size = delta.shape[0]
        delta_flat = delta.flatten(1)
        l1_norm = torch.norm(delta_flat, p=1, dim=1, keepdim=True)
        
        mask = l1_norm > eps
        if mask.any():
            for i in range(batch_size):
                if mask[i]:
                    lambda_val = _binary_search_l1_projection(delta_flat[i], eps)
                    delta_flat[i] = torch.sign(delta_flat[i]) * torch.clamp(torch.abs(delta_flat[i]) - lambda_val, min=0)
            delta = delta_flat.view_as(delta)
    
    elif norm == 'l0':
        # L0: keep only the k most important pixels
        k = int(eps)
        delta_flat = delta.flatten(1)
        delta_abs = torch.abs(delta_flat)
        
        for i in range(delta.shape[0]):
            if k < delta_flat.shape[1]:
                _, top_k_indices = torch.topk(delta_abs[i], k)
                mask_i = torch.zeros_like(delta_flat[i])
                mask_i[top_k_indices] = 1
                delta_flat[i] = delta_flat[i] * mask_i
        
        delta = delta_flat.view_as(delta)
    
    else:
        raise ValueError(f"Unsupported norm: {norm}. Use 'linf', 'l2', 'l1', or 'l0'")
    
    # Final clamp to maintain valid pixels
    projected = torch.clamp(original_inputs + delta, 0, 1)
    return projected


def _binary_search_l1_projection(delta, eps, max_iter=100, tol=1e-6):
    """Binary search to find optimal lambda for L1 projection"""
    lambda_low = 0.0
    lambda_high = torch.max(torch.abs(delta)).item()
    
    for _ in range(max_iter):
        lambda_mid = (lambda_low + lambda_high) / 2
        projected = torch.sign(delta) * torch.clamp(torch.abs(delta) - lambda_mid, min=0)
        l1_norm = torch.norm(projected, p=1).item()
        
        if abs(l1_norm - eps) < tol:
            break
        elif l1_norm > eps:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid
    
    return lambda_mid


def create_custom_model(model: torch.nn.Module):
    """Wrapper for custom models"""
    from .models.benchmodel_wrapper import BenchModel
    return BenchModel(model)