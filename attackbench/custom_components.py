from functools import wraps

import torch



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
        
    Example::

        def my_custom_pgd(model, inputs, labels, eps=0.1, steps=10):
            adv_inputs = inputs.clone()
            for i in range(steps):
                adv_inputs.requires_grad_(True)
                outputs = model(adv_inputs)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_inputs)[0]
                adv_inputs = adv_inputs + eps / steps * grad.sign()
                adv_inputs = torch.clamp(adv_inputs, 0, 1)
            return adv_inputs

        custom_attack = create_custom_attack(my_custom_pgd, attack_name="MyPGD")

        model = attackbench.get_model('carmon_2019')
        dataset = attackbench.get_loader('cifar10', num_samples=1000)
        results = attackbench.run_attack(model, dataset, custom_attack, 'linf')
    """
    @wraps(attack_function)
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
