import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import torch
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances

from .attacks.ingredient import get_attack
from .datasets.ingredient import get_loader
from .models.ingredient import get_model
from .utils import run_attack, set_seed


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Attack Evaluation Script')
    
    # General configuration
    parser.add_argument('--cpu', action='store_true', 
                       help='Force experiment to run on CPU')
    parser.add_argument('--save_adv', action='store_true',
                       help='Save the inputs and perturbed inputs')
    parser.add_argument('--cudnn_flag', choices=['deterministic', 'benchmark'], 
                       default='deterministic',
                       help='Choose between "deterministic" and "benchmark"')
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       help='Model configuration name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    
    # Attack configuration
    parser.add_argument('--attack', type=str, required=True,
                       help='Attack configuration name')
    parser.add_argument('--threat_model', type=str, required=True,
                       help='Threat model (e.g., l2, linf, l0, l1)')
    
    # Dataset configuration
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for data loading')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def create_output_directory(args):
    """Create output directory structure based on configuration"""
    subdirs = [args.dataset, args.threat_model, args.model, 
               f'batch_size_{args.batch_size}', args.attack]
    output_path = Path(args.output_dir).joinpath(*subdirs)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


metrics = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    args = parse_arguments()
    logger = setup_logging(args.debug)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    setattr(torch.backends.cudnn, args.cudnn_flag, True)

    set_seed(args.seed)
    logger.info(f'Running experiments with seed {args.seed}')

    # Create output directory
    save_dir = create_output_directory(args) if args.save_adv else None

    # Load components
    loader = get_loader(dataset=args.dataset, batch_size=args.batch_size)
    attack = get_attack(attack_name=args.attack, threat_model=args.threat_model)
    model = get_model(model_name=args.model, dataset=args.dataset)
    model.to(device)

    if len(loader) == 0:  # end experiment if there are no inputs to attack
        logger.warning("No inputs to attack, ending experiment")
        return

    logger.info(f"Running attack {args.attack} on model {args.model} with dataset {args.dataset}")
    logger.info(f"Threat model: {args.threat_model}, Batch size: {args.batch_size}")

    attack_data = run_attack(
        model=model, 
        loader=loader, 
        attack=attack, 
        metrics=metrics, 
        threat_model=args.threat_model,
        return_adv=args.save_adv and save_dir is not None, 
        debug=args.debug
    )

    # Save adversarial examples if requested
    if args.save_adv and save_dir is not None:
        torch.save(attack_data, save_dir / 'attack_data.pt')
        logger.info(f"Saved attack data to {save_dir / 'attack_data.pt'}")

    # Save results
    if save_dir:
        # Remove large data before saving results
        results = attack_data.copy()
        if 'inputs' in results:
            del results['inputs'], results['adv_inputs']
        
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {save_dir / 'results.json'}")

    if args.debug:
        logger.debug("Attack results:")
        pprint(attack_data)

    return attack_data


if __name__ == '__main__':
    main()

