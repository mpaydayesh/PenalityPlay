"""
Command-line interface for the Penalty Shootout RL project.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def train_cli():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train penalty shootout agents')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                      help='Path to training configuration file')
    parser.add_argument('--num-episodes', type=int, help='Number of training episodes')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from train import train
    
    # Update config with command line arguments
    config_override = {}
    if args.num_episodes:
        config_override['training'] = {'num_episodes': args.num_episodes}
    
    train(
        config_path=args.config,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        config_override=config_override
    )

def eval_cli():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate penalty shootout agents')
    parser.add_argument('--striker', type=str, required=True, 
                      help='Path to striker model')
    parser.add_argument('--goalkeeper', type=str, required=True, 
                      help='Path to goalkeeper model')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment during evaluation')
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from evaluate import evaluate_models
    
    results = evaluate_models(
        striker_path=args.striker,
        goalkeeper_path=args.goalkeeper,
        num_episodes=args.episodes,
        render=args.render
    )
    
    return results

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Penalty Shootout RL')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train agents')
    train_parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    train_parser.add_argument('--num-episodes', type=int)
    train_parser.add_argument('--log-dir', type=str, default='logs')
    train_parser.add_argument('--save-dir', type=str, default='models')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate agents')
    eval_parser.add_argument('--striker', type=str, required=True)
    eval_parser.add_argument('--goalkeeper', type=str, required=True)
    eval_parser.add_argument('--episodes', type=int, default=10)
    eval_parser.add_argument('--render', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_cli()
    elif args.command == 'eval':
        eval_cli()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
