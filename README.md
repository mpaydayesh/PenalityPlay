# Dual-Agent Competitive Football Penalty Shootout

A reinforcement learning project featuring competing striker and goalkeeper agents in a football penalty shootout scenario, powered by RL and Stable Baselines3.

## 🚀 Features

- **Dual-Agent Training**: Simultaneously train striker and goalkeeper agents
- **Custom Gym Environment**: Realistic penalty shootout simulation
- **Modular Architecture**: Easy to extend and modify
- **Comprehensive Logging**: Track training progress and metrics
- **Visualization Tools**: Built-in plotting utilities
- **CLI Interface**: Simple command-line tools for training and evaluation

## 📦 Project Structure

```
.
├── configs/                # Configuration files
│   ├── training_config.yaml  # Training hyperparameters
│   └── agent_params.yaml    # Agent-specific parameters
├── data/                   # Data storage
│   ├── logs/               # Training logs
│   └── models/             # Saved models
├── src/                    # Source code
│   ├── agents/             # RL agent implementations
│   │   ├── base_agent.py   # Base agent class
│   │   ├── striker.py     # Striker agent implementation
│   │   └── goalkeeper.py  # Goalkeeper agent implementation
│   ├── envs/               # Custom Gym environments
│   │   └── penalty_env.py  # Penalty shootout environment
│   ├── match/              # Match orchestration
│   ├── genai/              # AI/ML components
│   ├── utils/              # Helper functions
│   │   ├── logger.py      # Logging utilities
│   │   └── visualize.py   # Plotting functions
│   └── cli.py             # Command-line interface
├── tests/                  # Unit and integration tests
├── train.py               # Main training script
├── evaluate.py            # Model evaluation script
├── setup.py               # Package configuration
└── requirements.txt       # Python dependencies
```

## 🛠️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/penalty-shootout-rl.git
   cd penalty-shootout-rl
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .  # Install in development mode
   # Or install requirements directly:
   # pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import gymnasium; import torch; print('Libraries loaded successfully!')"
   ```

## 🚦 Quick Start

### Training

Train both striker and goalkeeper agents:

```bash
# Basic training with default config
python train.py

# Custom training configuration
python train.py --config configs/training_config.yaml --num-episodes 2000
```

### Evaluation

Evaluate trained models:

```bash
python evaluate.py \
  --striker models/striker_episode_1000 \
  --goalkeeper models/goalkeeper_episode_1000 \
  --episodes 20
```

### Using the CLI

The project includes a command-line interface for common tasks:

```bash
# Train agents
python -m src.cli train --num-episodes 1000

# Evaluate models
python -m src.cli eval --striker models/striker_final --goalkeeper models/goalkeeper_final
```

## 📊 Visualization

Plot training metrics:

```python
from utils.visualize import plot_training_metrics

# Plot rewards and episode lengths
plot_training_metrics('logs/')
```

## 🤖 Customization

### Environment

Modify `src/envs/penalty_env.py` to change:
- State and action spaces
- Reward function
- Game dynamics

### Agents

Adjust agent implementations in `src/agents/`:
- Network architecture
- Learning algorithm
- Exploration strategies

### Configuration

Update `configs/training_config.yaml` to customize:
- Training hyperparameters
- Model architectures
- Logging and saving options

## 📈 Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=logs/
```

Then open `http://localhost:6006` in your browser.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 🙏 Acknowledgments

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for the RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) for the environment interface
- [PyTorch](https://pytorch.org/) for deep learning
- Run evaluation: `python -m src.evaluate`
- Start web interface: `streamlit run streamlit_app/app.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
