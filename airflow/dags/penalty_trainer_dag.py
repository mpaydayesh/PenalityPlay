"""
Airflow DAG for training and evaluating penalty shootout RL agents.
"""
from datetime import datetime, timedelta
import os
import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'penalty_shootout_trainer',
    default_args=default_args,
    description='Train and evaluate penalty shootout RL agents',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    tags=['rl', 'penalty_shootout'],
    catchup=False,
)

def load_config():
    """Load configuration for the training job."""
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/agent_params.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_agents(**kwargs):
    """Train the striker and goalkeeper agents."""
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from train import train_agents as train_func
    
    # Get configuration
    config = load_config()
    
    # Update config with any parameters from the DAG
    ti = kwargs['ti']
    run_id = kwargs['run_id']
    
    # Set experiment name with run ID
    config['experiment_name'] = f"airflow_{run_id}"
    
    # Train the agents
    train_func(config)
    
    # Push the experiment name to XCom for downstream tasks
    ti.xcom_push(key='experiment_name', value=config['experiment_name'])

def evaluate_agents(**kwargs):
    """Evaluate the trained agents."""
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from train import evaluate_agents as evaluate_func
    from src.envs.penalty_env import PenaltyShootoutEnv
    from src.agents.striker import StrikerAgent
    from src.agents.goalkeeper import GoalkeeperAgent
    from src.utils.logger import Logger
    
    # Get the experiment name from the training task
    ti = kwargs['ti']
    experiment_name = ti.xcom_pull(task_ids='train_agents', key='experiment_name')
    
    if not experiment_name:
        raise ValueError("No experiment name found. Did training complete successfully?")
    
    # Set up logging
    logger = Logger(log_dir="logs", experiment_name=experiment_name)
    
    # Initialize environment and agents
    env = PenaltyShootoutEnv()
    
    # Load the trained models
    striker = StrikerAgent(env)
    striker = logger.load_model(striker, "striker_final")
    
    goalkeeper = GoalkeeperAgent(env)
    goalkeeper = logger.load_model(goalkeeper, "goalkeeper_final")
    
    # Evaluate the agents
    evaluate_func(striker, goalkeeper, env, logger, timestep=0, num_episodes=100)
    
    # Generate and save plots
    logger.plot_metrics(show=False)

def generate_report(**kwargs):
    """Generate a report of the training and evaluation results."""
    import os
    import json
    from datetime import datetime
    from jinja2 import Environment, FileSystemLoader
    
    # Get the experiment name from the training task
    ti = kwargs['ti']
    experiment_name = ti.xcom_pull(task_ids='train_agents', key='experiment_name')
    
    if not experiment_name:
        raise ValueError("No experiment name found. Did training complete successfully?")
    
    # Set up paths
    log_dir = os.path.join('logs', experiment_name)
    metrics_file = os.path.join(log_dir, 'metrics', 'eval_metrics.json')
    report_dir = os.path.join(log_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Prepare report data
    report_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics[-1] if metrics else {},
        'plots': {
            'training_reward': os.path.join('plots', 'train_episode_reward.png'),
            'evaluation_reward': os.path.join('plots', 'eval_mean_reward.png'),
        }
    }
    
    # Render report template
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('report_template.html')
    
    # Write report
    report_path = os.path.join(report_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(template.render(**report_data))
    
    print(f"Report generated at: {report_path}")

# Define tasks
train_task = PythonOperator(
    task_id='train_agents',
    python_callable=train_agents,
    provide_context=True,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_agents',
    python_callable=evaluate_agents,
    provide_context=True,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
train_task >> evaluate_task >> report_task

# Task to install dependencies if needed
install_deps = BashOperator(
    task_id='install_dependencies',
    bash_command='pip install -r requirements.txt',
    dag=dag,
)

# Set up the full workflow
install_deps >> train_task
