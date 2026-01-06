from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

API_BASE="http://localhost:8000"

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="lemonade_ml_cycle_bash",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 0 * * *",  # daily
    catchup=False,
) as dag:

    simulate_gt = BashOperator(
        task_id="simulate_ground_truth",
        bash_command=f"curl -X POST {API_BASE}/cycle/simulate-ground-truth"
    )

    create_dataset = BashOperator(
        task_id="create_latest_dataset",
        bash_command=f"curl -X POST {API_BASE}/cycle/create-dataset"
    )

    retrain = BashOperator(
        task_id="retrain_model",
        bash_command=f"curl -X POST {API_BASE}/cycle/retrain"
    )

    simulate_gt >> create_dataset >> retrain
