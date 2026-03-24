# Reinforcement Learning Recommender

This folder contains only the core files for the Reinforcement Learning recommender model.

## Main idea
- Algorithm: `Multi-Armed Bandit`
- Goal: maximize engagement reward
- Action: recommend a movie
- Reward:
  - rating `>= 4` → `+2`
  - rating `< 4` → `+1`
  - no interaction/skip → `-1`

## Core files
- `main.py` - trains, evaluates, and exports outputs
- `mab_recommender.py` - bandit algorithms and RL agent
- `train_mab.py` - training pipeline
- `prediction_eval.py` - recommendation prediction utilities
- `lstm_integration.py` - optional integration with Member 3's LSTM state vectors
- `requirements.txt` - dependencies
- `integration.md` - guide for backend/frontend integration

## Data
The model reads data from `data/`:
- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

## Run
From this folder:

```bash
python main.py --algorithm thompson --samples 5000
```

For a larger run:

```bash
python main.py --algorithm thompson --samples 50000
```

## Output
The run creates:
- `models/mab_agent_thompson.pkl`
- `results/training_report.txt`
- `results/test_metrics_thompson.json`
- `results/sample_recommendations.json`
- `results/comprehensive_analysis.txt`

## Notes
- `thompson` is the recommended algorithm.
- `contextual` can be used when Member 3's LSTM state integration is ready.
- The current model was verified to run successfully in this folder.
