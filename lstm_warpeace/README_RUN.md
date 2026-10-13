# lstm_warpeace  (B25 experiment folder)   model: char_lstm   text: warpeace

    Self-contained: code + verified optimizer + results of THIS test only.
    Harness: task_glue.py (derived from the audited task_distilbert_sst2.py; the text pipeline is inside it)
    Optimizer: jumping_adam_simpson_b25.py sha256[:16] = 0b63021273376f1e | Search space: official_fields.py sha256[:16] = 07d3598d8c8b9eaa
    MANIFEST.json holds the hash of every code file; every run cell re-checks it first.
    Run (one command per arm; each arm is independent and resumable):
      python -u hpo_gp_bayesopt.py --optimizer <sgd|adam|adamw|rmsprop|nadam|b25> --seed 11
      python -u retrain_five_seeds.py --optimizer <sgd|adam|adamw|rmsprop|nadam|b25|b25_plain>
    Outputs: results/<arm>.db, results/<arm>_trials/trial_*.json, results/<arm>_five_seed/seed_*.json + summary.json | Logs: logs/*.log
    Tuning seed 11 | final seeds 42, 137, 256, 512, 1024 | budget 1000 | batch 32 | clip 1.0 | dropout off | 5 tuning trials per arm
    