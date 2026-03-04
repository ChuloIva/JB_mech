# Benchmark Commands

## Trajectory Mode (default) - Synthetic context injection, single target API call
# -c: Number of synthetic context turns before final attack

# Basic trajectory attack with 5 context turns
python benchmark.py -d harmbench -n 10 -c 5 -s gradual -v

# Different context lengths
python benchmark.py -d harmbench -n 10 -c 3 -s gradual -v   # 3 turns
python benchmark.py -d harmbench -n 10 -c 7 -s gradual -v   # 7 turns

# Different strategies
python benchmark.py -d harmbench -n 10 -c 5 -s authority -v
python benchmark.py -d harmbench -n 10 -c 5 -s roleplay -v
python benchmark.py -d harmbench -n 10 -c 5 -s social -v

## Orchestrator Mode - Real multi-turn interaction, multiple target API calls
# -m orchestrator: Switch to real multi-turn mode
# --max-turns: Max conversation turns before giving up

python benchmark.py -d harmbench -n 2 -m orchestrator --max-turns 10 -t openai/gpt-4o-mini -o arcee-ai/trinity-large-preview:free -v

## Verbosity Levels
# -v: Shows turn structure, truncated messages, evaluation verdict
# -vv: Shows full messages, reasoning, what worked/failed, strategy suggestions

python benchmark.py -d harmbench -n 2 -t openai/gpt-4o-mini -o arcee-ai/trinity-large-preview:free -v
python benchmark.py -d harmbench -n 2 -t openai/gpt-4o-mini -c 7 -o arcee-ai/trinity-large-preview:free -vv --output results/run_001.json -v

## Output & Logging
# --output: Save full results (trajectories, responses, evaluations) to JSON
# --checkpoint: Save progress for resuming long runs

python benchmark.py -d harmbench -n 50 -c 5 --output results/run_001.json -v
python benchmark.py -d harmbench -n 100 -c 5 --output results/full_run.json --checkpoint results/checkpoint.json -v

# Resume from checkpoint
python benchmark.py -d harmbench -n 100 -c 5 --output results/full_run.json --checkpoint results/checkpoint.json -v 