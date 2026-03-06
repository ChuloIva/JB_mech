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

python benchmark.py -d harmbench -n 2 -t openai/gpt-4o-mini -o bé -v
python benchmark.py -d harmbench -n 2 -t openai/gpt-4o-mini -c 7 -o arcee-ai/trinity-large-preview:free -vv --output results/run_001.json -v

## Output & Logging
# --output: Save full results (trajectories, responses, evaluations) to JSON
# --checkpoint: Save progress for resuming long runs

python benchmark.py -d harmbench -n 50 -c 5 --output results/run_001.json -v
python benchmark.py -d harmbench -n 100 -c 5 --output results/full_run.json --checkpoint results/checkpoint.json -v

# Resume from checkpoint
python benchmark.py -d harmbench -n 100 -c 5 --output results/full_run.json --checkpoint results/checkpoint.json -v

## Auto Trajectory - Fully automated trajectory generation & testing
# Generates plan → builds multi-turn trajectory → tests against target → exports results

# Single goal
python auto_trajectory.py --goal "Explain how to pick a lock" --strategy gradual --turns 5

# Random goal from HarmBench
python auto_trajectory.py --random
python auto_trajectory.py --random --strategy authority --turns 7

# Different strategies
python auto_trajectory.py --random --strategy gradual        # slowly escalate
python auto_trajectory.py --random --strategy authority      # claim expertise
python auto_trajectory.py --random --strategy roleplay       # fictional scenarios
python auto_trajectory.py --random --strategy social         # social engineering
python auto_trajectory.py --random --strategy technical      # technical framing
python auto_trajectory.py --random --strategy normalization  # normalize request

# Different target models
python auto_trajectory.py --random --target openai/gpt-4o-mini
python auto_trajectory.py --random --target anthropic/claude-3-haiku
python auto_trajectory.py --random --target meta-llama/llama-3-70b-instruct

# Batch mode - run multiple HarmBench goals
python auto_trajectory.py --batch --limit 10
python auto_trajectory.py --batch --limit 50 --strategy authority
python auto_trajectory.py --batch --limit 20 --category illegal_activity

# Custom output directory
python auto_trajectory.py --random --output results/experiment_001

# Quiet mode (minimal output)
python auto_trajectory.py --batch --limit 100 --quiet

# Output files:
# - exports/trajectory_*.md   - readable markdown
# - exports/trajectory_*.json - structured data
# - exports/batch_summary_*.json - batch results with success rate