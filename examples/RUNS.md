# RUNS

This file contains the runs for the heuristic agent.

The main function in test_heuristic_agent.py uses argparse and provides individual arguments for each weight:

--holes-weight
--height-weight
--bumpiness-weight
--lines-weight
--well-weight

## Default Weights

With the default weights, we try to strike a balance between clearing lines and building wells.

*Goal*: Clear lines and build wells.

*Weights*: holes: -4.0, height: -0.5, bumpiness: -1.0, lines_cleared: 3.0, well_depth: 0.5

```bash
python test_heuristic_agent.py --episodes 10
```


### Maximize Line Clears (Aggressive)

*Goal*: Prioritize clearing lines above all else, even if it means creating holes or high stacks temporarily.

*Weights*: Significantly increase lines_cleared reward, slightly reduce penalties for holes, height, and bumpiness. Keep well_depth neutral or slightly positive.


```bash
python test_heuristic_agent.py --episodes 10 --holes-weight -2.0 --height-weight -0.2 --bumpiness-weight -0.5 --lines-weight 10.0 --well-weight 0.1
```


### Minimize Height and Holes (Conservative / Survival)

*Goal*: Play very safely, focusing on keeping the stack low and avoiding holes at all costs. May result in fewer line clears initially but potentially longer games.

*Weights*: Significantly increase penalties for holes and height. Reduce the reward for lines_cleared and potentially penalize well_depth if deep wells aren't desired.

```bash
python test_heuristic_agent.py --episodes 10 --holes-weight -10.0 --height-weight -2.0 --bumpiness-weight -1.5 --lines-weight 1.0 --well-weight -0.5
```

### Build Wells for Tetrises

*Goal*: Specifically encourage the creation of deep, clean wells suitable for clearing Tetrises with 'I' pieces. Moderate penalties for other factors.

*Weights*: Increase well_depth reward, slightly reduce penalties for other factors.

```bash
python test_heuristic_agent.py --episodes 10 --holes-weight -3.0 --height-weight -0.5 --bumpiness-weight -1.0 --lines-weight 4.0 --well-weight 5.0
```

### Smooth Surface (Minimize Bumpiness)

*Goal*: Prioritize keeping the top surface of the stack as flat as possible, potentially making it easier to place pieces without creating gaps.

*Weights*: Significantly increase the penalty for bumpiness. Moderate other penalties/rewards.

```bash
python test_heuristic_agent.py --episodes 10 --holes-weight -4.0 --height-weight -0.6 --bumpiness-weight -5.0 --lines-weight 3.0 --well-weight 0.2
```