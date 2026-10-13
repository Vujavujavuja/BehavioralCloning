# Behavioral Cloning in MiniGrid

Teaching an agent to reach the goal in a grid world by imitating a human, with no reward signal during training.

Behavioral cloning treats control as supervised learning: record what a person does in each state, then train a network to predict the same action from the same observation.

## Pipeline

1. **Record** (`record.py`): play `MiniGrid-Empty-6x6-v0` with the keyboard. Every fully observable frame and the action taken are saved to `observations.npy` and `actions.npy`.
2. **Train** (`training.py`): a small dense network (two hidden layers of 256 units with dropout) learns to map an observation to one of the 7 MiniGrid actions.
3. **Test** (`test.py`): the trained policy plays 10 episodes on `MiniGrid-Empty-Random-6x6-v0`, where the start position is random, and reports the mean reward.

`insp.py` prints a few recorded observations for a sanity check.

## Running it

```bash
pip install -r requirements.txt
python record.py     # arrow keys to move, Backspace to reset, Esc to finish and save
python training.py
python test.py
```

A recorded demonstration set and a trained model (`trained_model.keras`) are included, so `test.py` runs straight away.

## Stack

TensorFlow/Keras, Gymnasium, MiniGrid, pygame.
