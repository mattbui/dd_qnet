# Double Dueling Q Net

## Usage

### Model with image and laser
Train:
```
python qlearning.py
```

For more parameters: `python qlearning --help`

Test model: `python test.py <from_pretrain_dir> <epsilon>`
```
python test.py output/model 0.001
```

### Model with laser, only use target
Replace `gazebo_turtlebot_maze_color.py` with `gazebo_turtlebot_maze_color_laser_only.py`

Train
```
python laser_learning.py
```

Test model: `python test_laser_only.py <from_pretrain_dir> <epsilon>`
```
python test_laser_only.py output/laser_only 0.001
```

