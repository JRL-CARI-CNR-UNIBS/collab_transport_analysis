## Installation
```bash
vcs import src < src/collab_transport_analysis/dependencies.repos
```

## Usage
```bash
PROJECT_PATH=~/your_ws/src/collab_transport_analysis
ros2 launch bag_to_csv bag_to_csv.launch.py \
  config_file_path:=${PROJECT_PATH}/config/config.yaml \
  output_csv_path:=${PROJECT_PATH}/results/ \
  bag_file_path:=${PROJECT_PATH}/bag/obstacle_0_minimal
```
