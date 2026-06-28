from seqconvnet.core import (
    preprocess_las_file,
    VoxelParameters,
)

las_path = "dales_las/train/5110_54320.las"
device = "cuda"
num_classes = 8
force_teach_token = num_classes + 2
voxel_params = VoxelParameters(
    xy_resolution=0.5,
    z_resolution=0.5,
    max_z=256,
    min_rows=128,
    min_cols=128,
)


preprocess_las_file(
    las_path,
    "test/data",
    "label/data",
    num_classes,
    {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
    },
    128,
    True,
    voxel_params,
    device,
)
