## Dataset Training

This is a combination of scripts and files used to train my local Frigate model using [Yolo](https://docs.ultralytics.com/) models as a backbone and [CVAT](https://www.cvat.ai/) for data annotation.

This is mainly used with NVIDIA graphics cards as the training on a CPU would take a long time with the given setup.

Used to take clips from my Frigate feed and fine tune a YOLO model for my camera angles, lighting, etc.

Can use the base YOLO model for detection to generate clips [See Here](https://github.com/TeejMcSteez/YOLOV8s-ONNX)

## Workflow

Import Frigate clips into the `clips/` directory in root, upload clips to CVAT in a new task.

Use the `yolo_func.py` to run `cvat-cli --server-host http://<host>:8080 --auth username:password auto-annotate --function-file yolo_func.py <task-id>` to prelabel clips from Frigate to start out. 

Change model inside `yolo_func.py` to download and use a different model from ultralytics.

After prelabel, export the dataset as a YOLO format (make sure to save images) zip file then copy and unzip inside this directory to output to `obj_train_data` which is a collection of images and text files containing the mappings.

Also see `obj.data` and `obj.names` output from the zip file to ensure the label mappings are correct for the given setup.

Run `split_dataset.py` to take the annotated data to dumbly split 80/20 for train/val directories to use for training data. Run `thin_backgrounds.py` if there is a high background count (low number of images with found objects) this will attempt to lower the background count using [imagehash](https://pypi.org/project/ImageHash/) to delete common images.

Finally running `train.py` with whatever YOLO model you want to use as a base will train the model against the new dataset.
