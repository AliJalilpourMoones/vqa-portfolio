# VQA v2 Dataset

This project uses the VQA v2 dataset. Due to its large size, it is not included in this repository.

## Setup Instructions

1.  **Download the data** from the official site: [https://visualqa.org/download.html](https://visualqa.org/download.html). You will need:
    * The **training images** (2014).
    * The **validation images** (2014).
    * The **training questions**.
    * The **validation questions**.
    * The **training answers**.
    * The **validation answers**.

2.  **Unzip the files** and organize them in this `data` directory as follows:

    ```
    data/
    ├── train2014/
    │   ├── COCO_train2014_...jpg
    │   └── ...
    ├── val2014/
    │   ├── COCO_val2014_...jpg
    │   └── ...
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json
    ```