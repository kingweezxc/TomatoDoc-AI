from pathlib import Path
import random
import shutil

# ============================================
# TomatoDoc Dataset Preparation
# ============================================

SOURCE_TRAIN = Path("dataset/tomato/train")

OUTPUT = Path("processed_dataset")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================
# TomatoDoc's 10 Classes
# ============================================

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


DISPLAY_NAMES = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___healthy": "Healthy",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_images(folder):
    """Get all image files from a class folder."""

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_folder(folder):
    """Create a folder if it does not exist."""

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


def copy_images(images, destination):
    """Copy images into the destination folder."""

    create_folder(destination)

    for image in images:
        shutil.copy2(
            image,
            destination / image.name
        )


def main():

    print("=" * 60)
    print("TomatoDoc Dataset Preparation")
    print("=" * 60)

    # ----------------------------------------
    # Check original dataset
    # ----------------------------------------

    if not SOURCE_TRAIN.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{SOURCE_TRAIN}"
        )

    # ----------------------------------------
    # Create processed dataset folders
    # ----------------------------------------

    train_output = OUTPUT / "train"
    validation_output = OUTPUT / "validation"
    test_output = OUTPUT / "test"

    create_folder(train_output)
    create_folder(validation_output)
    create_folder(test_output)

    total_train = 0
    total_validation = 0
    total_test = 0

    # ----------------------------------------
    # Process each class
    # ----------------------------------------

    for class_name in CLASSES:

        source_class = SOURCE_TRAIN / class_name

        if not source_class.exists():
            raise FileNotFoundError(
                f"Class folder not found:\n{source_class}"
            )

        images = get_images(source_class)

        random.shuffle(images)

        total_images = len(images)

        # Calculate split sizes
        train_count = int(
            total_images * TRAIN_RATIO
        )

        validation_count = int(
            total_images * VALIDATION_RATIO
        )

        test_count = (
            total_images
            - train_count
            - validation_count
        )

        # Split images
        train_images = images[
            :train_count
        ]

        validation_images = images[
            train_count:
            train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        # Output folders
        train_class = (
            train_output / class_name
        )

        validation_class = (
            validation_output / class_name
        )

        test_class = (
            test_output / class_name
        )

        # Copy images
        copy_images(
            train_images,
            train_class
        )

        copy_images(
            validation_images,
            validation_class
        )

        copy_images(
            test_images,
            test_class
        )

        # Update totals
        total_train += len(train_images)
        total_validation += len(validation_images)
        total_test += len(test_images)

        # Display results
        print()
        print(
            DISPLAY_NAMES[class_name]
        )

        print(
            f"  Original     : {total_images}"
        )

        print(
            f"  Training     : {len(train_images)}"
        )

        print(
            f"  Validation   : {len(validation_images)}"
        )

        print(
            f"  Testing      : {len(test_images)}"
        )

    # ----------------------------------------
    # Final summary
    # ----------------------------------------

    total = (
        total_train
        + total_validation
        + total_test
    )

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Training images   : {total_train}"
    )

    print(
        f"Validation images : {total_validation}"
    )

    print(
        f"Testing images    : {total_test}"
    )

    print(
        f"Total images      : {total}"
    )

    print()
    print(
        f"Processed dataset: {OUTPUT}"
    )

    print()
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()