from os.path import join as j

relative_path_to_root = j("..","..")

# use and abuse from os.path.join() (here aliased as "j") it ensures cross OS compatible paths
data_folder = j(relative_path_to_root, "data")
figures_folder = j(relative_path_to_root, "report", "figures")

# STEP 0 dataset exploration
# ===================================

kaggle_folder = j(data_folder, "kaggle_painters_by_numbers")

all_data_info = j(kaggle_folder, "all_data_info.csv")

portraits_csv = j(data_folder, "portraits.csv")



# STEP 1 train model
# ===================================

image_folders = [
    j(kaggle_folder, train_set)
    for train_set in [
        "train",
        "train_1",
        "train_2",
        "train_3",
        "train_4",
        "train_5",
        "train_6",
        "train_7",
        "train_8",
        "train_9"
    ]
]
# ...



object_detection_results_folder = j(data_folder, "object_detection_results", "<CASESTUDY>_<MODEL>")

images_with_boxes_folder = j(object_detection_results_folder, "images_with_boxes")

object_detection_results_csv_folder = j(object_detection_results_folder, "objects")
object_detection_results_csv_path = j(object_detection_results_csv_folder, "<IMAGENAME>.csv")

