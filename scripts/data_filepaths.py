from os.path import join as j

relative_path_to_root = j("..","..")

# use and abuse from os.path.join() (here aliased as "j") it ensures cross OS compatible paths
data_folder = j(relative_path_to_root, "data")
figures_folder = j(relative_path_to_root, "report", "figures")

# STEP 0 dataset exploration
# ===================================

kaggle_folder = j(data_folder, "kaggle_painters_by_numbers")

all_data_info = j(kaggle_folder, "all_data_info.csv")

# STEP 1 train model
# ===================================

# ...