def read_data_set(basepath):
    import os
    vehicle_path = os.path.join(basepath, 'vehicles')
    vehicle_data = read_img_files_from_subfolders(vehicle_path)
    non_vehicle_path = os.path.join(basepath, 'non-vehicles')
    non_vehicle_data = read_img_files_from_subfolders(non_vehicle_path)
    return vehicle_data, non_vehicle_data

def read_img_files_from_subfolders(path):
    import glob
    pattern = path + '/**/*.png'
    files = glob.glob(pattern, recursive=True)
    return files