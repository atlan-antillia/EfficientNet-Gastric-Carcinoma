#
# DatasetCategorizer.py
#

import os
import sys
import glob
import shutil
import traceback

class DatasetCategorizer:

  def __init__(self):
    pass

  def get_fullpath(self, all_fullpath, filename):
    for full_path in all_fullpath:
      if full_path.endswith(filename):
       return full_path
    return None


  def run(self, image_dirs, csv_file, output_dir):
    all_fullpath = []

    for image_dir in image_dirs:
      #print(image_dir)
      files = glob.glob(image_dir + "/*.png")
      #print(files)
      temp_full = []
      for file in files:
        temp_full.append(file)

      all_fullpath += temp_full

    print("--- all_fullpath {}".format(all_fullpath))
    input("HIT")
    filenames = []
    categories = []
    #print("--- all_images {}".format(all_images))
    normal = []
    category1 = []
    category2 = []
    with open(csv_file, "r") as f:
       lines = f.readlines()
       for line in lines[1:]:
         row = line.split(",")
         filename = row[0]
         category = int(row[1])
       
         if category == 0:
           #print(" filename {} category {}".format(filename, category))
           fullpath = get_fullpath(all_fullpath, filename)
           print(" filename {} fullpath {} category {}".format(filename, fullpath, category))
           normal.append(fullpath)

         if category == 1:
           #print(" filename {} category {}".format(filename, category))
           fullpath = get_fullpath(all_fullpath, filename)
           print(" filename {} fullpath {} category {}".format(filename, fullpath, category))
           category1.append(fullpath)

         if category == 2:
           fullpath = get_fullpath(all_fullpath, filename)
           print(" filename {} fullpath {} category {}".format(filename, fullpath, category))
           category2.append(fullpath)

    print(" normal    len {}".format(len(normal)))
    print(" category1 len {}".format(len(category1)))
    print(" category2 len {}".format(len(category2)))
    print(" normal 0 {}".format(normal[0]))

    categorized_files_list = [normal, category1, category2]
    categories = ["category_0", "category_1", "category_2"]

    for i, category in enumerate(categories):
      print("---- {}".format(i))
      #print(categorized_files_list[i])
      category_dir = os.path.join(output_dir, category)

      if not os.path.exists(category_dir):
        print("---- category_dir {}".format(category_dir))
        os.makedirs(category_dir)
      categorized_files = categorized_files_list[i]
      print("--- categorized_files {}".format(categorized_files))
      for file in categorized_files:
        shutil.copy2(file, category_dir)


if __name__ == "__main__":
  try:
    images_dir = ["./stage1/p_image", "./stage2/p_image"]
    output_dir  = "./Image-dataset-master"
    csv_file   = "./train.csv"

    categorizer = DatasetCategorizer()

    categorizer.run(images_dir, csv_file, output_dir)

  except:
    traceback.print_exc()
