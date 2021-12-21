import glob
from create_annotations import *

# Label ids of the dataset
category_ids = {
    "sky": 3,
    "tree": 5,
    "grass": 10,
    "earth": 14,
    "rock": 14,
    "mountain": 17,
    "mount": 17,
    "plant": 18,
    "flora": 18,
    "life": 18,
    "water": 22,
    "sea": 27,
    "river": 61
}

# category_colors = {
#     "(0, 0, 0)": 0, # Outlier
#     "(5, 5, 5)": 3, # sky
#     "(3, 3, 3)": 5, # tree
#     "(10, 10, 10)": 10, # grass
#     "(18, 18, 18)": 14, # earth
#     "(18, 18, 18)": 14, # rock
#     "(22, 22, 22)": 17, # mountain
#     "(22, 22, 22)": 17, # mount
#     "(61, 61, 61)": 18, # plant
#     "(61, 61, 61)": 18, # flora
#     "(61, 61, 61)": 18, # life
#     "(14, 14, 14)": 22, # water
#     "(17, 17, 17)": 27, # sea
#     "(14, 14, 14)": 61 #river
# }

category_colors = {
    "(0, 0, 0)": 0, # Outlier
    "(3, 3, 3)": 3, # sky
    "(5, 5, 5)": 5, # tree
    "(10, 10, 10)": 10, # grass
    "(14, 14, 14)": 14, # earth
    "(14, 14, 14)": 14, # rock
    "(17, 17, 17)": 17, # mountain
    "(17, 17, 17)": 17, # mount
    "(18, 18, 18)": 18, # plant
    "(18, 18, 18)": 18, # flora
    "(18, 18, 18)": 18, # life
    "(22, 22, 22)": 22, # water
    "(27, 27, 27)": 27, # sea
    "(61, 61, 61)": 61 #river
}

multipolygon_ids = [3, 17, 14]

def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    for mask_image in glob.glob(maskpath + "*.png"):
        original_file_name = os.path.basename(mask_image).split(".")[0] + ".png"
        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size
        # "images" info 
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)
        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]
            # "annotations" info
            polygons, segmentations = create_sub_mask_annotation(sub_mask)
            # Check if we have classes that are a multipolygon
            if category_id in multipolygon_ids:
                # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)
                annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    
                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id


# if __name__ == "__main__":
#     # Get the standard COCO JSON format
#     coco_format = get_coco_json_format()
    
#     mask_path = "data/train1_mask/"
        
#     # Create category section
#     coco_format["categories"] = create_category_annotation(category_ids)
    
#     # Create images and annotations sections
#     coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

#     with open("output1/{}.json".format(keyword),"w") as outfile:
#         json.dump(coco_format, outfile)
        
    
