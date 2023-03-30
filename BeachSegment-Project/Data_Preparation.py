# %%
import json, os
import numpy as np
import matplotlib.pyplot as plt
import wget

# %% [markdown]
# ## Training Set

# %%
COCO_stuffpath = os.path.join(os.getcwd(), './Labels/stuff_train2017.json')

with open(COCO_stuffpath) as f:
  TrainDict = json.load(f)

# %%
TrainDict.keys()

# %%
[i for i in TrainDict['categories'] if i['supercategory']=='water']

# %%
[i for i in TrainDict['categories'] if i['supercategory']=='ground']

# %% [markdown]
# https://github.com/nightrome/cocostuff/blob/master/labels.md

# %%
sea_sand_annotations = [i for i in TrainDict['annotations'] if i['category_id'] in [154, 155]]
img_with_sea_or_sand = [i['image_id'] for i in sea_sand_annotations]

# %%
unq, unq_idx, unq_cnt = np.unique(img_with_sea_or_sand, return_inverse=True, return_counts=True)
img_ids_both_sea_sand = unq[unq_cnt > 1]
len(img_ids_both_sea_sand)

# %%
sea_beach_annotations = [i for i in TrainDict['annotations'] if i['image_id'] in img_ids_both_sea_sand]
len(sea_beach_annotations)

# %%
otherstuff_in_beach_name = [j['name'] for j in TrainDict['categories'] if j['id'] in
                            np.unique([i['category_id'] for i in sea_beach_annotations])]
# otherstuff_in_beach_name                         

# %%
stuff_id, freq = np.unique([i['category_id'] for i in sea_beach_annotations], return_counts=True)

# %%
mostfreq_stuff_id = stuff_id[np.where(freq>500)]
[j['name'] for j in TrainDict['categories'] if
 j['id'] in mostfreq_stuff_id]

# %%
AnnotationFinal = [i for i in sea_beach_annotations if i['category_id'] in mostfreq_stuff_id]
len(AnnotationFinal)

# %%
sea_beach_others_Train = {}
sea_beach_others_Train['info'] = TrainDict['info']
sea_beach_others_Train['licenses'] = TrainDict['licenses']
sea_beach_others_Train['categories'] = [i for i in TrainDict['categories'] if i['id'] in mostfreq_stuff_id]
sea_beach_others_Train['annotations'] = AnnotationFinal
sea_beach_others_Train['images'] = [i for i in TrainDict['images'] if i['id'] in img_ids_both_sea_sand]

# %%
with open("../Data/Sea_Beach_others_Train.json", "w") as outfile: 
    json.dump(sea_beach_others_Train, outfile)

# %% [markdown]
# ## Validation Set

# %%
COCO_stuffpath = os.path.join(os.getcwd(), '../Data/stuff_val2017.json')

with open(COCO_stuffpath) as f:
  ValDict = json.load(f)

# %%
sea_sand_annotations = [i for i in ValDict['annotations'] if i['category_id'] in [154, 155]]
img_with_sea_or_sand = [i['image_id'] for i in sea_sand_annotations]

# %%
unq, unq_idx, unq_cnt = np.unique(img_with_sea_or_sand, return_inverse=True, return_counts=True)
img_ids_both_sea_sand = unq[unq_cnt > 1]
len(img_ids_both_sea_sand)

# %%
sea_beach_annotations = [i for i in ValDict['annotations'] if i['image_id'] in img_ids_both_sea_sand]
len(sea_beach_annotations)

# %%
AnnotationFinal_Validation = [i for i in sea_beach_annotations if i['category_id'] in mostfreq_stuff_id]

[j['name'] for j in ValDict['categories'] if
 j['id'] in mostfreq_stuff_id]

# %%
sea_beach_others_Val = {}
sea_beach_others_Val['info'] = ValDict['info']
sea_beach_others_Val['licenses'] = ValDict['licenses']
sea_beach_others_Val['categories'] = [i for i in ValDict['categories'] if i['id'] in mostfreq_stuff_id]
sea_beach_others_Val['annotations'] = AnnotationFinal_Validation
sea_beach_others_Val['images'] = [i for i in ValDict['images'] if i['id'] in img_ids_both_sea_sand] 

# %%
print(len(sea_beach_others_Val['annotations']))
print(len(sea_beach_others_Val['images']))

# %%
with open("./Labels/Sea_Beach_others_Valid.json", "w") as outfile: 
    json.dump(sea_beach_others_Val, outfile)

# %%


# %% [markdown]
# # Download Data

# %%
datasetjson = os.path.join('./Labels/Sea_Beach_others_Train.json')

with open(datasetjson) as f:
  TrainDict = json.load(f)

# %%
TrainDict.keys()

# %%
print(len([i['file_name'] for i in TrainDict['images']]))
print(len(set([i['file_name'] for i in TrainDict['images']])))

# %% [markdown]
# So, there is not any duplicate photo

# %%
ds_path = './Beach-Image-Dataset'
if not os.path.exists(ds_path):
  os.mkdir(ds_path)
  print('folder created')

# %%
for i in range(len(TrainDict['images'])):

  if not os.path.exists(os.path.join(ds_path, TrainDict['images'][i]['file_name'])):
    filename = wget.download(TrainDict['images'][i]['coco_url'], out = ds_path) 

# %%
print(len(os.listdir(ds_path)))


