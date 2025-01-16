list_name=evaluation_path_list3.txt
echo '\n'
for var in $(cat ${list_name} | tr -d '\r'); do
  echo $var
  echo '\n'
done

echo '\n'

for var in $(cat ${list_name} | tr -d '\r'); do
  echo $var
  echo '\n'
  python3 evaluate.py \
  --result_path ${var}/result \
  --show_path ${var}/evaluation_ReMOVE_crop1 \
  --data_indexes_path dataset/Adobe_EntitySeg/image_indexes/val_lr_inpainting_d5_1105_indexes.yml \
  --crop 1
done

# --data_indexes_path dataset/coco/image_indexes/val2017_inpainting_indexes.yml
# dataset/Adobe_EntitySeg/image_indexes/val_lr_inpainting_d5_1105_indexes.yml