nerfbaselines train --method wildgaussians \
--data raw_data/camb/scene_KingsCollege/train_full_byorder_85 \
--output outputs30k/camb/scene_KingsCollege \
--set iterations=30000 \
--set densify_until_iter=15000 \
--set position_lr_max_steps=30000

nerfbaselines train --method wildgaussians \
--data raw_data/camb/scene_OldHospital/train_full_byorder_85 \
--output outputs30k/camb/scene_OldHospital \
--set iterations=30000 \
--set densify_until_iter=15000 \
--set position_lr_max_steps=30000

nerfbaselines train --method wildgaussians \
--data raw_data/camb/scene_StMarysChurch/train_full_byorder_85 \
--output outputs30k/camb/scene_StMarysChurch \
--set iterations=30000 \
--set densify_until_iter=15000 \
--set position_lr_max_steps=30000