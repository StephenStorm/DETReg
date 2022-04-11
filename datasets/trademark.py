from pathlib import Path
from datasets.coco import CocoDetection

def build_trademark(image_set, args):
    root = Path('data/trademark/')
    assert root.exists(), f'provided trademark path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "train" / "images", root / "val" / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    no_cats = False
    filter_pct = -1
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), no_cats=no_cats, filter_pct=filter_pct)
    return dataset