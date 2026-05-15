from collections import Counter

# ── Object group definitions ──────────────────────────────────────────────────
INDOOR_OBJECTS = {
    'chair', 'couch', 'sofa', 'bed', 'dining table', 'diningtable',
    'toilet', 'tv', 'tvmonitor', 'laptop', 'mouse', 'keyboard',
    'cell phone', 'book', 'clock', 'vase', 'scissors', 'remote',
    'microwave', 'oven', 'refrigerator', 'sink', 'pottedplant',
    'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'wine glass',
    'toaster', 'toothbrush', 'hair drier', 'teddy bear', 'backpack',
    'handbag', 'suitcase', 'umbrella', 'tie',
}

OUTDOOR_OBJECTS = {
    'car', 'truck', 'bus', 'motorbike', 'motorcycle', 'bicycle',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'boat', 'aeroplane',
    'airplane', 'train', 'kite', 'sports ball', 'frisbee',
    'skateboard', 'surfboard', 'tennis racket', 'baseball bat',
    'baseball glove', 'skis', 'snowboard',
}

VEHICLE_LABELS = {
    'car', 'truck', 'bus', 'motorbike', 'motorcycle', 'bicycle',
    'boat', 'aeroplane', 'airplane', 'train',
}

FOOD_LABELS = {
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake',
}

ELECTRONICS = {
    'laptop', 'tv', 'tvmonitor', 'cell phone', 'keyboard',
    'mouse', 'remote',
}

ANIMALS = {
    'cat', 'dog', 'bird', 'horse', 'cow', 'sheep',
    'elephant', 'bear', 'zebra', 'giraffe',
}

SPORTS = {
    'sports ball', 'frisbee', 'skateboard', 'surfboard',
    'tennis racket', 'baseball bat', 'baseball glove',
    'skis', 'snowboard', 'kite',
}


def analyze_scene(detected_objects):
    """
    Analyse the detected object list and return:
      (scene_type: str, observations: list[str])
    """
    if not detected_objects:
        return 'Unknown / Unclassified', [
            'No identifiable objects were detected in this scene.',
            'The image may be blurry, too dark, or contain unusual subjects.',
            'Try uploading a clearer image with better lighting.',
        ]

    normed   = [o.lower().strip() for o in detected_objects]
    obj_set  = set(normed)
    counts   = Counter(normed)
    total    = len(detected_objects)
    unique_n = len(obj_set)

    # ── Scene classification ──────────────────────────────────────────────────
    indoor_score  = len(obj_set & INDOOR_OBJECTS)
    outdoor_score = len(obj_set & OUTDOOR_OBJECTS)
    has_person    = 'person' in obj_set

    if indoor_score > outdoor_score:
        scene_type = 'Indoor / Interior Environment'
    elif outdoor_score > indoor_score:
        scene_type = 'Outdoor / Exterior Environment'
    elif indoor_score == outdoor_score and indoor_score > 0:
        scene_type = 'Mixed / Transitional Environment'
    else:
        scene_type = 'Unclassified Environment'

    # ── Build observations ────────────────────────────────────────────────────
    obs = []

    # Persons
    person_count = counts.get('person', 0)
    if person_count == 1:
        obs.append('A single human subject is present in the scene.')
    elif person_count == 2:
        obs.append('Two individuals have been identified in the frame.')
    elif person_count >= 3:
        obs.append(f'{person_count} people detected — this appears to be a group or crowd setting.')

    # Vehicles
    vehicles = obj_set & VEHICLE_LABELS
    if vehicles:
        vlist = ', '.join(sorted(vehicles))
        if len(vehicles) == 1:
            obs.append(f'A {vlist} is visible — possible road or transport environment.')
        else:
            obs.append(f'Multiple vehicle types identified: {vlist}.')

    # Animals
    animals = obj_set & ANIMALS
    if animals:
        alist = ', '.join(sorted(animals))
        obs.append(f'Animal presence confirmed: {alist} detected in the scene.')

    # Electronics / workspace
    elec = obj_set & ELECTRONICS
    if elec:
        elist = ', '.join(sorted(elec))
        obs.append(f'Electronic devices present ({elist}) — workspace or home setting likely.')

    # Food
    food = obj_set & FOOD_LABELS
    if food:
        flist = ', '.join(sorted(food))
        obs.append(f'Food items identified: {flist}. Possible dining or kitchen context.')

    # Sports
    sport = obj_set & SPORTS
    if sport:
        slist = ', '.join(sorted(sport))
        obs.append(f'Sports-related items detected: {slist}. Recreational activity suspected.')

    # Furniture / room context
    furniture = obj_set & {'chair', 'couch', 'sofa', 'bed', 'diningtable', 'dining table'}
    if furniture:
        obs.append('Furniture elements identified — interior/residential environment confirmed.')

    # Density assessment
    if total > 12:
        obs.append(f'Scene is highly active — {total} total detections across {unique_n} categories.')
    elif total > 6:
        obs.append(f'Moderate scene density — {total} objects across {unique_n} categories detected.')
    else:
        obs.append(f'{total} object(s) identified across {unique_n} unique categor{"y" if unique_n == 1 else "ies"}.')

    # Dominant object
    most_common, most_count = counts.most_common(1)[0]
    if most_count > 1:
        obs.append(f'Dominant detection: "{most_common}" appeared {most_count} time(s).')

    return scene_type, obs
