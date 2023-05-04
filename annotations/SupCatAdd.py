import json

# Load the JSON file
with open('/home/elliot/cours/BA6/BAProject/annotations/instances_default_2.json', 'r') as f:
    data = json.load(f)

# Loop through the categories and assign supercategories based on their name
for cat in data['categories']:
    if cat['name'] in ['Generic Object', 'Car', 'Bus', 'Bicycle', 'Motorbike', 'Airplane', 'Boat', 'Train', 'Chair', 'Sofa', 'Table', 'Plant']:
        cat['supercategory'] = 'OBJECT'
    elif cat['name'] in ['Cat', 'Dog', 'Cow', 'Horse', 'Sheep', 'Bird', 'Generic Animal']:
        cat['supercategory'] = 'ANIMAL'
    elif cat['name'] in ['Building', 'Background']:
        cat['supercategory'] = 'BACKGROUND'
    elif cat['name'] in ['Face', 'Hand', 'Character']:
        cat['supercategory'] = 'CHARACTER'
    elif cat['name'] in ['Comic Bubble', 'Text']:
        cat['supercategory'] = 'TEXT'
    elif cat['name'] in ['Panel','Horizon']:
        cat['supercategory'] = 'OTHER'

# Save the updated JSON file
with open('/home/elliot/cours/BA6/BAProject/annotations/instances_default_modified_2.json', 'w') as f:
    json.dump(data, f)

