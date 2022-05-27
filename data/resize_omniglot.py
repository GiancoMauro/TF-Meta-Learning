from PIL import Image
import glob

# add omniglot data in the /data folder. Change main_path whether needed.

main_path = "/omniglot"

background_eval_paths = glob.glob(main_path + '*')

all_characters_paths = []
all_images = []

for dataset_path in background_eval_paths:

    all_alphabet_paths = glob.glob(main_path + '*')

    for alphabet_path in all_alphabet_paths:

        all_characters_paths = glob.glob(alphabet_path.replace("\\", "/") + "/" + '*')

        for character_path in all_characters_paths:

            examples_paths = glob.glob(character_path.replace("\\", "/") + "/" + '*')

            for example_path in examples_paths:

                all_images.append(example_path)


for image_file in all_images:
    print(image_file)
    im = Image.open(image_file)
    im = im.resize((28, 28), resample=Image.LANCZOS)
    im.save(image_file)