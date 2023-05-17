import json
from PIL import Image
import sys
import os
import numpy as np
if "./" not in sys.path:
    sys.path.append("./")

from ..fromage import models
import matplotlib.pyplot as plt

from typing import Union, Tuple


class PromptParser:

    def __init__(self,
                 model_dir: str,
                 visual_emb_path: str = "src/fromage_inf/fromage_model/visual_embs.pt"):
        # Load model
        self.model = models.load_fromage(model_dir, visual_emb_path)

    def __call__(self, data: Union[dict, str], root: str = "./", max_img_per_ret: int = 3):
        """Predict data and display results in jupyter cell"""

        if isinstance(data, str):
            data = self.load_json(data)

        # Get input prompts and retrieved images
        prompt, pred, str_gt, gt = self.predict(data, max_img_per_ret=max_img_per_ret)

        print("Input conversation:")
        self.display(prompt, root=root)

        print(f"Expectation: {str_gt[0].replace(' [RET]', '')}")
        self.display(gt, root=root)

        print('Prediction:')
        self.display(pred, root=root)

    # =====================================================================
    # Parsing
    # =====================================================================

    def parse_prompt(self, prompt: str):
        """Parse a single prompt, replacing tagged images with their embeddings"""

        words = prompt.split()
        output = []

        for word in words:

            if self._is_parsed_image(word):
                # Insert image here
                output.append([word[1:-1]])
            elif len(output) and isinstance(output[-1], str):
                # Attach text to last text
                output[-1] += f" {word}"
            else:
                # Start new text
                output.append(word)

        return output

    def parse_dict(self, d: dict) -> Tuple[list, str]:
        """Given a dictionary of a config, generate input and output prompts"""

        # Generate entire input conversation
        inputs = []
        for prompt in d["inputs"]:
            inputs.extend(self.parse_prompt(prompt))

        # Double check if anything was generated
        if not len(inputs):
            raise ValueError("Generated input is empty. Your input was: ", d)

        # Append [RET] to end of input
        if not isinstance(inputs[-1], str):
            inputs.append("[RET]")
        elif not inputs[-1].endswith("[RET]"):
            inputs[-1] += " [RET]"

        # Generate expected output
        output = d["expected_output"]
        if not output.endswith("[RET]"):
            output += " [RET]"

        return inputs, [output]

    def parse_config(self, json_file: str) -> Tuple[list, str]:
        """Loads a config json and parses to model input"""
        return self.parse_dict(self.load_json(json_file))

    # =====================================================================
    # Predicting
    # =====================================================================

    def predict(self, data_dict: dict, max_img_per_ret: int = 3):
        """Runs retrieval for conig dictionary"""
        n = max_img_per_ret
        # Get inputs for models
        prompt, str_ground_truth = self.parse_dict(data_dict)
        # Encode conversation and run retrieval
        prediction = self.model.generate_for_images_and_texts(prompt, max_img_per_ret=n)
        ground_truth = self.model.generate_for_images_and_texts(str_ground_truth, max_img_per_ret=1)

        # Return with conversation and images
        return prompt, prediction, str_ground_truth, ground_truth

    def predict_json(self, json_file: str, **kwargs):
        """Runs retrieval prediction for json file"""
        data_dict = self.parse_config(json_file)
        return self.predict(data_dict, **kwargs)

    # =====================================================================
    # Display
    # =====================================================================

    def display(self, model_outputs, root="./"):
        """Display conversation"""
        root = os.path.join(root, "src/benchmark")

        for output in model_outputs:

            if type(output) == str:
                print(output)
            elif type(output) == list:
                # Use this to display the single prompt image
                if len(output) == 1:
                    if isinstance(output[0], str):
                        image = Image.open(os.path.join(root, output[0] + ".jpg"))
                    else:
                        image = output[0]
                    image = image.resize((224, 224))
                    image = image.convert('RGB')
                    plt.axis('off')
                    plt.imshow(image)
                # Use this to display the RET image/s
                else:
                    fig, ax = plt.subplots(1, len(output), figsize=(3 * len(output), 3))
                    for i, image in enumerate(output):
                        try:
                            image = image.resize((224, 224))
                            image = image.convert('RGB')
                        except:
                            image = np.zeros((224, 224, 3), dtype=np.uint8)
                        ax[i].imshow(image)
                        ax[i].set_title(f'Retrieval #{i+1}')
                        ax[i].set_axis_off()
                plt.show()

    # =====================================================================
    # Utility
    # =====================================================================

    def load_json(self, json_file: str) -> dict:
        """Load config json file and store as a dict"""
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def _is_parsed_image(self, word: str):
        """check if user wants to replace this with image embeddings"""
        return word.startswith("{") and word.endswith("}")


if __name__ == "__main__":
    parser = PromptParser("./fromage_inf/fromage_model",
                          "./fromage_inf/fromage_model/visual_embs.pt")
    print(parser.parse_config("res/example.json"))