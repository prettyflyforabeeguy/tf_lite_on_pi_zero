#
#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow Lite export package from Lobe.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


def get_model_and_sig(model_dir):
    """Method to get name of model file. Assumes model is in the tf_lite_model directory for script."""
    with open(os.path.join(model_dir, "./signature.json"), "r") as f:
        signature = json.load(f)
    model_file = model_dir + "/" + signature.get("filename")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file does not exist")
    return model_file, signature


def load_model(model_file):
    """Load the model from path to model file"""
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    return interpreter


def get_prediction(image, interpreter, signature):
    """
    Predict with the TFLite interpreter!
    """
    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}

    if "Image" not in model_inputs:
        raise ValueError("Tensorflow Lite model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")

    # process image to be compatible with the model
    input_data = process_image(image, model_inputs.get("Image").get("shape"))

    # set the input to run
    interpreter.set_tensor(model_inputs.get("Image").get("index"), input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}
    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs


def process_image(image, input_shape):
    """
    Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
    """
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    if image.mode != "RGB":
        image = image.convert("RGB")
    # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
    if width != height:
        square_size = min(width, height)
        left = (width - square_size) / 2
        top = (height - square_size) / 2
        right = (width + square_size) / 2
        bottom = (height + square_size) / 2
        # Crop the center of the image
        image = image.crop((left, top, right, bottom))
    # now the image is square, resize it to be the right shape for the model input
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)


def main(image, model_dir):
    """
    Load the model and signature files, start the TF Lite interpreter, and run prediction on the image.

    Output prediction will be a dictionary with the same keys as the outputs in the signature.json file.
    """
    model_file, signature = get_model_and_sig(model_dir)
    interpreter = load_model(model_file)
    prediction = get_prediction(image, interpreter, signature)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a label for an image.")
    parser.add_argument("image", help="Path to your image file.")
    args = parser.parse_args()
    if os.path.isfile(args.image):
        image = Image.open(args.image)
        # convert to rgb image if this isn't one
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Assume model is in the tf_lite_model directory for this file
        model_dir = os.getcwd() + "/tf_lite_model"
        start_time = datetime.now()
        print(str(start_time) + " Analyzing " + args.image + " please wait...")
        print(main(image, model_dir))
        print (f"Image Classification took {datetime.now() - start_time} seconds")
    else:
        print(f"Couldn't find image file {args.image}")
