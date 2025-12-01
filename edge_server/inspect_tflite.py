import numpy as np
import sys
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter
tflite = None
    
def load_interpreter(model_path):
    if tflite is not None:
        return tflite.Interpreter(model_path=model_path)
    else:
        return TFLiteInterpreter(model_path=model_path)

def inspect(model_path):
    print("Inspecting TFLite model:", model_path)
    interpreter = load_interpreter(model_path)
    interpreter.allocate_tensors()

    print("\n=== INPUT DETAILS ===")
    for d in interpreter.get_input_details():
        print(d)

    print("\n=== OUTPUT DETAILS ===")
    for d in interpreter.get_output_details():
        print(d)

    # optional: run inference on zeros to see output shape
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    dummy_input = np.zeros(input_shape, dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], dummy_input)

    interpreter.invoke()

    print("\n=== OUTPUT TENSOR VALUES (first 5 entries) ===")
    for out in interpreter.get_output_details():
        data = interpreter.get_tensor(out['index'])
        print(f"Output index {out['index']} shape {data.shape}")
        print(data.flatten()[:10])  # preview few values

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_tflite.py model.tflite")
        sys.exit(1)
    inspect(sys.argv[1])
