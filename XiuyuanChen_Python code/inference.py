# -*- coding: utf-8 -*-

# The script is implemented to predict the label of the transmitter.

import numpy as np
import tensorflow as tf

def infer(path):
    # Load the model and assign the tensor
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    
    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load the test data and test model output
    input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    loadData = np.load(r'./npydata/test/test1.npy')
    result = []
    for i in range(0,loadData.shape[0]):
        input_data = np.array(loadData[i][:][:].reshape(1,256,63,1), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
    
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        index = np.argmax(output_data)
        result.append(index+1)
    #print(result)
    #print({key: value for key, value in dict(Counter(result)).items() if value > 0})
    print("The inferred serial number of all LoPy4s is", max(result,key=result.count))
    # Return the label with the maximum occurrence
    return max(result,key=result.count)

#infer("rffi_model.tflite")