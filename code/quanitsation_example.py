import utils_quantisation

import numpy as np


np.random.seed(42)
# Example usage:
# Generate some random data
original_data = [np.array([-0.123, -0.4555, 0.2345, 0.778])]
print(np.add(original_data, 3.0)/6.0)


# Quantize the data
quantized_data = utils_quantisation.quantize(original_data, 3.0, 2**20) 

# Display the results
print("Original Data:")
print(original_data)
print("\nQuantized Data:")
print(quantized_data)

dequantized_data = utils_quantisation.dequantize(quantized_data, 3.0, 2**20, 1)
 
print("\nDequantized Data:")
print(dequantized_data)