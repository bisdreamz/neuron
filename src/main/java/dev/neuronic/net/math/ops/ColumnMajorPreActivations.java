package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Compute pre-activations using column-major weight layout for better vectorization.
 * 
 * <p>Column-major layout: weights[input][neuron] allows vectorized operations
 * across multiple neurons simultaneously.
 */
public final class ColumnMajorPreActivations {
    
    public interface Impl {
        void compute(float[] inputs, float[][] weights, float[] biases, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] inputs, float[][] weights, float[] biases, float[] output) {
            System.arraycopy(biases, 0, output, 0, biases.length);
            
            for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
                float inputValue = inputs[inputIdx];
                float[] weightRow = weights[inputIdx];
                
                for (int neuronIdx = 0; neuronIdx < output.length; neuronIdx++) {
                    output[neuronIdx] += inputValue * weightRow[neuronIdx];
                }
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ColumnMajorPreActivationsVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute pre-activations with column-major weight matrix.
     * 
     * <p>Computes: output[neuron] = sum(input[i] * weights[i][neuron]) + bias[neuron]
     * 
     * @param inputs input values (length = number of inputs)
     * @param weights column-major weight matrix: weights[input][neuron]
     * @param biases bias values (length = number of neurons)
     * @param output pre-allocated output buffer (length = number of neurons)
     */
    public static void compute(float[] inputs, float[][] weights, float[] biases, float[] output) {
        if (weights.length != inputs.length)
            throw new IllegalArgumentException("Weight matrix first dimension must match input length");
        if (weights.length > 0 && weights[0].length != output.length)
            throw new IllegalArgumentException("Weight matrix second dimension must match output length");
        if (biases.length != output.length)
            throw new IllegalArgumentException("Bias length must match output length");
        
        IMPL.compute(inputs, weights, biases, output);
    }
    
    public static void computeVectorized(float[] inputs, float[][] weights, float[] output) {
        // No biases version - initialize output to zero
        java.util.Arrays.fill(output, 0.0f);
        float[] zeroBiases = new float[output.length];
        IMPL.compute(inputs, weights, zeroBiases, output);
    }
    
    public static void computeScalar(float[] inputs, float[][] weights, float[] output) {
        // No biases version - initialize output to zero
        java.util.Arrays.fill(output, 0.0f);
        float[] zeroBiases = new float[output.length];
        new ScalarImpl().compute(inputs, weights, zeroBiases, output);
    }
    
    private ColumnMajorPreActivations() {}
}