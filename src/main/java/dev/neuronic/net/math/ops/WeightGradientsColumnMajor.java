package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Compute weight gradients in column-major format: gradients[input][neuron] = input * delta[neuron]
 * This is the outer product transposed to match our column-major weight layout.
 */
public final class WeightGradientsColumnMajor {
    
    public interface Impl {
        void compute(float[] inputs, float[] neuronDeltas, float[][] weightGradients);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
            for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
                float inputValue = inputs[inputIdx];
                for (int neuronIdx = 0; neuronIdx < neuronDeltas.length; neuronIdx++) {
                    weightGradients[inputIdx][neuronIdx] = inputValue * neuronDeltas[neuronIdx];
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
                        "dev.neuronic.net.math.ops.vector.WeightGradientsColumnMajorVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    public static void compute(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
        IMPL.compute(inputs, neuronDeltas, weightGradients);
    }
    
    private static void computeVectorized(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
        IMPL.compute(inputs, neuronDeltas, weightGradients);
    }
    
    private static void computeScalar(float[] inputs, float[] neuronDeltas, float[][] weightGradients) {
        new ScalarImpl().compute(inputs, neuronDeltas, weightGradients);
    }
    
    private WeightGradientsColumnMajor() {}
}