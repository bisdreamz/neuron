package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Parameter update operation: param[i] = param[i] - learningRate * gradient[i]
 * Used by optimizers to update weights and biases.
 */
public final class ParameterUpdate {
    
    public interface Impl {
        void compute(float[] parameters, float[] gradients, float learningRate);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] parameters, float[] gradients, float learningRate) {
            for (int i = 0; i < parameters.length; i++) {
                parameters[i] -= learningRate * gradients[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.ParameterUpdateVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Update parameters in-place using gradients.
     * 
     * @param parameters the parameters to update (modified in-place)
     * @param gradients the gradients to apply
     * @param learningRate the learning rate multiplier
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void compute(float[] parameters, float[] gradients, float learningRate) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Parameters and gradients must have same length: params=" + 
                                             parameters.length + ", gradients=" + gradients.length);
        
        IMPL.compute(parameters, gradients, learningRate);
    }
    
    private static void clipGradients(float[] gradients, float maxNorm) {
        float norm = 0.0f;
        for (float grad : gradients) {
            norm += grad * grad;
        }
        norm = (float) Math.sqrt(norm);
        
        if (norm > maxNorm) {
            float scale = maxNorm / norm;
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] *= scale;
            }
        }
    }
    
    static void computeVectorized(float[] parameters, float[] gradients, float learningRate) {
        IMPL.compute(parameters, gradients, learningRate);
    }
    
    static void computeScalar(float[] parameters, float[] gradients, float learningRate) {
        new ScalarImpl().compute(parameters, gradients, learningRate);
    }
    
    private ParameterUpdate() {}
}