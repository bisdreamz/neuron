package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Temperature scaling for probability distributions.
 * 
 * Temperature controls the "sharpness" of a probability distribution:
 * - T < 1.0: Makes distribution sharper (more confident)
 * - T = 1.0: No change
 * - T > 1.0: Makes distribution flatter (more diverse)
 * 
 * Algorithm:
 * 1. Convert probabilities to log-space
 * 2. Divide by temperature
 * 3. Apply softmax to renormalize
 */
public final class TemperatureScaling {
    
    private static final float LOG_EPSILON = -1e10f;
    
    public interface Impl {
        void apply(float[] probabilities, float temperature, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void apply(float[] probabilities, float temperature, float[] output) {
            applyScalar(probabilities, temperature, output);
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.TemperatureScalingVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Apply temperature scaling to a probability distribution.
     */
    public static void apply(float[] probabilities, float temperature, float[] output) {
        if (temperature <= 0)
            throw new IllegalArgumentException("Temperature must be positive, got: " + temperature);
        
        if (probabilities.length != output.length)
            throw new IllegalArgumentException("Input and output arrays must have same length");
        
        // Fast path for temperature = 1.0
        if (Math.abs(temperature - 1.0f) < 1e-6f) {
            System.arraycopy(probabilities, 0, output, 0, probabilities.length);
            return;
        }
        
        // Fast path for very low temperature (argmax-like behavior)
        if (temperature < 0.01f) {
            int maxIdx = 0;
            float maxVal = probabilities[0];
            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > maxVal) {
                    maxVal = probabilities[i];
                    maxIdx = i;
                }
            }
            for (int i = 0; i < output.length; i++) {
                output[i] = (i == maxIdx) ? 1.0f : 0.0f;
            }
            return;
        }
        
        IMPL.apply(probabilities, temperature, output);
    }
    
    /**
     * Apply temperature scaling in-place.
     */
    public static void applyInPlace(float[] probabilities, float temperature) {
        apply(probabilities, temperature, probabilities);
    }
    
    /**
     * Apply temperature scaling with excluded indices.
     */
    public static void applyWithExclusions(float[] probabilities, float temperature,
                                         int[] excludeIndices, float[] output) {
        if (temperature <= 0)
            throw new IllegalArgumentException("Temperature must be positive, got: " + temperature);
        
        if (probabilities.length != output.length)
            throw new IllegalArgumentException("Input and output arrays must have same length");
        
        // First, copy and zero out excluded indices
        System.arraycopy(probabilities, 0, output, 0, probabilities.length);
        
        if (excludeIndices != null && excludeIndices.length > 0) {
            for (int idx : excludeIndices) {
                if (idx >= 0 && idx < output.length) {
                    output[idx] = 0.0f;
                }
            }
        }
        
        // Renormalize
        float sum = 0.0f;
        for (float val : output) {
            sum += val;
        }
        
        if (sum <= 0) {
            throw new IllegalArgumentException("All probabilities were excluded");
        }
        
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }
        
        // Apply temperature to the normalized distribution
        applyInPlace(output, temperature);
    }
    
    public static void applyScalar(float[] probabilities, float temperature, float[] output) {
        // Convert to log-space and apply temperature
        float maxLogit = Float.NEGATIVE_INFINITY;
        
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > 0) {
                output[i] = (float) Math.log(probabilities[i]) / temperature;
            } else {
                output[i] = LOG_EPSILON / temperature;
            }
            maxLogit = Math.max(maxLogit, output[i]);
        }
        
        // Apply softmax for numerical stability
        float sumExp = 0.0f;
        for (int i = 0; i < output.length; i++) {
            output[i] = (float) Math.exp(output[i] - maxLogit);
            sumExp += output[i];
        }
        
        // Normalize
        for (int i = 0; i < output.length; i++) {
            output[i] /= sumExp;
        }
    }
    
    private TemperatureScaling() {} // Prevent instantiation
}