package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Bias initialization to constant value.
 */
public final class BiasInit {
    
    public interface Impl {
        void compute(float[] biases, float value);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] biases, float value) {
            for (int i = 0; i < biases.length; i++) {
                biases[i] = value;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.BiasInitVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Initialize biases to a constant value.
     * 
     * @param biases bias array to initialize
     * @param value initialization value (typically 0.0f or 0.01f)
     */
    public static void compute(float[] biases, float value) {
        IMPL.compute(biases, value);
    }
    
    static void computeVectorized(float[] biases, float value) {
        IMPL.compute(biases, value);
    }
    
    static void computeScalar(float[] biases, float value) {
        new ScalarImpl().compute(biases, value);
    }
    
    private BiasInit() {}
}