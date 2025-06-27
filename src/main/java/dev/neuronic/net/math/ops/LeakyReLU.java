package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Leaky ReLU activation function implementation.
 * 
 * f(x) = x if x > 0, else alpha * x
 * f'(x) = 1 if x > 0, else alpha
 */
public final class LeakyReLU {
    
    public interface Impl {
        void activate(float[] input, float alpha, float[] output);
        void derivative(float[] input, float alpha, float[] output);
        void activateRange(float[] input, float alpha, float[] output, int start, int end);
        void derivativeRange(float[] input, float alpha, float[] output, int start, int end);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void activate(float[] input, float alpha, float[] output) {
            for (int i = 0; i < input.length; i++)
                output[i] = input[i] > 0f ? input[i] : input[i] * alpha;
        }
        
        @Override
        public void derivative(float[] input, float alpha, float[] output) {
            for (int i = 0; i < input.length; i++)
                output[i] = input[i] > 0f ? 1.0f : alpha;
        }
        
        @Override
        public void activateRange(float[] input, float alpha, float[] output, int start, int end) {
            for (int i = start; i < end; i++)
                output[i] = input[i] > 0f ? input[i] : input[i] * alpha;
        }
        
        @Override
        public void derivativeRange(float[] input, float alpha, float[] output, int start, int end) {
            for (int i = start; i < end; i++)
                output[i] = input[i] > 0f ? 1.0f : alpha;
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.LeakyReLUVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Apply Leaky ReLU activation: output[i] = input[i] > 0 ? input[i] : alpha * input[i]
     */
    public static void activate(float[] input, float alpha, float[] output) {
        IMPL.activate(input, alpha, output);
    }
    
    /**
     * Compute Leaky ReLU derivative: output[i] = input[i] > 0 ? 1.0f : alpha
     */
    public static void derivative(float[] input, float alpha, float[] output) {
        IMPL.derivative(input, alpha, output);
    }
    
    static void activateVectorized(float[] input, float alpha, float[] output) {
        IMPL.activate(input, alpha, output);
    }

    static void activateScalar(float[] input, float alpha, float[] output) {
        new ScalarImpl().activate(input, alpha, output);
    }

    static void derivativeVectorized(float[] input, float alpha, float[] output) {
        IMPL.derivative(input, alpha, output);
    }

    static void derivativeScalar(float[] input, float alpha, float[] output) {
        new ScalarImpl().derivative(input, alpha, output);
    }
    
    // Range-based operations for parallel execution
    
    public static void activateRange(float[] input, float alpha, float[] output, int start, int end) {
        IMPL.activateRange(input, alpha, output, start, end);
    }
    
    public static void derivativeRange(float[] input, float alpha, float[] output, int start, int end) {
        IMPL.derivativeRange(input, alpha, output, start, end);
    }
    
    static void activateVectorizedRange(float[] input, float alpha, float[] output, int start, int end) {
        IMPL.activateRange(input, alpha, output, start, end);
    }
    
    static void activateScalarRange(float[] input, float alpha, float[] output, int start, int end) {
        new ScalarImpl().activateRange(input, alpha, output, start, end);
    }
    
    static void derivativeVectorizedRange(float[] input, float alpha, float[] output, int start, int end) {
        IMPL.derivativeRange(input, alpha, output, start, end);
    }
    
    static void derivativeScalarRange(float[] input, float alpha, float[] output, int start, int end) {
        new ScalarImpl().derivativeRange(input, alpha, output, start, end);
    }
    
    private LeakyReLU() {}
}