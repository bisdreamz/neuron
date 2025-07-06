package dev.neuronic.net.math;

import java.lang.reflect.Method;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * High-performance random number generator for neural networks.
 * Uses Xoroshiro128++ algorithm for speed and quality.
 * Supports both scalar and vectorized operations.
 * 
 * Each neural network should have its own FastRandom instance
 * to ensure reproducible results when using seeds.
 */
public final class FastRandom {
    private final RandomGenerator generator;
    
    // Cached vectorized method for fillUniform
    private static final Method VECTOR_FILL_UNIFORM;
    
    static {
        Method vectorFillUniform = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName("dev.neuronic.net.math.vector.FastRandomVector");
                vectorFillUniform = vectorClass.getMethod("fillUniform", 
                        float[].class, float.class, float.class, RandomGenerator.class);
            } catch (Exception e) {
                // Vector implementation not available, fall back to scalar
            }
        }
        VECTOR_FILL_UNIFORM = vectorFillUniform;
    }
    
    /**
     * Create a new FastRandom instance with a specific seed.
     * Use this for reproducible results.
     */
    public FastRandom(long seed) {
        this.generator = RandomGeneratorFactory.of("Xoroshiro128PlusPlus").create(seed);
    }
    
    /**
     * Create a new FastRandom instance with a random seed.
     * Use this when reproducibility is not required.
     */
    public FastRandom() {
        this.generator = RandomGeneratorFactory.of("Xoroshiro128PlusPlus").create();
    }
    
    /**
     * Get the underlying RandomGenerator for advanced usage.
     */
    public RandomGenerator getGenerator() {
        return generator;
    }
    
    /**
     * Generate a random float in [0, 1).
     */
    public float nextFloat() {
        return generator.nextFloat();
    }
    
    /**
     * Generate a random double in [0, 1).
     */
    public double nextDouble() {
        return generator.nextDouble();
    }
    
    /**
     * Generate a random Gaussian (normal) value with mean 0 and stddev 1.
     */
    public double nextGaussian() {
        return generator.nextGaussian();
    }
    
    /**
     * Generate a random int in [0, bound).
     */
    public int nextInt(int bound) {
        return generator.nextInt(bound);
    }

    /**
     * Fill buffer with uniform floats in [min, max).
     */
    public void fillUniform(float[] buffer, float min, float max) {
        int len = buffer.length;
        
        // Try vectorized implementation if available and worthwhile
        if (VECTOR_FILL_UNIFORM != null && len >= Vectorization.getVectorLength() * 2) {
            try {
                VECTOR_FILL_UNIFORM.invoke(null, buffer, min, max, generator);
                return;
            } catch (Exception e) {
                // Fall back to scalar on any error
            }
        }
        
        // Scalar implementation
        float range = max - min;
        for (int i = 0; i < len; i++) {
            buffer[i] = generator.nextFloat() * range + min;
        }
    }

    /** Fill buffer with uniform floats in (–limit, +limit). */
    public void fillSymmetric(float[] buffer, float limit) {
        fillUniform(buffer, -limit, limit);
    }

    /**
     * Fill buffer with Gaussian-distributed floats (mean, stddev).
     * Purely scalar, since the Vector API has no native Gaussian sampler.
     */
    public void fillGaussian(float[] buffer, float mean, float stddev) {
        for (int i = 0, len = buffer.length; i < len; i++) {
            buffer[i] = (float)(generator.nextGaussian() * stddev + mean);
        }
    }
}