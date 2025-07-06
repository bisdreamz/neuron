package dev.neuronic.net.math;

import java.lang.reflect.Method;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Utility for fast, parallel-capable random float generation.
 * Provides both scalar “nextXxx” methods and vectorized bulk fills,
 * dispatching into the Vector API only when it’s actually available.
 */
public final class FastRandom {
    private FastRandom() {}

    // thread-local Xoroshiro128++ instance
    private static final ThreadLocal<RandomGenerator> RNG = ThreadLocal.withInitial(() ->
            RandomGeneratorFactory.of("Xoroshiro128PlusPlus").create()
    );
    
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

    /** Grab the thread-local PRNG if you need it directly. */
    public static RandomGenerator get() {
        return RNG.get();
    }

    /**
     * Fill buffer with uniform floats in [min, max).
     */
    public static void fillUniform(float[] buffer, float min, float max) {
        RandomGenerator rnd = RNG.get();
        int len = buffer.length;
        
        // Try vectorized implementation if available and worthwhile
        if (VECTOR_FILL_UNIFORM != null && len >= Vectorization.getVectorLength() * 2) {
            try {
                VECTOR_FILL_UNIFORM.invoke(null, buffer, min, max, rnd);
                return;
            } catch (Exception e) {
                // Fall back to scalar on any error
            }
        }
        
        // Scalar implementation
        float range = max - min;
        for (int i = 0; i < len; i++) {
            buffer[i] = rnd.nextFloat() * range + min;
        }
    }

    /** Fill buffer with uniform floats in (–limit, +limit). */
    public static void fillSymmetric(float[] buffer, float limit) {
        fillUniform(buffer, -limit, limit);
    }

    /**
     * Fill buffer with Gaussian-distributed floats (mean, stddev).
     * Purely scalar, since the Vector API has no native Gaussian sampler.
     */
    public static void fillGaussian(float[] buffer, float mean, float stddev) {
        RandomGenerator rnd = RNG.get();
        for (int i = 0, len = buffer.length; i < len; i++) {
            buffer[i] = (float)(rnd.nextGaussian()*stddev + mean);
        }
    }
}