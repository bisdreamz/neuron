package dev.neuronic.net.math;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

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
        float range = max - min;

        if (Vectorization.isAvailable() && len >= Vectorization.getVectorLength()*2) {
            VectorSpecies<Float> species = Vectorization.getSpecies();
            int upper = Vectorization.loopBound(len);
            FloatVector minV   = FloatVector.broadcast(species, min);
            FloatVector rangeV = FloatVector.broadcast(species, range);
            float[] tmp = new float[species.length()];

            int i = 0;
            for (; i < upper; i += species.length()) {
                for (int j = 0; j < tmp.length; j++) tmp[j] = rnd.nextFloat();
                FloatVector.fromArray(species, tmp, 0)
                        .mul(rangeV)
                        .add(minV)
                        .intoArray(buffer, i);
            }
            for (; i < len; i++) {
                buffer[i] = rnd.nextFloat()*range + min;
            }
        } else {
            for (int i = 0; i < len; i++) {
                buffer[i] = rnd.nextFloat()*range + min;
            }
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