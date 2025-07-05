package dev.neuronic.net.math.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import dev.neuronic.net.math.Vectorization;

import java.util.random.RandomGenerator;

/**
 * Vectorized implementation of FastRandom operations.
 * This class is loaded dynamically only when Vector API is available.
 */
public final class FastRandomVector {
    
    /**
     * Vectorized implementation of fillUniform.
     * Fills buffer with uniform floats in [min, max).
     */
    public static void fillUniform(float[] buffer, float min, float max, RandomGenerator rnd) {
        int len = buffer.length;
        float range = max - min;
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upper = Vectorization.loopBound(len);
        FloatVector minV = FloatVector.broadcast(species, min);
        FloatVector rangeV = FloatVector.broadcast(species, range);
        float[] tmp = new float[species.length()];
        
        int i = 0;
        for (; i < upper; i += species.length()) {
            // Generate random values
            for (int j = 0; j < tmp.length; j++) {
                tmp[j] = rnd.nextFloat();
            }
            // Apply vectorized transformation: value * range + min
            FloatVector.fromArray(species, tmp, 0)
                    .mul(rangeV)
                    .add(minV)
                    .intoArray(buffer, i);
        }
        
        // Handle remaining elements
        for (; i < len; i++) {
            buffer[i] = rnd.nextFloat() * range + min;
        }
    }
}