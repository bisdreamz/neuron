package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.GradientNorm;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * Vector implementation of GradientNorm.
 * This class is only loaded when Vector API is available.
 */
public final class GradientNormVector implements GradientNorm.Impl {
    
    @Override
    public float computeNormSquared(float[] array) {
        if (!Vectorization.shouldVectorize(array.length)) {
            return scalarComputeNormSquared(array);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int loopBound = Vectorization.loopBound(array.length);
        FloatVector sumVec = FloatVector.zero(species);
        
        // Vectorized main loop
        int i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vec = FloatVector.fromArray(species, array, i);
            sumVec = sumVec.add(vec.mul(vec));
        }
        
        // Reduce vector to scalar
        float sum = sumVec.reduceLanes(VectorOperators.ADD);
        
        // Handle remaining elements
        for (; i < array.length; i++) {
            sum += array[i] * array[i];
        }
        
        return sum;
    }
    
    private float scalarComputeNormSquared(float[] array) {
        float sum = 0.0f;
        for (float val : array) {
            sum += val * val;
        }
        return sum;
    }
}