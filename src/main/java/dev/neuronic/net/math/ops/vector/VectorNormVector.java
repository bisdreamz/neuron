package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.VectorNorm;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * Vector implementation of VectorNorm.
 * This class is only loaded when Vector API is available.
 */
public final class VectorNormVector implements VectorNorm.Impl {
    
    @Override
    public float computeL2Squared(float[] vector) {
        if (!Vectorization.shouldVectorize(vector.length)) {
            return scalarComputeL2Squared(vector);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector sum = FloatVector.zero(species);
        int i = 0;
        int upperBound = Vectorization.loopBound(vector.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, vector, i);
            sum = v.fma(v, sum); // sum += v * v
        }
        
        float result = sum.reduceLanes(VectorOperators.ADD);
        
        for (; i < vector.length; i++) {
            float value = vector[i];
            result += value * value;
        }
        
        return result;
    }
    
    @Override
    public float computeL1(float[] vector) {
        if (!Vectorization.shouldVectorize(vector.length)) {
            return scalarComputeL1(vector);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector sum = FloatVector.zero(species);
        int i = 0;
        int upperBound = Vectorization.loopBound(vector.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, vector, i);
            sum = v.abs().add(sum);
        }
        
        float result = sum.reduceLanes(VectorOperators.ADD);
        
        for (; i < vector.length; i++) {
            result += Math.abs(vector[i]);
        }
        
        return result;
    }
    
    private float scalarComputeL2Squared(float[] vector) {
        float sum = 0.0f;
        for (float value : vector) {
            sum += value * value;
        }
        return sum;
    }
    
    private float scalarComputeL1(float[] vector) {
        float sum = 0.0f;
        for (float value : vector) {
            sum += Math.abs(value);
        }
        return sum;
    }
}