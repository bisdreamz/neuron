package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ElementwiseScale;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ElementwiseScale.
 * This class is only loaded when Vector API is available.
 */
public final class ElementwiseScaleVector implements ElementwiseScale.Impl {
    
    @Override
    public void compute(float[] input, float scale, float[] output) {
        if (!Vectorization.shouldVectorize(input.length)) {
            scalarCompute(input, scale, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scaleVec = FloatVector.broadcast(species, scale);
        int upperBound = Vectorization.loopBound(input.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, input, i);
            v.mul(scaleVec).intoArray(output, i);
        }
        
        for (; i < input.length; i++) {
            output[i] = scale * input[i];
        }
    }
    
    @Override
    public void computeInPlace(float[] array, float scale) {
        if (!Vectorization.shouldVectorize(array.length)) {
            scalarComputeInPlace(array, scale);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scaleVec = FloatVector.broadcast(species, scale);
        int upperBound = Vectorization.loopBound(array.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, array, i);
            v.mul(scaleVec).intoArray(array, i);
        }
        
        for (; i < array.length; i++) {
            array[i] = scale * array[i];
        }
    }
    
    private void scalarCompute(float[] input, float scale, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = scale * input[i];
        }
    }
    
    private void scalarComputeInPlace(float[] array, float scale) {
        for (int i = 0; i < array.length; i++) {
            array[i] = scale * array[i];
        }
    }
}