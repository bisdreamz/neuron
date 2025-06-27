package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ElementwiseSquare;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ElementwiseSquare.
 * This class is only loaded when Vector API is available.
 */
public final class ElementwiseSquareVector implements ElementwiseSquare.Impl {
    
    @Override
    public void compute(float[] input, float[] output) {
        if (!Vectorization.shouldVectorize(input.length)) {
            scalarCompute(input, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int upperBound = Vectorization.loopBound(input.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector v = FloatVector.fromArray(species, input, i);
            v.mul(v).intoArray(output, i);
        }
        
        for (; i < input.length; i++) {
            output[i] = input[i] * input[i];
        }
    }
    
    private void scalarCompute(float[] input, float[] output) {
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] * input[i];
        }
    }
}