package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.FusedBiasCorrectScale;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized implementation of FusedBiasCorrectScale.
 * This class is only loaded when Vector API is available.
 */
public final class FusedBiasCorrectScaleVector implements FusedBiasCorrectScale.Impl {
    
    @Override
    public void compute(float[] input, float scale, float correction, float[] output) {
        if (Vectorization.shouldVectorize(input.length)) {
            computeVectorized(input, scale, correction, output);
        } else {
            computeScalar(input, scale, correction, output);
        }
    }
    
    @Override
    public void computeInPlace(float[] array, float scale, float correction) {
        compute(array, scale, correction, array);
    }
    
    private void computeVectorized(float[] input, float scale, float correction, float[] output) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int len = input.length;
        int upper = Vectorization.loopBound(len);
        
        // Pre-compute the combined factor
        float scaleDivCorrection = scale / correction;
        FloatVector factorV = FloatVector.broadcast(species, scaleDivCorrection);
        
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector inputV = FloatVector.fromArray(species, input, i);
            FloatVector resultV = inputV.mul(factorV);
            resultV.intoArray(output, i);
        }
        
        // Handle remaining elements
        for (; i < len; i++) {
            output[i] = input[i] * scaleDivCorrection;
        }
    }
    
    private void computeScalar(float[] input, float scale, float correction, float[] output) {
        float scaleDivCorrection = scale / correction;
        
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] * scaleDivCorrection;
        }
    }
}