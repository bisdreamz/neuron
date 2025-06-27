package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ElementwiseMultiply;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ElementwiseMultiply.
 * This class is only loaded when Vector API is available.
 */
public final class ElementwiseMultiplyVector implements ElementwiseMultiply.Impl {
    
    @Override
    public void compute(float[] a, float[] b, float[] output) {
        // Note: original code mentioned vectorization only benefits small arrays
        if (!Vectorization.shouldVectorizeLimited(a.length)) {
            scalarCompute(a, b, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int i = 0;
        int upperBound = Vectorization.loopBound(a.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            va.mul(vb).intoArray(output, i);
        }
        
        for (; i < a.length; i++) {
            output[i] = a[i] * b[i];
        }
    }
    
    private void scalarCompute(float[] a, float[] b, float[] output) {
        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] * b[i];
        }
    }
}