package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ElementwiseSubtract;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ElementwiseSubtract.
 * This class is only loaded when Vector API is available.
 */
public final class ElementwiseSubtractVector implements ElementwiseSubtract.Impl {
    
    @Override
    public void compute(float[] a, float[] b, float[] output) {
        if (!Vectorization.shouldVectorizeLimited(a.length, 32)) {
            scalarCompute(a, b, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int i = 0;
        int upperBound = Vectorization.loopBound(a.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            va.sub(vb).intoArray(output, i);
        }
        
        for (; i < a.length; i++) {
            output[i] = a[i] - b[i];
        }
    }
    
    private void scalarCompute(float[] a, float[] b, float[] output) {
        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] - b[i];
        }
    }
}