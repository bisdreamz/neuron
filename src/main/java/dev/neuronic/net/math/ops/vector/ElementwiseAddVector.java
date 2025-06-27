package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ElementwiseAdd;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ElementwiseAdd.
 * This class is only loaded when Vector API is available.
 */
public final class ElementwiseAddVector implements ElementwiseAdd.Impl {
    
    @Override
    public void compute(float[] a, float[] b, float[] output) {
        // Check if vectorization is worthwhile for this array size
        if (!Vectorization.shouldVectorizeLimited(a.length, 32)) {
            // Fall back to scalar for small arrays
            scalarCompute(a, b, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int i = 0;
        int upperBound = Vectorization.loopBound(a.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            va.add(vb).intoArray(output, i);
        }
        
        // Handle remaining elements
        for (; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }
    }
    
    private void scalarCompute(float[] a, float[] b, float[] output) {
        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }
    }
}