package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of DotProduct.
 * This class contains Vector API imports and is only loaded when needed.
 */
public final class DotProductVector {
    
    /**
     * Compute dot product using Vector API.
     * Called via reflection from VectorDispatcher.
     */
    public static float compute(float[] a, float[] b) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int i = 0;
        int upperBound = Vectorization.loopBound(a.length);
        
        FloatVector vsum = FloatVector.zero(species);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            vsum = va.fma(vb, vsum);  // Fused multiply-add: va * vb + vsum
        }
        
        float sum = vsum.reduceLanes(VectorOperators.ADD);
        
        // Handle remaining elements not species aligned
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
            
        return sum;
    }
    
    private DotProductVector() {} // Prevent instantiation
}