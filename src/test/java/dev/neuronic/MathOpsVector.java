package dev.neuronic;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation WITH Vector API imports.
 * This class is only loaded if actually called.
 */
class MathOpsVector {
    
    static void add(float[] a, float[] b, float[] result) {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        int i = 0;
        int bound = species.loopBound(a.length);
        
        // Vectorized loop
        for (; i < bound; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            va.add(vb).intoArray(result, i);
        }
        
        // Scalar remainder
        for (; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
    }
}