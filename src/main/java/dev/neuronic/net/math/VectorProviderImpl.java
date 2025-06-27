package dev.neuronic.net.math;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector API implementation of VectorProvider.
 * This class contains all Vector API dependencies and is only loaded when available.
 */
final class VectorProviderImpl implements VectorProvider {
    
    private static final VectorProviderImpl INSTANCE = new VectorProviderImpl();
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    static VectorProviderImpl getInstance() {
        return INSTANCE;
    }
    
    @Override
    public boolean isAvailable() {
        return true;
    }
    
    @Override
    public int getVectorLength() {
        return SPECIES.length();
    }
    
    @Override
    public int getLoopBound(int length) {
        return SPECIES.loopBound(length);
    }
    
    @Override
    public float dotProduct(float[] a, float[] b) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        FloatVector vsum = FloatVector.zero(SPECIES);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            vsum = va.fma(vb, vsum);
        }
        
        float sum = vsum.reduceLanes(VectorOperators.ADD);
        
        // Handle remainder
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    @Override
    public void elementwiseAdd(float[] a, float[] b, float[] output) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(output, i);
        }
        
        // Handle remainder
        for (; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }
    }
    
    @Override
    public void elementwiseMultiply(float[] a, float[] b, float[] output) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.mul(vb).intoArray(output, i);
        }
        
        // Handle remainder
        for (; i < a.length; i++) {
            output[i] = a[i] * b[i];
        }
    }
    
    private VectorProviderImpl() {} // Singleton
}