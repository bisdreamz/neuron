package dev.neuronic;

import dev.neuronic.net.math.ops.DotProduct;
import dev.neuronic.net.math.ops.ElementwiseMultiply;
import dev.neuronic.net.math.Vectorization;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verification test to ensure both scalar and vector implementations 
 * are actually being tested and produce correct results.
 */
class VectorImplementationVerificationTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void verifyVectorApiIsAvailable() {
        System.out.println("Vector API available: " + Vectorization.isAvailable());
        System.out.println("Vector length: " + Vectorization.getVectorLength());
        System.out.println("Should vectorize array of 8: " + Vectorization.shouldVectorize(8));
        System.out.println("Should vectorize array of 2: " + Vectorization.shouldVectorize(2));
        
        assertTrue(Vectorization.isAvailable(), 
            "Vector API should be available during tests");
        assertTrue(Vectorization.getVectorLength() > 0, 
            "Vector length should be positive");
    }
    
    @Test 
    void verifyDotProductImplementations() {
        // Use array size that forces vectorization
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        
        // Expected: 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 + 5*0.5 + 6*0.6 + 7*0.7 + 8*0.8
        // = 0.1 + 0.4 + 0.9 + 1.6 + 2.5 + 3.6 + 4.9 + 6.4 = 20.4
        float expected = 20.4f;
        
        float scalarResult = DotProduct.computeScalar(a, b);
        float vectorResult = DotProduct.computeVectorized(a, b);
        float autoResult = DotProduct.compute(a, b);
        
        System.out.println("DotProduct - Scalar: " + scalarResult + 
                          ", Vector: " + vectorResult + 
                          ", Auto: " + autoResult);
        
        assertEquals(expected, scalarResult, DELTA, "Scalar implementation incorrect");
        assertEquals(expected, vectorResult, DELTA, "Vector implementation incorrect");
        assertEquals(expected, autoResult, DELTA, "Auto-selection incorrect");
        assertEquals(scalarResult, vectorResult, DELTA, "Scalar and vector implementations differ");
    }
    
    @Test
    void verifyElementwiseMultiplyImplementations() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        float[] scalarOutput = new float[8];
        float[] vectorOutput = new float[8];
        float[] autoOutput = new float[8];
        
        // Expected: {2, 6, 12, 20, 30, 42, 56, 72}
        float[] expected = {2.0f, 6.0f, 12.0f, 20.0f, 30.0f, 42.0f, 56.0f, 72.0f};
        
        ElementwiseMultiply.computeScalar(a, b, scalarOutput);
        ElementwiseMultiply.computeVectorized(a, b, vectorOutput);
        ElementwiseMultiply.compute(a, b, autoOutput);
        
        System.out.println("ElementwiseMultiply - Scalar[0-3]: [" + 
                          scalarOutput[0] + ", " + scalarOutput[1] + ", " + 
                          scalarOutput[2] + ", " + scalarOutput[3] + "]");
        System.out.println("ElementwiseMultiply - Vector[0-3]: [" + 
                          vectorOutput[0] + ", " + vectorOutput[1] + ", " + 
                          vectorOutput[2] + ", " + vectorOutput[3] + "]");
        
        assertArrayEquals(expected, scalarOutput, DELTA);
        assertArrayEquals(expected, vectorOutput, DELTA);
        assertArrayEquals(expected, autoOutput, DELTA);
        assertArrayEquals(scalarOutput, vectorOutput, DELTA);
    }
    
    @Test
    void verifySmallArraysUseScalar() {
        // Small arrays should use scalar even when vector API is available
        float[] a = {1.0f, 2.0f};  // Only 2 elements
        float[] b = {3.0f, 4.0f};
        
        boolean shouldVectorize = Vectorization.shouldVectorize(a.length);
        System.out.println("Should vectorize array of length " + a.length + ": " + shouldVectorize);
        
        // For very small arrays, vectorization overhead isn't worth it
        assertFalse(shouldVectorize, "Small arrays should not be vectorized");
        
        float result = DotProduct.compute(a, b);
        assertEquals(11.0f, result, DELTA); // 1*3 + 2*4 = 11
    }
}