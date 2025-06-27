package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DotProductTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicDotProduct() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {4.0f, 5.0f, 6.0f};
        
        float result = DotProduct.compute(a, b);
        
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assertEquals(32.0f, result, DELTA);
    }
    
    @Test
    void testScalarImplementation() {
        float[] a = {2.0f, 3.0f, 4.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        
        float result = DotProduct.computeScalar(a, b);
        
        // 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
        assertEquals(20.0f, result, DELTA);
    }
    
    @Test
    void testVectorizedImplementation() {
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        
        float result = DotProduct.computeVectorized(a, b);
        
        // Sum of 1+2+3+4+5+6+7+8 = 36
        assertEquals(36.0f, result, DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] a = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f};
        float[] b = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        
        float result1 = DotProduct.computeScalar(a, b);
        float result2 = DotProduct.computeVectorized(a, b);
        
        assertEquals(result1, result2, DELTA);
    }
    
    @Test
    void testZeroVectors() {
        float[] a = {0.0f, 0.0f, 0.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        
        float result = DotProduct.compute(a, b);
        
        assertEquals(0.0f, result, DELTA);
    }
    
    @Test
    void testNegativeValues() {
        float[] a = {1.0f, -2.0f, 3.0f};
        float[] b = {-1.0f, 2.0f, -3.0f};
        
        float result = DotProduct.compute(a, b);
        
        // 1*(-1) + (-2)*2 + 3*(-3) = -1 + (-4) + (-9) = -14
        assertEquals(-14.0f, result, DELTA);
    }
    
    @Test
    void testOrthogonalVectors() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {0.0f, 1.0f, 0.0f};
        
        float result = DotProduct.compute(a, b);
        
        assertEquals(0.0f, result, DELTA);
    }
    
    @Test
    void testUnitVectors() {
        float[] a = {1.0f, 0.0f, 0.0f};
        float[] b = {1.0f, 0.0f, 0.0f};
        
        float result = DotProduct.compute(a, b);
        
        assertEquals(1.0f, result, DELTA);
    }
    
    @Test
    void testSingleElement() {
        float[] a = {5.0f};
        float[] b = {3.0f};
        
        float result = DotProduct.compute(a, b);
        
        assertEquals(15.0f, result, DELTA);
    }
    
    @Test
    void testDimensionMismatch() {
        float[] a = {1.0f, 2.0f};
        float[] b = {1.0f, 2.0f, 3.0f};
        
        assertThrows(IllegalArgumentException.class, () -> 
            DotProduct.compute(a, b));
    }
    
    @Test
    void testLargeArrays() {
        int size = 1000;
        float[] a = new float[size];
        float[] b = new float[size];
        
        for (int i = 0; i < size; i++) {
            a[i] = 1.0f;
            b[i] = 2.0f;
        }
        
        float result = DotProduct.compute(a, b);
        
        assertEquals(2000.0f, result, DELTA);
    }
}