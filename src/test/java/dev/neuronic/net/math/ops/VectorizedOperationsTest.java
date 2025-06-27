package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the new vectorized operations used in Adam optimizers.
 */
class VectorizedOperationsTest {

    private float[] array1;
    private float[] array2;
    private float[] output;

    @BeforeEach
    void setUp() {
        array1 = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        array2 = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        output = new float[5];
    }

    @Test
    void testExponentialMovingAverageInPlace() {
        float[] current = {10.0f, 20.0f, 30.0f};
        float[] newValues = {1.0f, 2.0f, 3.0f};
        float decay = 0.9f;
        
        ExponentialMovingAverage.computeInPlace(current, newValues, decay);
        
        // Expected: decay * current + (1 - decay) * newValues
        // = 0.9 * 10 + 0.1 * 1 = 9.1
        assertEquals(9.1f, current[0], 1e-6f);
        assertEquals(18.2f, current[1], 1e-6f);
        assertEquals(27.3f, current[2], 1e-6f);
    }
    
    @Test
    void testExponentialMovingAverageToOutput() {
        float[] current = {10.0f, 20.0f, 30.0f};
        float[] newValues = {1.0f, 2.0f, 3.0f};
        float[] output = new float[3];
        float decay = 0.9f;
        
        ExponentialMovingAverage.compute(current, newValues, decay, output);
        
        // Original arrays should be unchanged
        assertEquals(10.0f, current[0], 1e-6f);
        assertEquals(1.0f, newValues[0], 1e-6f);
        
        // Output should have EMA values
        assertEquals(9.1f, output[0], 1e-6f);
        assertEquals(18.2f, output[1], 1e-6f);
        assertEquals(27.3f, output[2], 1e-6f);
    }

    @Test
    void testElementwiseDivide() {
        float[] numerator = {10.0f, 20.0f, 30.0f};
        float[] denominator = {2.0f, 4.0f, 5.0f};
        float[] output = new float[3];
        
        ElementwiseDivide.compute(numerator, denominator, output);
        
        assertEquals(5.0f, output[0], 1e-6f);
        assertEquals(5.0f, output[1], 1e-6f);
        assertEquals(6.0f, output[2], 1e-6f);
    }

    @Test
    void testFusedMultiplyDivideSubtract() {
        float[] params = {10.0f, 20.0f, 30.0f};
        float[] numerator = {2.0f, 4.0f, 6.0f};
        float[] denominator = {1.0f, 2.0f, 3.0f};
        float scale = 0.1f;
        
        FusedMultiplyDivideSubtract.compute(params, numerator, denominator, scale);
        
        // Expected: params -= scale * (numerator / denominator)
        // params[0] = 10 - 0.1 * (2/1) = 10 - 0.2 = 9.8
        assertEquals(9.8f, params[0], 1e-6f);
        assertEquals(19.8f, params[1], 1e-6f);
        assertEquals(29.8f, params[2], 1e-6f);
    }
    
    @Test
    void testFusedMultiplyDivideAdd() {
        float[] params = {10.0f, 20.0f, 30.0f};
        float[] numerator = {2.0f, 4.0f, 6.0f};
        float[] denominator = {1.0f, 2.0f, 3.0f};
        float scale = 0.1f;
        
        FusedMultiplyDivideSubtract.computeAdd(params, numerator, denominator, scale);
        
        // Expected: params += scale * (numerator / denominator)
        // params[0] = 10 + 0.1 * (2/1) = 10 + 0.2 = 10.2
        assertEquals(10.2f, params[0], 1e-6f);
        assertEquals(20.2f, params[1], 1e-6f);
        assertEquals(30.2f, params[2], 1e-6f);
    }

    @Test
    void testLargeArrayVectorization() {
        // Test with large array to trigger vectorization
        int size = 1000;
        float[] current = new float[size];
        float[] newValues = new float[size];
        
        for (int i = 0; i < size; i++) {
            current[i] = i * 0.1f;
            newValues[i] = i * 0.01f;
        }
        
        float decay = 0.9f;
        ExponentialMovingAverage.computeInPlace(current, newValues, decay);
        
        // Verify a few values
        assertEquals(0.9f * 0.0f + 0.1f * 0.0f, current[0], 1e-6f);
        assertEquals(0.9f * 0.1f + 0.1f * 0.01f, current[1], 1e-6f);
        assertEquals(0.9f * 0.2f + 0.1f * 0.02f, current[2], 1e-6f);
    }
    
    @Test
    void testVectorizedVsScalarConsistency() {
        // Test that vectorized and scalar paths produce identical results
        float[] current1 = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] current2 = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] newValues1 = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] newValues2 = {0.1f, 0.2f, 0.3f, 0.4f};
        float decay = 0.95f;
        
        // Force scalar computation (small array)
        ExponentialMovingAverage.computeScalarInPlace(current1, newValues1, decay);
        
        // Force vectorized computation
        ExponentialMovingAverage.computeVectorizedInPlace(current2, newValues2, decay);
        
        // Results should be identical
        assertArrayEquals(current1, current2, 1e-6f, 
            "Vectorized and scalar results should be identical");
    }
    
    @Test
    void testErrorHandling() {
        float[] short_array = {1.0f, 2.0f};
        float[] long_array = {1.0f, 2.0f, 3.0f};
        
        // Test array length mismatch
        assertThrows(IllegalArgumentException.class, () -> {
            ExponentialMovingAverage.computeInPlace(short_array, long_array, 0.9f);
        }, "Should throw exception for mismatched array lengths");
        
        assertThrows(IllegalArgumentException.class, () -> {
            ElementwiseDivide.compute(short_array, long_array, new float[2]);
        }, "Should throw exception for mismatched array lengths");
        
        assertThrows(IllegalArgumentException.class, () -> {
            FusedMultiplyDivideSubtract.compute(short_array, long_array, new float[2], 0.1f);
        }, "Should throw exception for mismatched array lengths");
    }
}