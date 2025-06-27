package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BiasInitTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testComputeBasic() {
        float[] biases = new float[5];
        float value = 0.1f;
        
        BiasInit.compute(biases, value);
        
        for (float bias : biases) {
            assertEquals(value, bias, DELTA);
        }
    }
    
    @Test
    void testComputeScalar() {
        float[] biases = new float[4];
        float value = -0.05f;
        
        BiasInit.computeScalar(biases, value);
        
        assertArrayEquals(new float[]{-0.05f, -0.05f, -0.05f, -0.05f}, biases, DELTA);
    }
    
    @Test
    void testComputeVectorized() {
        float[] biases = new float[16]; // Good size for vectorization
        float value = 0.25f;
        
        BiasInit.computeVectorized(biases, value);
        
        for (float bias : biases) {
            assertEquals(0.25f, bias, DELTA);
        }
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] biases1 = new float[12];
        float[] biases2 = new float[12];
        float value = 0.333f;
        
        BiasInit.computeScalar(biases1, value);
        BiasInit.computeVectorized(biases2, value);
        
        assertArrayEquals(biases1, biases2, DELTA);
    }
    
    @Test
    void testZeroValue() {
        float[] biases = new float[8];
        
        BiasInit.compute(biases, 0.0f);
        
        for (float bias : biases) {
            assertEquals(0.0f, bias, DELTA);
        }
    }
    
    @Test
    void testPositiveValue() {
        float[] biases = new float[6];
        float value = 1.5f;
        
        BiasInit.compute(biases, value);
        
        for (float bias : biases) {
            assertEquals(1.5f, bias, DELTA);
        }
    }
    
    @Test
    void testNegativeValue() {
        float[] biases = new float[3];
        float value = -2.7f;
        
        BiasInit.compute(biases, value);
        
        for (float bias : biases) {
            assertEquals(-2.7f, bias, DELTA);
        }
    }
    
    @Test
    void testSingleElement() {
        float[] biases = new float[1];
        float value = 0.42f;
        
        BiasInit.computeScalar(biases, value);
        assertEquals(0.42f, biases[0], DELTA);
        
        BiasInit.computeVectorized(biases, value);
        assertEquals(0.42f, biases[0], DELTA);
    }
    
    @Test
    void testLargeArray() {
        float[] biases = new float[1000];
        float value = 0.01f;
        
        BiasInit.computeVectorized(biases, value);
        
        // Check that vectorization handled remainder correctly
        for (float bias : biases) {
            assertEquals(0.01f, bias, DELTA);
        }
    }
    
    @Test
    void testSmallArray() {
        float[] biases = new float[3]; // Too small for vectorization
        float value = 0.7f;
        
        BiasInit.compute(biases, value); // Should use scalar path
        
        for (float bias : biases) {
            assertEquals(0.7f, bias, DELTA);
        }
    }
    
    @Test
    void testVectorizedRemainder() {
        // Test array size that's not perfectly divisible by vector length
        float[] biases = new float[13]; // Odd size to test remainder handling
        float value = 0.123f;
        
        BiasInit.computeVectorized(biases, value);
        
        for (float bias : biases) {
            assertEquals(0.123f, bias, DELTA);
        }
    }
    
    @Test
    void testExtremeValues() {
        float[] biases = new float[5];
        
        // Test very small value
        BiasInit.compute(biases, 1e-8f);
        for (float bias : biases) {
            assertEquals(1e-8f, bias, 1e-9f);
        }
        
        // Test very large value
        BiasInit.compute(biases, 1e6f);
        for (float bias : biases) {
            assertEquals(1e6f, bias, 1e-1f);
        }
    }
    
    @Test
    void testOverwriteExistingValues() {
        float[] biases = {1.0f, 2.0f, 3.0f, 4.0f};
        float value = 0.5f;
        
        BiasInit.compute(biases, value);
        
        // Should overwrite all existing values
        for (float bias : biases) {
            assertEquals(0.5f, bias, DELTA);
        }
    }
    
    @Test
    void testTypicalUsageMNIST() {
        // Test typical MNIST output layer bias initialization
        float[] outputBiases = new float[10]; // 10 classes
        
        BiasInit.compute(outputBiases, 0.0f); // Common to init to zero
        
        for (float bias : outputBiases) {
            assertEquals(0.0f, bias, DELTA);
        }
    }
    
    @Test
    void testTypicalUsageHiddenLayer() {
        // Test typical hidden layer bias initialization
        float[] hiddenBiases = new float[128]; // Common hidden layer size
        
        BiasInit.compute(hiddenBiases, 0.01f); // Small positive bias
        
        for (float bias : hiddenBiases) {
            assertEquals(0.01f, bias, DELTA);
        }
    }
}