package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ColumnMajorPreActivationsTest {
    
    private static final float DELTA = 1e-6f;
    
    @Test
    void testBasicPreActivations() {
        float[] inputs = {1.0f, 2.0f};
        float[][] weights = {{0.5f, 1.0f}, {1.5f, 2.0f}}; // [inputs][neurons]
        float[] biases = {0.1f, 0.2f};
        float[] output = new float[2];
        
        ColumnMajorPreActivations.compute(inputs, weights, biases, output);
        
        // neuron 0: input[0]*weight[0][0] + input[1]*weight[1][0] + bias[0] = 1*0.5 + 2*1.5 + 0.1 = 3.6
        // neuron 1: input[0]*weight[0][1] + input[1]*weight[1][1] + bias[1] = 1*1.0 + 2*2.0 + 0.2 = 5.2
        assertArrayEquals(new float[]{3.6f, 5.2f}, output, DELTA);
    }
    
    @Test
    void testScalarImplementation() {
        float[] inputs = {2.0f, 3.0f, 1.0f};
        float[][] weights = {{1.0f, 2.0f}, {0.5f, 1.5f}, {2.0f, 0.5f}};
        float[] output = new float[2];
        
        ColumnMajorPreActivations.computeScalar(inputs, weights, output);
        
        // neuron 0: 2*1.0 + 3*0.5 + 1*2.0 = 2 + 1.5 + 2 = 5.5
        // neuron 1: 2*2.0 + 3*1.5 + 1*0.5 = 4 + 4.5 + 0.5 = 9.0
        assertArrayEquals(new float[]{5.5f, 9.0f}, output, DELTA);
    }
    
    @Test
    void testVectorizedImplementation() {
        float[] inputs = {1.0f, 2.0f, 3.0f, 4.0f};
        float[][] weights = {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
        float[] output = new float[8];
        
        ColumnMajorPreActivations.computeVectorized(inputs, weights, output);
        
        // All neurons should have: 1+2+3+4 = 10
        assertArrayEquals(new float[]{10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f}, output, DELTA);
    }
    
    @Test
    void testScalarVsVectorizedConsistency() {
        float[] inputs = {1.5f, 2.5f, 3.5f, 4.5f};
        float[][] weights = {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
                            {0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
                            {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f},
                            {0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f}};
        float[] output1 = new float[8];
        float[] output2 = new float[8];
        
        ColumnMajorPreActivations.computeScalar(inputs, weights, output1);
        ColumnMajorPreActivations.computeVectorized(inputs, weights, output2);
        
        assertArrayEquals(output1, output2, DELTA);
    }
    
    @Test
    void testZeroBiases() {
        float[] inputs = {1.0f, 2.0f};
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases = {0.0f, 0.0f};
        float[] output = new float[2];
        
        ColumnMajorPreActivations.compute(inputs, weights, biases, output);
        
        // neuron 0: 1*1 + 2*3 = 7
        // neuron 1: 1*2 + 2*4 = 10
        assertArrayEquals(new float[]{7.0f, 10.0f}, output, DELTA);
    }
    
    @Test
    void testZeroInputs() {
        float[] inputs = {0.0f, 0.0f, 0.0f};
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        float[] biases = {1.0f, 2.0f};
        float[] output = new float[2];
        
        ColumnMajorPreActivations.compute(inputs, weights, biases, output);
        
        // Should just return biases
        assertArrayEquals(biases, output, DELTA);
    }
    
    @Test
    void testNegativeValues() {
        float[] inputs = {-1.0f, 2.0f};
        float[][] weights = {{1.0f, -2.0f}, {-3.0f, 4.0f}};
        float[] biases = {0.5f, -0.5f};
        float[] output = new float[2];
        
        ColumnMajorPreActivations.compute(inputs, weights, biases, output);
        
        // neuron 0: -1*1 + 2*(-3) + 0.5 = -1 - 6 + 0.5 = -6.5
        // neuron 1: -1*(-2) + 2*4 + (-0.5) = 2 + 8 - 0.5 = 9.5
        assertArrayEquals(new float[]{-6.5f, 9.5f}, output, DELTA);
    }
    
    @Test
    void testSingleNeuron() {
        float[] inputs = {2.0f, 3.0f, 4.0f};
        float[][] weights = {{0.5f}, {1.0f}, {1.5f}};
        float[] biases = {2.0f};
        float[] output = new float[1];
        
        ColumnMajorPreActivations.compute(inputs, weights, biases, output);
        
        // 2*0.5 + 3*1.0 + 4*1.5 + 2.0 = 1 + 3 + 6 + 2 = 12
        assertEquals(12.0f, output[0], DELTA);
    }
    
    @Test
    void testInputsDimensionMismatch() {
        float[] inputs = {1.0f, 2.0f};
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}}; // 3 inputs expected
        float[] biases = {0.1f, 0.2f};
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            ColumnMajorPreActivations.compute(inputs, weights, biases, output));
    }
    
    @Test
    void testOutputDimensionMismatch() {
        float[] inputs = {1.0f, 2.0f};
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases = {0.1f, 0.2f};
        float[] output = new float[3]; // Wrong size
        
        assertThrows(IllegalArgumentException.class, () -> 
            ColumnMajorPreActivations.compute(inputs, weights, biases, output));
    }
    
    @Test
    void testBiasesDimensionMismatch() {
        float[] inputs = {1.0f, 2.0f};
        float[][] weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[] biases = {0.1f}; // Wrong size
        float[] output = new float[2];
        
        assertThrows(IllegalArgumentException.class, () -> 
            ColumnMajorPreActivations.compute(inputs, weights, biases, output));
    }
}