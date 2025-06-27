package dev.neuronic.net.math.ops;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WeightGradientsColumnMajorTest {

    @Test
    void testBasicComputation() {
        float[] inputs = {1.0f, 2.0f, 3.0f};
        float[] neuronDeltas = {0.5f, -0.8f};
        float[][] weightGradients = new float[3][2];
        
        WeightGradientsColumnMajor.compute(inputs, neuronDeltas, weightGradients);
        
        // Expected: gradients[input][neuron] = input * delta[neuron]
        assertEquals(1.0f * 0.5f, weightGradients[0][0], 1e-6f);
        assertEquals(1.0f * -0.8f, weightGradients[0][1], 1e-6f);
        assertEquals(2.0f * 0.5f, weightGradients[1][0], 1e-6f);
        assertEquals(2.0f * -0.8f, weightGradients[1][1], 1e-6f);
        assertEquals(3.0f * 0.5f, weightGradients[2][0], 1e-6f);
        assertEquals(3.0f * -0.8f, weightGradients[2][1], 1e-6f);
    }
    
    @Test
    void testZeroInputs() {
        float[] inputs = {0.0f, 0.0f};
        float[] neuronDeltas = {1.0f, -1.0f};
        float[][] weightGradients = new float[2][2];
        
        WeightGradientsColumnMajor.compute(inputs, neuronDeltas, weightGradients);
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0.0f, weightGradients[i][j], 1e-6f);
            }
        }
    }
    
    @Test
    void testZeroDeltas() {
        float[] inputs = {1.0f, 2.0f};
        float[] neuronDeltas = {0.0f, 0.0f};
        float[][] weightGradients = new float[2][2];
        
        WeightGradientsColumnMajor.compute(inputs, neuronDeltas, weightGradients);
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0.0f, weightGradients[i][j], 1e-6f);
            }
        }
    }
    
    @Test
    void testLargeArrays() {
        // Test with larger arrays to trigger vectorization
        int numInputs = 64;
        int numNeurons = 32;
        
        float[] inputs = new float[numInputs];
        float[] neuronDeltas = new float[numNeurons];
        float[][] weightGradients = new float[numInputs][numNeurons];
        
        // Initialize with some pattern
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = (i + 1) * 0.1f;
        }
        for (int j = 0; j < numNeurons; j++) {
            neuronDeltas[j] = (j + 1) * 0.01f;
        }
        
        WeightGradientsColumnMajor.compute(inputs, neuronDeltas, weightGradients);
        
        // Verify a few random spots
        assertEquals(inputs[0] * neuronDeltas[0], weightGradients[0][0], 1e-6f);
        assertEquals(inputs[10] * neuronDeltas[5], weightGradients[10][5], 1e-6f);
        assertEquals(inputs[numInputs-1] * neuronDeltas[numNeurons-1], 
                    weightGradients[numInputs-1][numNeurons-1], 1e-6f);
    }
    
    @Test
    void testSingleElement() {
        float[] inputs = {2.5f};
        float[] neuronDeltas = {-0.3f};
        float[][] weightGradients = new float[1][1];
        
        WeightGradientsColumnMajor.compute(inputs, neuronDeltas, weightGradients);
        
        assertEquals(2.5f * -0.3f, weightGradients[0][0], 1e-6f);
    }
}