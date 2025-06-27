package dev.neuronic.net;

import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test new prediction methods in NeuralNet.
 */
class ArgMaxTest {

    @Test
    void testPredictArgmax() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create a simple 3-class classifier
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(3, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        float[] input = {1.0f, -1.0f};
        
        // Test predictArgmax returns single class index
        float classIndex = net.predictArgmax(input);
        
        // Should be a valid class index (0, 1, or 2)
        assertTrue(classIndex >= 0 && classIndex < 3);
        assertEquals((int) classIndex, classIndex); // Should be whole number
    }
    
    @Test
    void testPredictTopK() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create classifier with 5 classes
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(4, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(5, optimizer));
        
        float[] input = {1.0f, -1.0f};
        
        // Test getting top 3 classes
        float[] top3 = net.predictTopK(input, 3);
        
        assertEquals(3, top3.length);
        
        // All should be valid class indices
        for (float idx : top3) {
            assertTrue(idx >= 0 && idx < 5);
            assertEquals((int) idx, idx); // Should be whole numbers
        }
        
        // Should be unique indices
        assertTrue(top3[0] != top3[1]);
        assertTrue(top3[1] != top3[2]);
        assertTrue(top3[0] != top3[2]);
    }
    
    @Test
    void testPredictWithTemperature() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create classifier
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(3, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        float[] input = {1.0f, -1.0f};
        
        // Test temperature sampling
        float sampled = net.predictWithTemperature(input, 1.0f);
        
        // Should be a valid class index
        assertTrue(sampled >= 0 && sampled < 3);
        assertEquals((int) sampled, sampled); // Should be whole number
    }
    
    @Test
    void testPredictSampleTopK() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create classifier with 10 classes
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(5, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
        
        float[] input = {1.0f, -1.0f};
        
        // Sample from top 5 with temperature
        float sampled = net.predictSampleTopK(input, 5, 0.8f);
        
        // Should be a valid class index
        assertTrue(sampled >= 0 && sampled < 10);
        assertEquals((int) sampled, sampled); // Should be whole number
    }
    
    @Test
    void testPredictSampleTopP() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create classifier
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(5, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
        
        float[] input = {1.0f, -1.0f};
        
        // Sample from nucleus (top-p)
        float sampled = net.predictSampleTopP(input, 0.9f, 1.0f);
        
        // Should be a valid class index
        assertTrue(sampled >= 0 && sampled < 10);
        assertEquals((int) sampled, sampled); // Should be whole number
    }
    
    @Test
    void testPredictRawStillWorks() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f);
        
        // Create classifier
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .layer(Layers.hiddenDenseRelu(3, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        float[] input = {1.0f, -1.0f};
        float[] probabilities = net.predict(input);
        
        // Should return raw probabilities (3 classes)
        assertEquals(3, probabilities.length);
        
        // Should sum to approximately 1.0 (softmax)
        float sum = 0;
        for (float p : probabilities) {
            assertTrue(p >= 0 && p <= 1);
            sum += p;
        }
        assertEquals(1.0f, sum, 0.01f);
    }
}