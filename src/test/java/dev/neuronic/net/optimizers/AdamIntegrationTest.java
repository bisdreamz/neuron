package dev.neuronic.net.optimizers;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for Adam optimizers with the neural network system.
 */
class AdamIntegrationTest {

    @Test
    void testAdamWithNeuralNetBuilder() {
        // Test that Adam optimizer works with the neural network builder
        AdamOptimizer adam = new AdamOptimizer(0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(adam)
            .input(10)
            .layer(Layers.hiddenDenseRelu(20))
            .layer(Layers.hiddenDenseRelu(15))
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        // Test forward pass
        float[] input = new float[10];
        for (int i = 0; i < input.length; i++) {
            input[i] = i * 0.1f;
        }
        
        float[] output = net.predict(input);
        assertNotNull(output);
        assertEquals(5, output.length);
        
        // Test training
        float[] targets = new float[]{0, 1, 0, 0, 0}; // One-hot encoded
        assertDoesNotThrow(() -> net.train(input, targets));
    }

    @Test
    void testAdamWWithNeuralNetBuilder() {
        // Test that AdamW optimizer works with the neural network builder
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(adamW)
            .input(5)
            .layer(Layers.hiddenDenseRelu(10))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        // Test forward pass
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] output = net.predict(input);
        assertNotNull(output);
        assertEquals(3, output.length);
        
        // Test training
        float[] targets = {1.0f, 0.0f, 0.0f}; // One-hot encoded
        assertDoesNotThrow(() -> net.train(input, targets));
    }

    @Test
    void testMixedOptimizers() {
        // Test using different optimizers for different layers
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(new AdamOptimizer(0.001f))
            .input(5)
            .layer(Layers.hiddenDenseRelu(10)
                .optimizer(new AdamWOptimizer(0.001f, 0.01f))) // Override with AdamW
            .layer(Layers.hiddenDenseRelu(8))  // Uses default Adam
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        // Test that it works
        float[] input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        float[] output = net.predict(input);
        assertNotNull(output);
        assertEquals(3, output.length);
        
        // Test training
        float[] targets = {0.0f, 1.0f, 0.0f};
        assertDoesNotThrow(() -> net.train(input, targets));
    }

    @Test
    void testLearningRateRatioWithAdam() {
        // Test that learning rate ratio works with Adam
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(new AdamOptimizer(0.001f))
            .input(5)
            .layer(Layers.hiddenDenseRelu(10)
                .learningRateRatio(0.1f)) // 10x slower learning
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        // Test that it works
        float[] input = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float[] output = net.predict(input);
        assertNotNull(output);
        assertEquals(3, output.length);
        
        // Test training works with scaled learning rate
        float[] targets = {1.0f, 0.0f, 0.0f};
        assertDoesNotThrow(() -> net.train(input, targets));
    }

    @Test
    void testTrainingConvergence() {
        // Simple convergence test with Adam on a learnable problem
        AdamOptimizer adam = new AdamOptimizer(0.01f); // Higher learning rate for faster convergence
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(adam)
            .input(2)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        // Simple XOR-like problem data
        float[][] inputs = {
            {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
        };
        float[][] targets = {
            {1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}
        };
        
        // Train for a few epochs
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.train(inputs[i], targets[i]);
            }
        }
        
        // Network should still be functional (not NaN)
        float[] result = net.predict(inputs[0]);
        assertNotNull(result);
        assertEquals(2, result.length);
        assertFalse(Float.isNaN(result[0]), "Output should not be NaN");
        assertFalse(Float.isNaN(result[1]), "Output should not be NaN");
    }
}