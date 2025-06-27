package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for BaseLayerSpec functionality including default optimizer support
 * and learning rate ratio scaling.
 */
public class BaseLayerSpecTest {
    
    @Test
    public void testDefaultOptimizerSupport() {
        // Create a network with default optimizer
        Optimizer defaultOptimizer = new SgdOptimizer(0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(defaultOptimizer)
            .layer(Layers.hiddenDenseRelu(20))  // Should use default optimizer
            .layer(Layers.hiddenDenseRelu(15))  // Should use default optimizer
            .output(Layers.outputSoftmaxCrossEntropy(5));  // Should use default optimizer
        
        // Test forward pass works
        float[] input = new float[10];
        for (int i = 0; i < input.length; i++) {
            input[i] = i * 0.1f;
        }
        
        float[] output = net.predict(input);
        assertEquals(5, output.length);
        
        // Verify probabilities sum to 1 (softmax property)
        float sum = 0;
        for (float prob : output) {
            sum += prob;
        }
        assertEquals(1.0f, sum, 0.001f);
    }
    
    @Test
    public void testMixedOptimizers() {
        // Create a network with mixed optimizers
        Optimizer defaultOptimizer = new SgdOptimizer(0.01f);
        Optimizer customOptimizer = new SgdOptimizer(0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(defaultOptimizer)
            .layer(Layers.hiddenDenseRelu(20))  // Uses default
            .layer(Layers.hiddenDenseRelu(15, customOptimizer))  // Uses custom
            .output(Layers.outputSoftmaxCrossEntropy(5));  // Uses default
        
        // Test forward pass
        float[] input = new float[10];
        float[] output = net.predict(input);
        assertEquals(5, output.length);
    }
    
    @Test
    public void testLearningRateRatio() {
        // Create a network with custom learning rate ratios
        Optimizer defaultOptimizer = new SgdOptimizer(0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(784)
            .setDefaultOptimizer(defaultOptimizer)
            .layer(Layers.hiddenDenseRelu(256, 0.1))  // 10x slower learning
            .layer(Layers.hiddenDenseRelu(128, 2.0))  // 2x faster learning
            .output(Layers.outputSoftmaxCrossEntropy(10));
        
        // Test forward pass
        float[] input = new float[784];
        float[] output = net.predict(input);
        assertEquals(10, output.length);
    }
    
    @Test
    public void testNoDefaultOptimizerWithExplicitOptimizers() {
        // Should work without default optimizer if all layers specify their own
        Optimizer optimizer = new SgdOptimizer(0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            // No default optimizer set
            .layer(Layers.hiddenDenseRelu(20, optimizer))
            .layer(Layers.hiddenDenseRelu(15, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(5, optimizer));
        
        // Test forward pass
        float[] input = new float[10];
        float[] output = net.predict(input);
        assertEquals(5, output.length);
    }
    
    @Test
    public void testTrainingWithDefaultOptimizer() {
        // Test that training works with default optimizer
        Optimizer defaultOptimizer = new SgdOptimizer(0.1f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(defaultOptimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        // Simple XOR-like data
        float[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        float[][] targets = {
            {1, 0},  // Class 0
            {0, 1},  // Class 1
            {0, 1},  // Class 1
            {1, 0}   // Class 0
        };
        
        // Train for a few iterations
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.train(inputs[i], targets[i]);
            }
        }
        
        // Verify the network still works after training
        float[] result = net.predict(inputs[0]);
        assertNotNull(result);
        assertEquals(2, result.length);
    }
    
}