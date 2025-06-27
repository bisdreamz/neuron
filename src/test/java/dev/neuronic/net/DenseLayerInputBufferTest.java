package dev.neuronic.net;

import dev.neuronic.net.activators.LinearActivator;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Test to verify that DenseLayer doesn't modify input arrays and properly
 * preserves LayerContext state. This ensures ThreadLocal buffers don't cause
 * contamination between forward passes.
 */
public class DenseLayerInputBufferTest {
    
    @Test
    public void testInputPreservation() {
        // Create a simple network with known behavior
        // We'll use a single dense layer and train it to produce predictable outputs
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
            .layer(Layers.hiddenDense(2, LinearActivator.INSTANCE, null))
            .output(Layers.outputLinearRegression(2));
        
        // Train the network to learn a simple linear transformation
        // After training, we expect:
        // [1, 0] -> [1.1, 2.2]
        // [0, 1] -> [3.1, 4.2]
        for (int i = 0; i < 500; i++) {
            net.train(new float[]{1.0f, 0.0f}, new float[]{1.1f, 2.2f});
            net.train(new float[]{0.0f, 1.0f}, new float[]{3.1f, 4.2f});
        }
        
        // Create original input arrays
        float[] originalInputA = {1.0f, 0.0f};
        float[] originalInputB = {0.0f, 1.0f};
        
        // Make copies to check if they get modified
        float[] inputA = originalInputA.clone();
        float[] inputB = originalInputB.clone();
        
        // Get predictions and save them
        float[] outputA = net.predict(inputA);
        float[] outputB = net.predict(inputB);
        
        // Verify inputs were not modified
        assertArrayEquals(originalInputA, inputA, "Input A should not be modified during prediction");
        assertArrayEquals(originalInputB, inputB, "Input B should not be modified during prediction");
        
        // Verify outputs are approximately what we trained for
        assertEquals(1.1f, outputA[0], 0.1f);
        assertEquals(2.2f, outputA[1], 0.1f);
        assertEquals(3.1f, outputB[0], 0.1f);
        assertEquals(4.2f, outputB[1], 0.1f);
        
        // Test that multiple predictions don't interfere with each other
        // This tests the ThreadLocal buffer isolation
        float[] output1 = net.predict(inputA);
        float[] output2 = net.predict(inputB);
        float[] output3 = net.predict(inputA);
        
        // First and third should be identical (same input)
        assertArrayEquals(output1, output3, 0.001f, "Repeated predictions with same input should yield same output");
        
        // Outputs should match what we got before
        assertArrayEquals(outputA, output1, 0.001f, "Output should be consistent");
        assertArrayEquals(outputB, output2, 0.001f, "Output should be consistent");
    }
    
    @Test
    public void testRepeatedCalls() {
        // Create a network and train it
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
            .layer(Layers.hiddenDense(3, LinearActivator.INSTANCE, null))
            .output(Layers.outputLinearRegression(2));
        
        // Train with some data
        for (int i = 0; i < 100; i++) {
            net.train(new float[]{1.0f, 0.0f}, new float[]{0.5f, 0.7f});
            net.train(new float[]{0.0f, 1.0f}, new float[]{0.3f, 0.9f});
        }
        
        // Test repeated calls with same input
        float[] input = {0.5f, 0.5f};
        
        float[] output1 = net.predict(input.clone());
        float[] output2 = net.predict(input.clone());
        float[] output3 = net.predict(input.clone());
        
        // All should be identical
        assertArrayEquals(output1, output2, 0.0001f, "Repeated calls should yield identical results");
        assertArrayEquals(output1, output3, 0.0001f, "Repeated calls should yield identical results");
        
        // Test that the original input is never modified
        float[] originalInput = {0.5f, 0.5f};
        assertArrayEquals(originalInput, input, "Input should never be modified");
    }
    
    @Test
    public void testConcurrentPredictions() {
        // Create and train a network
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.05f, 0.0f))
            .layer(Layers.hiddenDense(4, LinearActivator.INSTANCE, null))
            .output(Layers.outputLinearRegression(2));
        
        // Train with some patterns
        for (int i = 0; i < 200; i++) {
            net.train(new float[]{1.0f, 0.0f, 0.0f}, new float[]{1.0f, 0.0f});
            net.train(new float[]{0.0f, 1.0f, 0.0f}, new float[]{0.0f, 1.0f});
            net.train(new float[]{0.0f, 0.0f, 1.0f}, new float[]{0.5f, 0.5f});
        }
        
        // Test that concurrent predictions don't interfere
        // This simulates what might happen with ThreadLocal buffer reuse
        float[] input1 = {1.0f, 0.0f, 0.0f};
        float[] input2 = {0.0f, 1.0f, 0.0f};
        float[] input3 = {0.0f, 0.0f, 1.0f};
        
        // Get baseline predictions
        float[] baseline1 = net.predict(input1.clone());
        float[] baseline2 = net.predict(input2.clone());
        float[] baseline3 = net.predict(input3.clone());
        
        // Interleave predictions to test buffer isolation
        float[] test1a = net.predict(input1.clone());
        float[] test2a = net.predict(input2.clone());
        float[] test1b = net.predict(input1.clone());
        float[] test3a = net.predict(input3.clone());
        float[] test2b = net.predict(input2.clone());
        float[] test3b = net.predict(input3.clone());
        
        // All predictions for the same input should match
        assertArrayEquals(baseline1, test1a, 0.0001f, "Predictions should be consistent");
        assertArrayEquals(baseline1, test1b, 0.0001f, "Predictions should be consistent");
        assertArrayEquals(baseline2, test2a, 0.0001f, "Predictions should be consistent");
        assertArrayEquals(baseline2, test2b, 0.0001f, "Predictions should be consistent");
        assertArrayEquals(baseline3, test3a, 0.0001f, "Predictions should be consistent");
        assertArrayEquals(baseline3, test3b, 0.0001f, "Predictions should be consistent");
        
        // Verify inputs were never modified
        assertArrayEquals(new float[]{1.0f, 0.0f, 0.0f}, input1, "Input should not be modified");
        assertArrayEquals(new float[]{0.0f, 1.0f, 0.0f}, input2, "Input should not be modified");
        assertArrayEquals(new float[]{0.0f, 0.0f, 1.0f}, input3, "Input should not be modified");
    }
}