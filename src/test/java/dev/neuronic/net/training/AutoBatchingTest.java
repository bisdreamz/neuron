package dev.neuronic.net.training;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetInt;
import dev.neuronic.net.*;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the auto-batching functionality in SimpleNet classes.
 */
class AutoBatchingTest {
    
    @Test
    void testAutoBatchingBasicFunctionality() {
        // Create a simple network
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Enable auto-batching
        classifier.withAutoBatching(4);
        assertEquals(4, classifier.getAutoBatchSize(), "Auto-batch size should be set");
        assertTrue(classifier.isAutoBatchingEnabled(), "Auto-batching should be enabled");
        
        // Train with samples - should buffer until batch size reached
        classifier.train(new float[]{1.0f, 0.0f}, 0);
        assertEquals(1, classifier.getBufferedSampleCount(), "Should have 1 buffered sample");
        
        classifier.train(new float[]{0.0f, 1.0f}, 1);
        assertEquals(2, classifier.getBufferedSampleCount(), "Should have 2 buffered samples");
        
        classifier.train(new float[]{1.0f, 1.0f}, 0);
        assertEquals(3, classifier.getBufferedSampleCount(), "Should have 3 buffered samples");
        
        // Fourth sample should trigger batch training
        classifier.train(new float[]{0.0f, 0.0f}, 1);
        assertEquals(0, classifier.getBufferedSampleCount(), "Buffer should be empty after batch training");
    }
    
    @Test
    void testFlushBatch() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(3))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net)
            .withAutoBatching(10);  // Large batch size
        
        // Add 3 samples
        classifier.train(new float[]{1.0f, 0.0f}, 0);
        classifier.train(new float[]{0.0f, 1.0f}, 1);
        classifier.train(new float[]{0.5f, 0.5f}, 0);
        
        assertEquals(3, classifier.getBufferedSampleCount(), "Should have 3 buffered samples");
        
        // Flush the partial batch
        int flushed = classifier.flushBatch();
        assertEquals(3, flushed, "Should have flushed 3 samples");
        assertEquals(0, classifier.getBufferedSampleCount(), "Buffer should be empty after flush");
    }
    
    @Test
    void testDisableAutoBatching() {
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
            .layer(Layers.hiddenDenseRelu(3))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Initially disabled
        assertFalse(classifier.isAutoBatchingEnabled(), "Auto-batching should be disabled by default");
        assertEquals(0, classifier.getAutoBatchSize(), "Default batch size should be 0");
        
        // Enable then disable
        classifier.withAutoBatching(16);
        assertTrue(classifier.isAutoBatchingEnabled(), "Should be enabled");
        
        // Add some samples
        classifier.train(new float[]{1.0f, 0.0f}, 0);
        classifier.train(new float[]{0.0f, 1.0f}, 1);
        assertEquals(2, classifier.getBufferedSampleCount(), "Should have buffered samples");
        
        // Disable - should flush remaining samples
        classifier.withoutAutoBatching();
        assertFalse(classifier.isAutoBatchingEnabled(), "Should be disabled");
        assertEquals(0, classifier.getBufferedSampleCount(), "Buffer should be flushed");
    }
    
    @Test
    void testMethodChaining() {
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
            .layer(Layers.hiddenDenseRelu(3))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        // Test method chaining
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net)
            .withAutoBatching(32);
        
        assertNotNull(classifier, "Method chaining should return classifier instance");
        assertEquals(32, classifier.getAutoBatchSize(), "Batch size should be set");
        
        // Chain disable
        SimpleNetInt same = classifier.withoutAutoBatching();
        assertSame(classifier, same, "Should return same instance");
        assertEquals(0, classifier.getAutoBatchSize(), "Should be disabled");
    }
    
    @Test
    void testConcurrentAutoBatching() throws InterruptedException {
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net)
            .withAutoBatching(10);
        
        // Train from multiple threads
        int numThreads = 4;
        int samplesPerThread = 25;
        Thread[] threads = new Thread[numThreads];
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            threads[t] = new Thread(() -> {
                Random rand = new Random(threadId);
                for (int i = 0; i < samplesPerThread; i++) {
                    float[] input = {rand.nextFloat(), rand.nextFloat()};
                    int label = rand.nextInt(2);
                    classifier.train(input, label);
                }
            });
        }
        
        // Start all threads
        for (Thread thread : threads) {
            thread.start();
        }
        
        // Wait for completion
        for (Thread thread : threads) {
            thread.join();
        }
        
        // Flush any remaining
        classifier.flushBatch();
        
        // Should have processed all samples without errors
        assertEquals(0, classifier.getBufferedSampleCount(), 
                    "All samples should be processed");
    }
    
    @Test
    void testAutoBatchingWithMixedFeatures() {
        // Test with mixed feature input
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 8),
                Feature.passthrough()
            ))
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net)
            .withAutoBatching(3);
        
        // Train with maps
        Map<String, Object> input1 = Map.of("feature_0", "token1", "feature_1", 0.5f);
        Map<String, Object> input2 = Map.of("feature_0", "token2", "feature_1", 0.7f);
        Map<String, Object> input3 = Map.of("feature_0", "token3", "feature_1", 0.3f);
        
        classifier.train(input1, 0);
        classifier.train(input2, 1);
        assertEquals(2, classifier.getBufferedSampleCount(), "Should buffer first 2");
        
        classifier.train(input3, 0);  // Should trigger batch
        assertEquals(0, classifier.getBufferedSampleCount(), "Should have trained batch");
        
        // Verify we can still predict
        int prediction = classifier.predictInt(input1);
        assertTrue(prediction >= 0, "Should make valid prediction");
    }
}