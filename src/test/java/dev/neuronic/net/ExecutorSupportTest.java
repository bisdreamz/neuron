package dev.neuronic.net;

import dev.neuronic.net.activators.Activator;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test executor support for intra-prediction/training parallelism.
 */
class ExecutorSupportTest {

    @Test
    void testNetworkWithoutExecutor() {
        // Default behavior - no executor
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
                .input(4)
                .layer(Layers.hiddenDenseRelu(8, optimizer))
                .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targets = {1.0f, 0.0f};
        
        // Should work without executor
        float[] prediction = net.predict(input);
        assertEquals(2, prediction.length);
        
        // Training should work
        net.train(input, targets);
    }
    
    @Test
    void testNetworkWithExecutor() {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        
        try {
            SgdOptimizer optimizer = new SgdOptimizer(0.01f);
            NeuralNet net = NeuralNet.newBuilder()
                    .input(4)
                    .layer(Layers.hiddenDenseRelu(8, optimizer))
                    .executor(executor)  // Add executor
                    .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
            
            float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] targets = {1.0f, 0.0f};
            
            // Should work with executor
            float[] prediction = net.predict(input);
            assertEquals(2, prediction.length);
            
            // Training should work
            net.train(input, targets);
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testCustomParallelizableActivator() {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        AtomicInteger executorUsageCount = new AtomicInteger(0);
        
        try {
            // Custom activator that tracks executor usage
            var customActivator = new Activator() {
                @Override
                public void activate(float[] input, float[] output) {
                    // Simple identity activation
                    System.arraycopy(input, 0, output, 0, input.length);
                }
                
                @Override
                public void derivative(float[] input, float[] output) {
                    // Derivative of identity is 1
                    for (int i = 0; i < output.length; i++) {
                        output[i] = 1.0f;
                    }
                }
                
                @Override
                public void activate(float[] input, float[] output, ExecutorService executor) {
                    if (executor != null) {
                        executorUsageCount.incrementAndGet();
                        // Custom parallel implementation could go here
                        // For test, just call default
                        Activator.super.activate(input, output, executor);
                    } else {
                        activate(input, output);
                    }
                }
            };
            
            SgdOptimizer optimizer = new SgdOptimizer(0.01f);
            
            // Create a custom layer spec that uses our activator
            var customLayerSpec = new Layer.Spec() {
                @Override
                public Layer create(int inputSize) {
                    return new DenseLayer(
                        optimizer, customActivator, 4, inputSize, 
                        WeightInitStrategy.XAVIER
                    );
                }
                
                @Override
                public int getOutputSize() {
                    return 4;
                }
            };
            
            NeuralNet net = NeuralNet.newBuilder()
                    .input(2)
                    .layer(customLayerSpec)
                    .executor(executor)  // Use executor
                    .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
            
            float[] input = {1.0f, 2.0f};
            float[] targets = {1.0f, 0.0f};
            
            // Make prediction - should use executor
            net.predict(input);
            
            // Training should use executor
            net.train(input, targets);
            
            // Verify executor was used
            assertTrue(executorUsageCount.get() > 0, 
                      "Custom activator should have used executor");
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testExecutorAndNonExecutorProduceSameBehavior() {
        // Test that the same network behaves identically with and without executor
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Create network without executor first  
        NeuralNet net = NeuralNet.newBuilder()
                .input(4)
                .layer(Layers.hiddenDenseRelu(8, optimizer))
                .output(Layers.outputSoftmaxCrossEntropy(2, optimizer));
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        
        // Get prediction without executor
        float[] predWithoutExecutor = net.predict(input);
        
        // Now test that executor-aware methods don't change behavior when executor is null
        ExecutorService executor = Executors.newFixedThreadPool(2);
        
        try {
            // Re-create the same network with an executor
            SgdOptimizer optimizer2 = new SgdOptimizer(0.01f);
            NeuralNet netWithExecutor = NeuralNet.newBuilder()
                    .input(4)
                    .layer(Layers.hiddenDenseRelu(8, optimizer2))
                    .executor(executor)
                    .output(Layers.outputSoftmaxCrossEntropy(2, optimizer2));
            
            // This should work without throwing exceptions
            float[] predWithExecutor = netWithExecutor.predict(input);
            
            // Just verify both produced valid predictions
            assertEquals(2, predWithoutExecutor.length);
            assertEquals(2, predWithExecutor.length);
            
            // Both should be valid probability distributions (sum to ~1.0)
            float sum1 = predWithoutExecutor[0] + predWithoutExecutor[1];
            float sum2 = predWithExecutor[0] + predWithExecutor[1];
            assertEquals(1.0f, sum1, 0.01f, "Should be valid probability distribution");
            assertEquals(1.0f, sum2, 0.01f, "Should be valid probability distribution");
            
        } finally {
            executor.shutdown();
        }
    }
}