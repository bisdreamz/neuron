package dev.neuronic.net.activators;

import dev.neuronic.net.math.Parallelization;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test parallel execution of activators.
 */
class ParallelActivatorTest {

    @Test
    void testReluParallelVsSequential() {
        ReluActivator relu = ReluActivator.INSTANCE;
        
        // Create large input that should trigger parallelization
        int size = 5000; // Larger than default threshold
        float[] input = new float[size];
        float[] outputSequential = new float[size];
        float[] outputParallel = new float[size];
        
        // Fill with test data (mix of positive and negative)
        for (int i = 0; i < size; i++) {
            input[i] = (i % 2 == 0) ? i * 0.1f : -i * 0.1f;
        }
        
        // Sequential execution
        relu.activate(input, outputSequential);
        
        // Parallel execution
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            relu.activate(input, outputParallel, executor);
            
            // Results should be identical
            assertArrayEquals(outputSequential, outputParallel, 0.0001f);
            
            // Verify ReLU behavior on a few samples
            assertTrue(outputSequential[0] == 0.0f); // input[0] = 0
            assertTrue(outputSequential[2] >= 0.0f);  // positive input
            assertTrue(outputSequential[1] == 0.0f); // negative input
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testReluDerivativeParallelVsSequential() {
        ReluActivator relu = ReluActivator.INSTANCE;
        
        // Create large input that should trigger parallelization
        int size = 5000;
        float[] input = new float[size];
        float[] outputSequential = new float[size];
        float[] outputParallel = new float[size];
        
        // Fill with test data
        for (int i = 0; i < size; i++) {
            input[i] = (i % 2 == 0) ? i * 0.1f : -i * 0.1f;
        }
        
        // Sequential execution
        relu.derivative(input, outputSequential);
        
        // Parallel execution
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            relu.derivative(input, outputParallel, executor);
            
            // Results should be identical
            assertArrayEquals(outputSequential, outputParallel, 0.0001f);
            
            // Verify derivative behavior
            assertEquals(1.0f, outputSequential[2]); // positive input -> derivative = 1
            assertEquals(0.0f, outputSequential[1]); // negative input -> derivative = 0
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testSmallInputDoesNotParallelize() {
        ReluActivator relu = ReluActivator.INSTANCE;
        
        // Small input that should NOT trigger parallelization
        int size = 100; // Much smaller than threshold
        float[] input = new float[size];
        float[] output = new float[size];
        
        for (int i = 0; i < size; i++) {
            input[i] = i * 0.1f;
        }
        
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            // Should not parallelize (but should still work)
            assertFalse(Parallelization.shouldParallelize(size, executor));
            
            // Should work correctly regardless
            relu.activate(input, output, executor);
            
            // Verify results
            for (int i = 0; i < size; i++) {
                assertEquals(Math.max(0, input[i]), output[i], 0.0001f);
            }
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testParallelizationThresholds() {
        // Test that our threshold makes sense
        int threshold = Parallelization.getMinWorkSizePerThread();
        ExecutorService executor = Executors.newFixedThreadPool(4);
        
        try {
            // Just below threshold * 2 - should not parallelize
            assertFalse(Parallelization.shouldParallelize(threshold * 2 - 1, executor));
            
            // At threshold * 2 - should parallelize
            assertTrue(Parallelization.shouldParallelize(threshold * 2, executor));
            
            // Well above threshold - should parallelize
            assertTrue(Parallelization.shouldParallelize(threshold * 10, executor));
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testSigmoidParallelVsSequential() {
        SigmoidActivator sigmoid = SigmoidActivator.INSTANCE;
        
        int size = 5000;
        float[] input = new float[size];
        float[] outputSequential = new float[size];
        float[] outputParallel = new float[size];
        
        for (int i = 0; i < size; i++) {
            input[i] = (i - size/2) * 0.01f; // Range from negative to positive
        }
        
        // Sequential execution
        sigmoid.activate(input, outputSequential);
        
        // Parallel execution
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            sigmoid.activate(input, outputParallel, executor);
            assertArrayEquals(outputSequential, outputParallel, 0.0001f);
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testTanhParallelVsSequential() {
        TanhActivator tanh = TanhActivator.INSTANCE;
        
        int size = 5000;
        float[] input = new float[size];
        float[] outputSequential = new float[size];
        float[] outputParallel = new float[size];
        
        for (int i = 0; i < size; i++) {
            input[i] = (i - size/2) * 0.01f;
        }
        
        // Sequential execution
        tanh.activate(input, outputSequential);
        
        // Parallel execution
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            tanh.activate(input, outputParallel, executor);
            assertArrayEquals(outputSequential, outputParallel, 0.0001f);
        } finally {
            executor.shutdown();
        }
    }
}