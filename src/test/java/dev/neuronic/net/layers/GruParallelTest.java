package dev.neuronic.net.layers;

import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for GRU layer parallel execution.
 */
class GruParallelTest {
    
    @Test
    void testGruParallelForward() throws InterruptedException {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        // Use large hidden size to trigger parallel execution
        GruLayer gru = new GruLayer(optimizer, 128, 64, WeightInitStrategy.XAVIER);
        
        // Create test input
        float[] input = new float[64 * 3]; // 3 timesteps
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
        
        // Test sequential execution
        Layer.LayerContext sequentialResult = gru.forward(input);
        
        // Test parallel execution with ExecutorService
        ExecutorService executor = Executors.newFixedThreadPool(3);
        try {
            Layer.LayerContext parallelResult = gru.forward(input, executor);
            
            // Results should be identical
            assertArrayEquals(sequentialResult.outputs(), parallelResult.outputs(), 1e-6f,
                "Parallel and sequential execution should produce identical results");
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testGruParallelBackward() throws InterruptedException {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        // Use large hidden size to trigger parallel execution  
        GruLayer gru = new GruLayer(optimizer, 128, 64, WeightInitStrategy.XAVIER);
        
        // Create test input
        float[] input = new float[64 * 2]; // 2 timesteps
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
        
        // Forward pass
        Layer.LayerContext context = gru.forward(input);
        
        // Create upstream gradient
        float[] upstreamGrad = new float[128 * 2]; // 2 timesteps * hiddenSize
        for (int i = 0; i < upstreamGrad.length; i++) {
            upstreamGrad[i] = (float) Math.random();
        }
        
        // Test that parallel backward execution works without errors
        ExecutorService executor = Executors.newFixedThreadPool(3);
        try {
            float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad, executor);
            
            // Verify that gradients are reasonable (not NaN/Infinite and have expected size)
            assertNotNull(inputGradients, "Input gradients should not be null");
            assertEquals(input.length, inputGradients.length, "Input gradients should match input size");
            
            // All gradients should be finite numbers
            for (float grad : inputGradients) {
                assertTrue(Float.isFinite(grad), "All gradients should be finite: " + grad);
            }
            
            // Should have some non-zero gradients for meaningful learning
            boolean hasNonZeroGrad = false;
            for (float grad : inputGradients) {
                if (Math.abs(grad) > 1e-6f) {
                    hasNonZeroGrad = true;
                    break;
                }
            }
            assertTrue(hasNonZeroGrad, "Should have some non-zero gradients for learning");
                
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testGruSmallSizeFallback() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        // Use small hidden size - should fall back to sequential
        GruLayer gru = new GruLayer(optimizer, 32, 16, WeightInitStrategy.XAVIER);
        
        float[] input = new float[16 * 2]; // 2 timesteps
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
        
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Layer.LayerContext result1 = gru.forward(input);
            Layer.LayerContext result2 = gru.forward(input, executor);
            
            // Should produce identical results since it falls back to sequential
            assertArrayEquals(result1.outputs(), result2.outputs(), 1e-10f,
                "Small hidden size should fall back to sequential execution");
                
        } finally {
            executor.shutdown();
        }
    }
}