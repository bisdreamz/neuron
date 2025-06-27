package dev.neuronic.net.layers;

import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive edge case testing for GRU layer implementation.
 */
class GruEdgeCaseTest {
    
    @Test
    void testMinimalDimensions() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 1, 1, WeightInitStrategy.XAVIER);
        
        // Test with minimal viable dimensions
        float[] input = {0.5f}; // Single timestep, single feature
        Layer.LayerContext context = gru.forward(input);
        
        assertEquals(1, context.outputs().length, "Output should be hiddenSize=1");
        assertTrue(Float.isFinite(context.outputs()[0]), "Output should be finite");
        
        // Test backward pass
        float[] upstreamGrad = {0.1f};
        float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad);
        
        assertEquals(1, inputGradients.length, "Input gradients should match input size");
        assertTrue(Float.isFinite(inputGradients[0]), "Input gradients should be finite");
    }
    
    @Test
    void testSingleTimestep() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 8, 4, WeightInitStrategy.XAVIER);
        
        // Test with single timestep
        float[] input = {1.0f, 0.5f, -0.5f, 0.0f}; // Single timestep
        Layer.LayerContext context = gru.forward(input);
        
        assertEquals(8, context.outputs().length, "Output should be hiddenSize * 1 timestep");
        
        // All outputs should be bounded by tanh
        for (float val : context.outputs()) {
            assertTrue(val >= -1.0f && val <= 1.0f, "GRU output should be bounded by tanh: " + val);
        }
        
        // Test backward pass
        float[] upstreamGrad = new float[8];
        for (int i = 0; i < 8; i++) {
            upstreamGrad[i] = (float) Math.random() - 0.5f;
        }
        
        float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad);
        assertEquals(4, inputGradients.length, "Input gradients should match input size");
    }
    
    @Test
    void testLongSequence() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 2, WeightInitStrategy.XAVIER);
        
        // Test with long sequence (100 timesteps)
        int seqLen = 100;
        float[] input = new float[seqLen * 2];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.sin(i * 0.1); // Sinusoidal pattern
        }
        
        Layer.LayerContext context = gru.forward(input);
        assertEquals(seqLen * 4, context.outputs().length, 
            "Output should be seqLen * hiddenSize");
        
        // Check for gradient explosion/vanishing in long sequences
        float[] upstreamGrad = new float[seqLen * 4];
        for (int i = 0; i < upstreamGrad.length; i++) {
            upstreamGrad[i] = 0.1f; // Small consistent gradient
        }
        
        float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad);
        
        // Check that gradients don't explode or vanish
        boolean hasReasonableGradients = false;
        for (float grad : inputGradients) {
            assertTrue(Float.isFinite(grad), "Gradients should be finite: " + grad);
            assertTrue(Math.abs(grad) < 100.0f, "Gradients should not explode: " + grad);
            if (Math.abs(grad) > 1e-6f) {
                hasReasonableGradients = true;
            }
        }
        assertTrue(hasReasonableGradients, "Should have some non-vanishing gradients");
    }
    
    @Test
    void testExtremeInputValues() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        // Test with extreme input values
        float[] extremeInput = {
            1000.0f, -1000.0f, 0.0f,    // Large positive/negative
            Float.MIN_VALUE, Float.MAX_VALUE / 1e6f, 1e-10f  // Very small/large
        };
        
        // Should handle extreme values gracefully (activations will saturate)
        Layer.LayerContext context = gru.forward(extremeInput);
        
        // All outputs should still be finite and bounded
        for (float val : context.outputs()) {
            assertTrue(Float.isFinite(val), "Output should be finite even with extreme inputs: " + val);
            assertTrue(val >= -1.0f && val <= 1.0f, "Output should be bounded: " + val);
        }
    }
    
    @Test
    void testNaNInfinity() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        // Test with NaN input
        float[] nanInput = {1.0f, Float.NaN, 0.5f, 0.0f, 1.0f, 0.5f};
        assertThrows(IllegalArgumentException.class, () -> gru.forward(nanInput),
            "Should reject NaN input");
        
        // Test with Infinity input
        float[] infInput = {1.0f, Float.POSITIVE_INFINITY, 0.5f, 0.0f, 1.0f, 0.5f};
        assertThrows(IllegalArgumentException.class, () -> gru.forward(infInput),
            "Should reject infinite input");
        
        // Test with negative infinity
        float[] negInfInput = {1.0f, Float.NEGATIVE_INFINITY, 0.5f, 0.0f, 1.0f, 0.5f};
        assertThrows(IllegalArgumentException.class, () -> gru.forward(negInfInput),
            "Should reject negative infinite input");
    }
    
    @Test
    void testBackwardPassValidation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        float[] input = {1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f};
        Layer.LayerContext context = gru.forward(input);
        
        // Test with null gradient
        assertThrows(IllegalArgumentException.class, () -> 
            gru.backward(new Layer.LayerContext[]{context}, 0, null),
            "Should reject null gradient");
        
        // Test with wrong gradient size
        float[] wrongSizeGrad = {0.1f, 0.2f}; // Too small
        assertThrows(IllegalArgumentException.class, () -> 
            gru.backward(new Layer.LayerContext[]{context}, 0, wrongSizeGrad),
            "Should reject wrong gradient size");
        
        // Test with NaN gradient
        float[] nanGrad = {0.1f, Float.NaN, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        assertThrows(IllegalArgumentException.class, () -> 
            gru.backward(new Layer.LayerContext[]{context}, 0, nanGrad),
            "Should reject NaN gradient");
        
        // Test with null stack
        float[] validGrad = new float[8];
        assertThrows(IllegalArgumentException.class, () -> 
            gru.backward(null, 0, validGrad),
            "Should reject null stack");
        
        // Test with invalid stack index
        assertThrows(IndexOutOfBoundsException.class, () -> 
            gru.backward(new Layer.LayerContext[]{context}, 1, validGrad),
            "Should reject invalid stack index");
    }
    
    @Test
    void testConcurrentAccess() throws InterruptedException {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 8, 4, WeightInitStrategy.XAVIER);
        
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try {
            // Run multiple forward passes concurrently
            var futures = new java.util.concurrent.Future[10];
            for (int i = 0; i < 10; i++) {
                final int threadId = i;
                futures[i] = executor.submit(() -> {
                    float[] input = {threadId * 0.1f, 0.5f, 0.0f, 1.0f, 
                                   0.2f, threadId * 0.05f, 0.8f, 0.3f};
                    Layer.LayerContext context = gru.forward(input);
                    
                    // Verify output is reasonable
                    assertEquals(16, context.outputs().length);
                    for (float val : context.outputs()) {
                        assertTrue(Float.isFinite(val), "Concurrent output should be finite");
                    }
                    return context;
                });
            }
            
            // Wait for all to complete
            for (var future : futures) {
                try {
                    assertNotNull(future.get(), "Concurrent execution should succeed");
                } catch (Exception e) {
                    fail("Concurrent execution failed: " + e.getMessage());
                }
            }
            
        } finally {
            executor.shutdown();
        }
    }
    
    @Test
    void testMemoryStability() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        GruLayer gru = new GruLayer(optimizer, 64, 32, WeightInitStrategy.XAVIER);
        
        // Run many iterations to test for memory leaks or buffer issues
        for (int i = 0; i < 1000; i++) {
            float[] input = new float[32 * 5]; // 5 timesteps
            for (int j = 0; j < input.length; j++) {
                input[j] = (float) Math.sin(i * 0.01 + j * 0.1);
            }
            
            Layer.LayerContext context = gru.forward(input);
            
            float[] upstreamGrad = new float[64 * 5];
            for (int j = 0; j < upstreamGrad.length; j++) {
                upstreamGrad[j] = 0.01f;
            }
            
            float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, upstreamGrad);
            
            // Verify stability
            for (float val : context.outputs()) {
                assertTrue(Float.isFinite(val), "Output should remain stable after " + i + " iterations");
            }
            for (float grad : inputGradients) {
                assertTrue(Float.isFinite(grad), "Gradients should remain stable after " + i + " iterations");
            }
            
            // Periodic memory pressure test
            if (i % 100 == 0) {
                System.gc(); // Suggest garbage collection
            }
        }
    }
    
    @Test 
    void testNumericalStability() {
        SgdOptimizer optimizer = new SgdOptimizer(0.001f); // Small learning rate
        GruLayer gru = new GruLayer(optimizer, 16, 8, WeightInitStrategy.XAVIER);
        
        // Test with very small inputs (near underflow)
        float[] tinyInput = new float[8 * 3];
        for (int i = 0; i < tinyInput.length; i++) {
            tinyInput[i] = 1e-30f; // Near float underflow
        }
        
        Layer.LayerContext context = gru.forward(tinyInput);
        for (float val : context.outputs()) {
            assertTrue(Float.isFinite(val), "Should handle tiny inputs gracefully");
        }
        
        // Test learning with small gradients
        float[] tinyGrad = new float[16 * 3];
        for (int i = 0; i < tinyGrad.length; i++) {
            tinyGrad[i] = 1e-10f; // Very small gradients
        }
        
        float[] inputGradients = gru.backward(new Layer.LayerContext[]{context}, 0, tinyGrad);
        for (float grad : inputGradients) {
            assertTrue(Float.isFinite(grad), "Should handle tiny gradients gracefully");
        }
    }
}