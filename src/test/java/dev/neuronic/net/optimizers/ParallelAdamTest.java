package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests to verify that Adam and AdamW optimizers properly utilize parallel execution.
 */
class ParallelAdamTest {

    private float[][] largeWeights;
    private float[] biases;
    private float[][] weightGradients;
    private float[] biasGradients;
    private ExecutorService executor;

    @BeforeEach
    void setUp() {
        // Create large weight matrix to trigger parallelization (>= 2048 rows for 2 threads)
        int rows = 2500;  // Above parallelization threshold
        int cols = 100;
        
        largeWeights = new float[rows][cols];
        weightGradients = new float[rows][cols];
        biases = new float[cols];
        biasGradients = new float[cols];
        
        // Initialize with small random values
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                largeWeights[i][j] = (float) (Math.random() * 0.1 - 0.05);
                weightGradients[i][j] = (float) (Math.random() * 0.01 - 0.005);
            }
        }
        
        for (int j = 0; j < cols; j++) {
            biases[j] = (float) (Math.random() * 0.1 - 0.05);
            biasGradients[j] = (float) (Math.random() * 0.01 - 0.005);
        }
        
        executor = Executors.newFixedThreadPool(4);
    }
    
    @Test
    void testAdamParallelOptimization() {
        AdamOptimizer optimizer = new AdamOptimizer(0.001f);
        
        // Store original values
        float[][] originalWeights = copyWeights(largeWeights);
        float[] originalBiases = biases.clone();
        
        // Apply parallel optimization
        optimizer.optimize(largeWeights, biases, weightGradients, biasGradients, executor);
        
        // Verify that weights have changed (optimization occurred)
        boolean weightsChanged = false;
        for (int i = 0; i < largeWeights.length && !weightsChanged; i++) {
            for (int j = 0; j < largeWeights[i].length; j++) {
                if (Math.abs(largeWeights[i][j] - originalWeights[i][j]) > 1e-8f) {
                    weightsChanged = true;
                    break;
                }
            }
        }
        assertTrue(weightsChanged, "Weights should have been updated by optimization");
        
        // Verify biases changed
        boolean biasesChanged = false;
        for (int j = 0; j < biases.length; j++) {
            if (Math.abs(biases[j] - originalBiases[j]) > 1e-8f) {
                biasesChanged = true;
                break;
            }
        }
        assertTrue(biasesChanged, "Biases should have been updated by optimization");
    }
    
    @Test
    void testAdamWParallelOptimization() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Store original values
        float[][] originalWeights = copyWeights(largeWeights);
        
        // Apply parallel optimization
        optimizer.optimize(largeWeights, biases, weightGradients, biasGradients, executor);
        
        // Verify that weights have changed
        boolean weightsChanged = false;
        for (int i = 0; i < largeWeights.length && !weightsChanged; i++) {
            for (int j = 0; j < largeWeights[i].length; j++) {
                if (Math.abs(largeWeights[i][j] - originalWeights[i][j]) > 1e-8f) {
                    weightsChanged = true;
                    break;
                }
            }
        }
        assertTrue(weightsChanged, "AdamW weights should have been updated by optimization");
    }
    
    @Test
    void testParallelVsSequentialConsistency() {
        // Test that parallel and sequential optimization produce similar results
        AdamOptimizer optimizerParallel = new AdamOptimizer(0.001f);
        AdamOptimizer optimizerSequential = new AdamOptimizer(0.001f);
        
        // Create identical starting conditions
        float[][] weightsParallel = copyWeights(largeWeights);
        float[][] weightsSequential = copyWeights(largeWeights);
        float[] biasesParallel = biases.clone();
        float[] biasesSequential = biases.clone();
        
        // Apply optimizations
        optimizerParallel.optimize(weightsParallel, biasesParallel, weightGradients, biasGradients, executor);
        optimizerSequential.optimize(weightsSequential, biasesSequential, weightGradients, biasGradients); // No executor
        
        // Results should be identical (deterministic)
        for (int i = 0; i < weightsParallel.length; i++) {
            assertArrayEquals(weightsParallel[i], weightsSequential[i], 1e-6f, 
                "Parallel and sequential results should be identical for row " + i);
        }
        assertArrayEquals(biasesParallel, biasesSequential, 1e-6f, 
            "Parallel and sequential bias results should be identical");
    }
    
    @Test
    void testSmallMatrixFallsBackToSequential() {
        // Test that small matrices don't trigger parallelization
        float[][] smallWeights = new float[10][10]; // Well below threshold
        float[] smallBiases = new float[10];
        float[][] smallWeightGradients = new float[10][10];
        float[] smallBiasGradients = new float[10];
        
        // Initialize small arrays
        for (int i = 0; i < 10; i++) {
            smallBiases[i] = 0.1f;
            smallBiasGradients[i] = 0.01f;
            for (int j = 0; j < 10; j++) {
                smallWeights[i][j] = 0.1f;
                smallWeightGradients[i][j] = 0.01f;
            }
        }
        
        AdamOptimizer optimizer = new AdamOptimizer(0.01f);
        
        // This should not throw any exceptions and should work correctly
        assertDoesNotThrow(() -> {
            optimizer.optimize(smallWeights, smallBiases, smallWeightGradients, smallBiasGradients, executor);
        });
        
        // Verify optimization occurred
        assertEquals(0.09f, smallWeights[0][0], 0.01f, "Small matrix optimization should still work");
    }
    
    private float[][] copyWeights(float[][] original) {
        float[][] copy = new float[original.length][];
        for (int i = 0; i < original.length; i++) {
            copy[i] = original[i].clone();
        }
        return copy;
    }
}