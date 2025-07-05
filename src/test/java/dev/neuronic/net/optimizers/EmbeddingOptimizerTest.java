package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test the forEmbeddings() method behavior for different optimizers.
 */
public class EmbeddingOptimizerTest {
    
    @Test
    void testAdamWForEmbeddings() {
        // AdamW with weight decay should create a new optimizer with reduced decay and higher LR
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.01f);
        Optimizer embeddingOpt = adamW.forEmbeddings();
        
        assertNotSame(adamW, embeddingOpt, "Should create new optimizer for embeddings");
        assertTrue(embeddingOpt instanceof AdamWOptimizer, "Should return AdamW optimizer");
        
        AdamWOptimizer embeddingAdamW = (AdamWOptimizer) embeddingOpt;
        assertEquals(0.005f, embeddingAdamW.getLearningRate(), 1e-7f, "Learning rate should be 5x higher");
        assertEquals(0.001f, embeddingAdamW.getWeightDecay(), 1e-7f, "Weight decay should be 10x less");
    }
    
    @Test
    void testAdamWWithZeroDecay() {
        // AdamW with zero weight decay should still create new optimizer due to LR change
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.0f);
        Optimizer embeddingOpt = adamW.forEmbeddings();
        
        assertNotSame(adamW, embeddingOpt, "Should create new optimizer due to LR change");
        assertTrue(embeddingOpt instanceof AdamWOptimizer, "Should return AdamW optimizer");
        
        AdamWOptimizer embeddingAdamW = (AdamWOptimizer) embeddingOpt;
        assertEquals(0.005f, embeddingAdamW.getLearningRate(), 1e-7f, "Learning rate should be 5x higher");
        assertEquals(0.0f, embeddingAdamW.getWeightDecay(), 1e-7f, "Weight decay should remain zero");
    }
    
    @Test
    void testSgdForEmbeddings() {
        // SGD should return same optimizer (no weight decay)
        SgdOptimizer sgd = new SgdOptimizer(0.01f);
        Optimizer embeddingOpt = sgd.forEmbeddings();
        
        assertSame(sgd, embeddingOpt, "SGD should return same optimizer");
    }
    
    @Test
    void testAdamForEmbeddings() {
        // Adam should return same optimizer (no weight decay)
        AdamOptimizer adam = new AdamOptimizer(0.001f);
        Optimizer embeddingOpt = adam.forEmbeddings();
        
        assertSame(adam, embeddingOpt, "Adam should return same optimizer");
    }
}