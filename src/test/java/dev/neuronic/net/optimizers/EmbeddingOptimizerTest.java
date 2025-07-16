package dev.neuronic.net.optimizers;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test the forEmbeddings() method behavior for different optimizers.
 */
public class EmbeddingOptimizerTest {
    
    @Test
    void testAdamWForEmbeddings() {
        // AdamW with weight decay should create a new optimizer with NO decay for embeddings
        // This matches modern NLP practice (GPT, BERT, T5, LLaMA)
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.01f);
        Optimizer embeddingOpt = adamW.forEmbeddings();
        
        assertNotSame(adamW, embeddingOpt, "Should create new optimizer for embeddings");
        assertTrue(embeddingOpt instanceof AdamWOptimizer, "Should return AdamW optimizer");
        
        AdamWOptimizer embeddingAdamW = (AdamWOptimizer) embeddingOpt;
        assertEquals(0.001f, embeddingAdamW.getLearningRate(), 1e-7f, "Learning rate should remain the same");
        assertEquals(0.0f, embeddingAdamW.getWeightDecay(), 1e-7f, "Weight decay should be zero for embeddings");
    }
    
    @Test
    void testAdamWWithZeroDecay() {
        // AdamW with zero weight decay should return same optimizer (no changes needed)
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.0f);
        Optimizer embeddingOpt = adamW.forEmbeddings();
        
        assertSame(adamW, embeddingOpt, "Should return same optimizer when weight decay is already zero");
        assertEquals(0.001f, adamW.getLearningRate(), 1e-7f, "Learning rate should remain unchanged");
        assertEquals(0.0f, adamW.getWeightDecay(), 1e-7f, "Weight decay should remain zero");
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