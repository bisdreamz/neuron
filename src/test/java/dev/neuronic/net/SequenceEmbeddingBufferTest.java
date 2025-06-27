package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify that the InputSequenceEmbeddingLayer produces different outputs for different token sequences.
 * This directly tests the layer that seemed to be causing identical outputs.
 */
public class SequenceEmbeddingBufferTest {
    
    @Test
    public void testSequenceEmbeddingProducesDifferentOutputs() {
        // Create a standalone InputSequenceEmbeddingLayer
        InputSequenceEmbeddingLayer embeddingLayer = new InputSequenceEmbeddingLayer(
            new AdamWOptimizer(0.01f, 0.0001f),
            3, // sequence length
            10, // max vocab size  
            8, // embedding dimension
            WeightInitStrategy.HE
        );
        
        // First, populate vocabulary by getting token IDs
        int tokenA = embeddingLayer.getTokenId("a");
        int tokenB = embeddingLayer.getTokenId("b");
        int tokenC = embeddingLayer.getTokenId("c");
        
        System.out.println("Token IDs: a=" + tokenA + ", b=" + tokenB + ", c=" + tokenC);
        
        // Create different token sequences
        float[] sequence1 = {tokenA, tokenB, tokenC}; // a, b, c
        float[] sequence2 = {tokenB, tokenC, tokenA}; // b, c, a  
        float[] sequence3 = {tokenC, tokenA, tokenB}; // c, a, b
        
        // Test forward pass multiple times to check for buffer corruption
        Layer.LayerContext result1a = embeddingLayer.forward(sequence1);
        Layer.LayerContext result2a = embeddingLayer.forward(sequence2);
        Layer.LayerContext result3a = embeddingLayer.forward(sequence3);
        
        // Call forward again with same inputs - results should be identical
        Layer.LayerContext result1b = embeddingLayer.forward(sequence1);
        Layer.LayerContext result2b = embeddingLayer.forward(sequence2);
        Layer.LayerContext result3b = embeddingLayer.forward(sequence3);
        
        System.out.println("Sequence 'a b c' first call:  " + Arrays.toString(result1a.outputs()));
        System.out.println("Sequence 'a b c' second call: " + Arrays.toString(result1b.outputs()));
        System.out.println("Sequence 'b c a' first call:  " + Arrays.toString(result2a.outputs()));
        System.out.println("Sequence 'b c a' second call: " + Arrays.toString(result2b.outputs()));
        System.out.println("Sequence 'c a b' first call:  " + Arrays.toString(result3a.outputs()));
        System.out.println("Sequence 'c a b' second call: " + Arrays.toString(result3b.outputs()));
        
        // Verify that repeated calls produce identical results (no buffer corruption)
        assertArrayEquals(result1a.outputs(), result1b.outputs(), 1e-7f,
            "Repeated calls with same input should produce identical outputs");
        assertArrayEquals(result2a.outputs(), result2b.outputs(), 1e-7f,
            "Repeated calls with same input should produce identical outputs");
        assertArrayEquals(result3a.outputs(), result3b.outputs(), 1e-7f,
            "Repeated calls with same input should produce identical outputs");
        
        // Verify that different inputs produce different outputs
        assertFalse(Arrays.equals(result1a.outputs(), result2a.outputs()),
            "Different token sequences should produce different embeddings");
        assertFalse(Arrays.equals(result1a.outputs(), result3a.outputs()),
            "Different token sequences should produce different embeddings");
        assertFalse(Arrays.equals(result2a.outputs(), result3a.outputs()),
            "Different token sequences should produce different embeddings");
    }
    
    @Test
    public void testSequenceEmbeddingLayerContextIntegrity() {
        // Test that LayerContext objects maintain their integrity when stored
        InputSequenceEmbeddingLayer embeddingLayer = new InputSequenceEmbeddingLayer(
            new AdamWOptimizer(0.01f, 0.0001f),
            3, 10, 8, WeightInitStrategy.HE
        );
        
        int tokenA = embeddingLayer.getTokenId("a");
        int tokenB = embeddingLayer.getTokenId("b");
        int tokenC = embeddingLayer.getTokenId("c");
        
        float[] sequence1 = {tokenA, tokenB, tokenC};
        float[] sequence2 = {tokenB, tokenC, tokenA};
        
        // Get contexts and store them
        Layer.LayerContext context1 = embeddingLayer.forward(sequence1);
        float[] savedOutput1 = context1.outputs().clone(); // Save a copy
        
        Layer.LayerContext context2 = embeddingLayer.forward(sequence2);
        float[] savedOutput2 = context2.outputs().clone(); // Save a copy
        
        // Verify the first context wasn't corrupted by the second call
        assertArrayEquals(savedOutput1, context1.outputs(), 1e-7f,
            "First LayerContext should not be corrupted by subsequent forward calls");
        
        // Call forward again and check that stored contexts are still intact
        embeddingLayer.forward(sequence1);
        
        assertArrayEquals(savedOutput1, context1.outputs(), 1e-7f,
            "Stored LayerContext should remain intact after more forward calls");
        assertArrayEquals(savedOutput2, context2.outputs(), 1e-7f,
            "Stored LayerContext should remain intact after more forward calls");
    }
}