package dev.neuronic.net.layers;

import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.SerializationConstants;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for InputEmbeddingLayer functionality.
 */
class InputEmbeddingLayerTest {
    
    @Test
    void testBasicEmbeddingLookup() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 5, 3, WeightInitStrategy.XAVIER);
        
        // Set known embeddings for testing
        layer.setEmbedding(0, new float[]{1.0f, 0.0f, 0.0f});
        layer.setEmbedding(1, new float[]{0.0f, 1.0f, 0.0f});
        layer.setEmbedding(2, new float[]{0.0f, 0.0f, 1.0f});
        
        // Test single token lookup
        float[] tokens = {1.0f};
        Layer.LayerContext context = layer.forward(tokens, false);
        float[] output = context.outputs();
        
        assertArrayEquals(new float[]{0.0f, 1.0f, 0.0f}, output, 1e-6f,
            "Single token embedding should match");
        
        // Test multiple token lookup
        float[] multiTokens = {0.0f, 2.0f, 1.0f};
        Layer.LayerContext multiContext = layer.forward(multiTokens, false);
        float[] multiOutput = multiContext.outputs();
        
        float[] expected = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        assertArrayEquals(expected, multiOutput, 1e-6f,
            "Multiple token embeddings should be concatenated");
    }
    
    @Test
    void testEmbeddingDimensions() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 100, 50, WeightInitStrategy.XAVIER);
        
        assertEquals(100, layer.getVocabSize(), "Vocab size should match");
        assertEquals(50, layer.getEmbeddingDim(), "Embedding dim should match");
        assertEquals(50, layer.getOutputSize(), "Output size should be embedding dim");
        
        // Test sequence length scaling
        float[] tokens = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}; // 5 tokens
        Layer.LayerContext context = layer.forward(tokens, false);
        assertEquals(5 * 50, context.outputs().length, 
            "Output should be seqLen * embeddingDim");
    }
    
    @Test
    void testGradientAccumulation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.1f); // High learning rate for visible changes
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 3, 2, WeightInitStrategy.XAVIER);
        
        // Set initial embeddings
        layer.setEmbedding(0, new float[]{1.0f, 0.0f});
        layer.setEmbedding(1, new float[]{0.0f, 1.0f});
        
        // Get initial embedding for token 0
        float[] initialEmbedding = layer.getEmbedding(0);
        
        // Forward pass with repeated token
        float[] tokens = {0.0f, 1.0f, 0.0f}; // Token 0 appears twice
        Layer.LayerContext context = layer.forward(tokens, false);
        
        // Backward pass with gradients
        float[] gradients = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}; // 3 tokens * 2 dims
        Layer.LayerContext[] stack = {context};
        layer.backward(stack, 0, gradients);
        
        // Check that token 0 embedding changed (accumulated gradients from positions 0 and 2)
        float[] updatedEmbedding = layer.getEmbedding(0);
        assertNotEquals(initialEmbedding[0], updatedEmbedding[0], 
            "Embedding should have changed after gradient update");
        
        // Token 0 should have accumulated gradients: (0.1 + 0.5) and (0.2 + 0.6)
        // Expected change: -learningRate * accumulatedGradient
        float expectedChange0 = -0.1f * (0.1f + 0.5f);
        float expectedChange1 = -0.1f * (0.2f + 0.6f);
        
        assertEquals(initialEmbedding[0] + expectedChange0, updatedEmbedding[0], 1e-6f,
            "Token 0 dim 0 should accumulate gradients correctly");
        assertEquals(initialEmbedding[1] + expectedChange1, updatedEmbedding[1], 1e-6f,
            "Token 0 dim 1 should accumulate gradients correctly");
    }
    
    @Test
    void testInvalidInputs() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 5, 3, WeightInitStrategy.XAVIER);
        
        // Test invalid token IDs
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{-1.0f}, false); // Negative token ID
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{5.0f}, false); // Token ID >= vocabSize
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{}, false); // Empty sequence
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.forward(new float[]{1.5f}, false); // Non-integer token ID
        });
        
        // Test invalid embedding operations
        assertThrows(IllegalArgumentException.class, () -> {
            layer.getEmbedding(-1); // Invalid token ID
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            layer.setEmbedding(0, new float[]{1.0f, 2.0f, 3.0f, 4.0f}); // Wrong dimension
        });
    }
    
    @Test
    void testLargeSequence() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 1000, 128, WeightInitStrategy.XAVIER);
        
        // Test with a large sequence (should automatically resize buffers)
        float[] tokens = new float[600]; // Larger than initial buffer size
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = i % 1000; // Valid token IDs
        }
        
        Layer.LayerContext context = layer.forward(tokens, false);
        assertEquals(600 * 128, context.outputs().length,
            "Should handle large sequences correctly");
    }
    
    @Test
    void testSerialization() throws IOException {
        SgdOptimizer optimizer = new SgdOptimizer(0.02f);
        InputEmbeddingLayer original = new InputEmbeddingLayer(optimizer, 4, 3, WeightInitStrategy.XAVIER);
        
        // Set some known embeddings
        original.setEmbedding(0, new float[]{1.0f, 2.0f, 3.0f});
        original.setEmbedding(1, new float[]{4.0f, 5.0f, 6.0f});
        original.setEmbedding(2, new float[]{7.0f, 8.0f, 9.0f});
        original.setEmbedding(3, new float[]{10.0f, 11.0f, 12.0f});
        
        // Test forward pass
        float[] tokens = {1.0f, 3.0f, 0.0f};
        float[] originalOutput = original.forward(tokens, false).outputs();
        
        // Serialize
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        original.writeTo(out, SerializationConstants.CURRENT_VERSION);
        out.close();
        
        // Deserialize
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        DataInputStream in = new DataInputStream(bais);
        InputEmbeddingLayer deserialized = InputEmbeddingLayer.deserialize(in, SerializationConstants.CURRENT_VERSION);
        in.close();
        
        // Test equivalence
        assertEquals(original.getVocabSize(), deserialized.getVocabSize());
        assertEquals(original.getEmbeddingDim(), deserialized.getEmbeddingDim());
        assertEquals(original.getTypeId(), deserialized.getTypeId());
        
        // Test embeddings
        for (int i = 0; i < 4; i++) {
            assertArrayEquals(original.getEmbedding(i), deserialized.getEmbedding(i), 1e-6f,
                "Embedding " + i + " should match after serialization");
        }
        
        // Test forward pass equivalence
        float[] deserializedOutput = deserialized.forward(tokens, false).outputs();
        assertArrayEquals(originalOutput, deserializedOutput, 1e-6f,
            "Forward pass should be identical after serialization");
    }
    
    @Test
    void testSerializationSizeAccuracy() throws IOException {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        InputEmbeddingLayer layer = new InputEmbeddingLayer(optimizer, 10, 5, WeightInitStrategy.XAVIER);
        
        // Get estimated size
        int estimatedSize = layer.getSerializedSize(SerializationConstants.CURRENT_VERSION);
        
        // Actual serialization
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        layer.writeTo(out, SerializationConstants.CURRENT_VERSION);
        out.close();
        
        int actualSize = baos.toByteArray().length;
        
        assertEquals(estimatedSize, actualSize, 
            "Estimated serialization size should match actual size");
    }
    
    @Test
    void testLayerSpec() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = InputEmbeddingLayer.spec(100, 64, optimizer, WeightInitStrategy.HE);
        
        assertEquals(64, spec.getOutputSize(), "Spec should return embedding dimension");
        
        // Create layer from spec
        Layer layer = spec.create(999); // Input size should be ignored
        assertTrue(layer instanceof InputEmbeddingLayer, "Should create InputEmbeddingLayer");
        
        InputEmbeddingLayer embeddingLayer = (InputEmbeddingLayer) layer;
        assertEquals(100, embeddingLayer.getVocabSize(), "Vocab size should match spec");
        assertEquals(64, embeddingLayer.getEmbeddingDim(), "Embedding dim should match spec");
    }
    
    @Test
    void testConstructorValidation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        
        // Invalid vocab size
        assertThrows(IllegalArgumentException.class, () -> {
            new InputEmbeddingLayer(optimizer, 0, 10, WeightInitStrategy.XAVIER);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new InputEmbeddingLayer(optimizer, -5, 10, WeightInitStrategy.XAVIER);
        });
        
        // Invalid embedding dimension
        assertThrows(IllegalArgumentException.class, () -> {
            new InputEmbeddingLayer(optimizer, 10, 0, WeightInitStrategy.XAVIER);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new InputEmbeddingLayer(optimizer, 10, -3, WeightInitStrategy.XAVIER);
        });
    }
}