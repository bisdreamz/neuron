package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;
import java.util.Map;
import java.util.HashMap;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for dictionary size limits and LRU behavior in SimpleNet.
 */
class SimpleNetDictionaryLimitsTest {
    
    @Test
    void testDictionarySizeLimitExceeded_FloatArray() {
        // Create model with small dictionary limit
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(2)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(5, 8, "item_id"),  // Max 5 unique values
                Feature.passthrough("price")
            ))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train with values within limit
        for (int i = 0; i < 5; i++) {
            model.train(new float[]{(float)i, 10.0f}, i * 0.1f);
        }
        
        // Adding 6th unique value should throw exception
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> {
            model.train(new float[]{5.0f, 10.0f}, 0.5f);
        });
        
        assertTrue(ex.getMessage().contains("Dictionary for feature"));
        assertTrue(ex.getMessage().contains("exceeding maxUniqueValues=5"));
        assertTrue(ex.getMessage().contains("Feature.hashedEmbedding()"));
        assertTrue(ex.getMessage().contains("Feature.embeddingLRU()"));
    }
    
    @Test
    void testDictionarySizeLimitExceeded_Map() {
        // Create model with small dictionary limit
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(2)
            .layer(Layers.inputMixed(optimizer,
                Feature.oneHot(3, "category"),  // Max 3 categories
                Feature.passthrough("score")
            ))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train with categories within limit
        model.train(Map.of("category", "A", "score", 1.0f), 0.1f);
        model.train(Map.of("category", "B", "score", 2.0f), 0.2f);
        model.train(Map.of("category", "C", "score", 3.0f), 0.3f);
        
        // Adding 4th category should throw exception
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> {
            model.train(Map.of("category", "D", "score", 4.0f), 0.4f);
        });
        
        assertTrue(ex.getMessage().contains("exceeding maxUniqueValues=3"));
    }
    
    @Test
    void testLRUDictionaryDoesNotThrow() {
        // Create model with LRU dictionary
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(2)
            .layer(Layers.inputMixed(optimizer,
                Feature.embeddingLRU(3, 8, "user_id"),  // LRU with max 3
                Feature.passthrough("value")
            ))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Should NOT throw even when exceeding limit - LRU evicts old entries
        assertDoesNotThrow(() -> {
            for (int i = 0; i < 10; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("user_id", "user" + i);
                input.put("value", (float)i);
                model.train(input, i * 0.1f);
            }
        });
        
        // Verify we can still predict with a recent entry
        Map<String, Object> testInput = new HashMap<>();
        testInput.put("user_id", "user9");  // Most recent entry
        testInput.put("value", 9.0f);
        float prediction = model.predictFloat(testInput);
        assertFalse(Float.isNaN(prediction));
    }
    
    @Test
    void testMixedLRUAndRegularDictionaries() {
        // Create model with both LRU and regular dictionaries
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(3)
            .layer(Layers.inputMixed(optimizer,
                Feature.embeddingLRU(100, 16, "user_id"),    // LRU - won't throw
                Feature.embedding(5, 8, "category"),         // Regular - will throw
                Feature.passthrough("score")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Train with some data
        for (int i = 0; i < 5; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("user_id", "user" + i);
            input.put("category", "cat" + i);
            input.put("score", (float)i);
            model.train(input, i * 0.1f);
        }
        
        // User IDs can keep growing (LRU)
        assertDoesNotThrow(() -> {
            for (int i = 5; i < 150; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("user_id", "user" + i);
                input.put("category", "cat0");  // Reuse existing category
                input.put("score", (float)i);
                model.train(input, i * 0.1f);
            }
        });
        
        // But category dictionary is limited
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> {
            Map<String, Object> input = new HashMap<>();
            input.put("user_id", "user999");
            input.put("category", "cat5");  // 6th category - exceeds limit
            input.put("score", 999.0f);
            model.train(input, 0.999f);
        });
        
        assertTrue(ex.getMessage().contains("category"));
        assertTrue(ex.getMessage().contains("exceeding maxUniqueValues=5"));
    }
    
    @Test
    void testOneHotLRU() {
        // Test LRU with one-hot encoding
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(2)
            .layer(Layers.inputMixed(optimizer,
                Feature.oneHotLRU(5, "status"),  // LRU one-hot with max 5
                Feature.passthrough("amount")
            ))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Should handle more than 5 statuses via LRU eviction
        assertDoesNotThrow(() -> {
            for (int i = 0; i < 20; i++) {
                Map<String, Object> input = new HashMap<>();
                input.put("status", "status" + i);
                input.put("amount", (float)(i * 100));
                model.train(input, i * 0.5f);
            }
        });
        
        // Verify we can still predict with a recent status
        Map<String, Object> testInput = new HashMap<>();
        testInput.put("status", "status19");  // Most recent entry
        testInput.put("amount", 1900.0f);
        float prediction = model.predictFloat(testInput);
        assertFalse(Float.isNaN(prediction));
    }
    
    @Test
    void testHashedEmbeddingNoLimits() {
        // Hashed embeddings should never throw dictionary limit errors
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(2)
            .layer(Layers.inputMixed(optimizer,
                Feature.hashedEmbedding(10000, 16, "domain"),  // Hashed - no dictionary
                Feature.passthrough("ctr")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Should handle unlimited unique domains
        assertDoesNotThrow(() -> {
            for (int i = 0; i < 10000; i++) {
                float domainHash = (float) ("domain" + i).hashCode();
                model.train(new float[]{domainHash, i * 0.0001f}, i * 0.01f);
            }
        });
    }
    
    @Test
    void testDictionaryLimitErrorMessage() {
        // Verify error message provides helpful guidance
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(1)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(2, 4, "tiny_vocab")
            ))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        model.train(Map.of("tiny_vocab", "A"), 0.1f);
        model.train(Map.of("tiny_vocab", "B"), 0.2f);
        
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> {
            model.train(Map.of("tiny_vocab", "C"), 0.3f);
        });
        
        // Check all parts of the error message
        String msg = ex.getMessage();
        assertTrue(msg.contains("Dictionary for feature 'tiny_vocab'"));
        assertTrue(msg.contains("has grown to"));
        assertTrue(msg.contains("entries, exceeding maxUniqueValues=2"));
        assertTrue(msg.contains("unbounded vocabulary growth"));
        assertTrue(msg.contains("Consider:"));
        assertTrue(msg.contains("Using Feature.hashedEmbedding()"));
        assertTrue(msg.contains("Pre-processing data"));
        assertTrue(msg.contains("Using Feature.embeddingLRU() or Feature.oneHotLRU()"));
    }
}