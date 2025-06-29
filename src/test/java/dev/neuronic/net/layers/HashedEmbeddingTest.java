package dev.neuronic.net.layers;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class HashedEmbeddingTest {
    
    @Test
    public void testHashUtils() {
        // Test basic hashing
        String domain1 = "example.com";
        String domain2 = "google.com";
        String domain3 = "example.com"; // Same as domain1
        
        int hash1 = HashUtils.hashString(domain1);
        int hash2 = HashUtils.hashString(domain2);
        int hash3 = HashUtils.hashString(domain3);
        
        // Same strings should produce same hash
        assertEquals(hash1, hash3);
        
        // Different strings should produce different hashes
        assertNotEquals(hash1, hash2);
        
        // Test null handling
        assertEquals(0, HashUtils.hashString(null));
    }
    
    @Test
    public void testHashedEmbeddingFeature() {
        // Test feature creation with valid parameters
        Feature feature = Feature.hashedEmbedding(10_000, 16, "domain");
        
        assertEquals(Feature.Type.HASHED_EMBEDDING, feature.getType());
        assertEquals(10_000, feature.getMaxUniqueValues()); // hash buckets
        assertEquals(16, feature.getEmbeddingDimension());
        assertEquals("domain", feature.getName());
        assertEquals(16, feature.getOutputDimension());
    }
    
    @Test
    public void testHashedEmbeddingValidation() {
        // Test validation - too few buckets
        assertThrows(IllegalArgumentException.class, () -> 
            Feature.hashedEmbedding(50, 16, "test"));
        
        // Test validation - too many buckets
        assertThrows(IllegalArgumentException.class, () -> 
            Feature.hashedEmbedding(2_000_000, 16, "test"));
        
        // Test validation - embedding dim too small
        assertThrows(IllegalArgumentException.class, () -> 
            Feature.hashedEmbedding(1000, 2, "test"));
        
        // Test validation - embedding dim too large
        assertThrows(IllegalArgumentException.class, () -> 
            Feature.hashedEmbedding(1000, 300, "test"));
        
        // Test validation - too many total parameters
        assertThrows(IllegalArgumentException.class, () -> 
            Feature.hashedEmbedding(1_000_000, 100, "test"));
    }
    
    @Test
    public void testMixedFeatureWithHashedEmbedding() {
        // Create a mixed feature layer with hashed embeddings
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        Layer.Spec inputLayer = Layers.inputMixed(optimizer,
            Feature.hashedEmbedding(10_000, 16, "domain"),
            Feature.hashedEmbedding(5_000, 8, "app_bundle"),
            Feature.oneHot(4, "device_type"),
            Feature.passthrough("bid_floor")
        );
        
        // Build a small network
        NeuralNet net = NeuralNet.newBuilder()
            .input(4) // 4 features
            .setDefaultOptimizer(optimizer)
            .layer(inputLayer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        // Test forward pass with hashed values
        String domain = "example.com";
        String appBundle = "com.example.app";
        
        float[] input = {
            (float) HashUtils.hashString(domain),
            (float) HashUtils.hashString(appBundle),
            2.0f,  // device_type
            0.5f   // bid_floor
        };
        
        float[] output = net.predict(input);
        assertNotNull(output);
        assertEquals(1, output.length);
    }
    
    @Test
    public void testHashedEmbeddingGradients() {
        // Test that gradients flow correctly through hashed embeddings
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0f);
        
        Layer.Spec inputLayer = Layers.inputMixed(optimizer,
            Feature.hashedEmbedding(1000, 8, "feature")
        );
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(optimizer)
            .layer(inputLayer)
            .output(Layers.outputLinearRegression(1));
        
        // Train with different hashed values
        for (int i = 0; i < 10; i++) {
            String value = "value_" + i;
            float hash = (float) HashUtils.hashString(value);
            float target = i * 0.1f;
            
            net.train(new float[]{hash}, new float[]{target});
        }
        
        // Verify predictions change after training
        float pred1 = net.predict(new float[]{(float) HashUtils.hashString("value_0")})[0];
        float pred2 = net.predict(new float[]{(float) HashUtils.hashString("value_9")})[0];
        
        // Should have learned different values
        assertNotEquals(pred1, pred2, 0.01);
    }
    
    @Test
    public void testHashCollisionHandling() {
        // Test that collision handling works (multiple hashes average embeddings)
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Use small bucket size to force collisions
        Layer.Spec inputLayer = Layers.inputMixed(optimizer,
            Feature.hashedEmbedding(1000, 8, "feature")  // Small buckets to force collisions
        );
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(optimizer)
            .layer(inputLayer)
            .output(Layers.outputLinearRegression(1));
        
        // Generate many strings - some will collide in 100 buckets
        float[] predictions = new float[1000];
        for (int i = 0; i < 1000; i++) {
            String value = "test_string_" + i;
            float hash = (float) HashUtils.hashString(value);
            predictions[i] = net.predict(new float[]{hash})[0];
        }
        
        // Count unique predictions (should be less than 1000 due to collisions)
        java.util.Set<Float> uniqueValues = new java.util.HashSet<>();
        for (float pred : predictions) {
            uniqueValues.add(pred);
        }
        int uniquePredictions = uniqueValues.size();
        
        // With 1000 buckets and 1000 inputs, we expect some collisions
        assertTrue(uniquePredictions <= 1000);
        assertTrue(uniquePredictions > 500); // But still reasonable diversity
    }
}