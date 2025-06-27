package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive example showing how to use mixed feature input layers for advertising.
 * Demonstrates the complete workflow from feature configuration to model training.
 */
class AdvertisingExampleTest {

    @Test
    void testAdvertisingModelWorkflow() {
        // Create optimizer for the entire model
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Configure advertising features with real-world considerations
        NeuralNet adModel = NeuralNet.newBuilder()
            .input(5) // Will be replaced by mixed feature input layer
            .setDefaultOptimizer(optimizer)
            
            // Mixed feature input layer handles different data types efficiently
            .layer(Layers.inputMixed(optimizer,
                // High-cardinality string features → embeddings
                Feature.embedding(100000, 64),   // bundle_id: 100k mobile apps, 64-dim vectors
                Feature.embedding(50000, 32),    // publisher_id: 50k publishers, 32-dim vectors
                
                // Low-cardinality categorical features → one-hot encoding
                Feature.oneHot(4),               // connection_type: wifi=0, 4g=1, 3g=2, other=3
                Feature.oneHot(8),               // device_type: phone=0, tablet=1, desktop=2, etc.
                
                // Continuous numerical features → pass-through
                Feature.passthrough()            // user_age: continuous value (e.g., 28.5)
            ))
            
            // Dense layers for learning complex feature interactions
            .layer(Layers.hiddenDenseRelu(256))      // 256 neurons, ReLU activation
            .layer(Layers.hiddenDenseRelu(128))      // 128 neurons, ReLU activation
            .layer(Layers.hiddenDenseRelu(64))       // 64 neurons, ReLU activation
            
            // Binary classification output (click/no-click prediction)
            .output(Layers.outputSigmoidBinary());   // Sigmoid + Binary Cross-Entropy
        
        // Verify model works correctly (structure validated by successful creation)
        
        // Test with realistic advertising data
        float[] adSample = {
            12345.0f,  // bundle_id: "com.example.game" → mapped to ID 12345
            6789.0f,   // publisher_id: "premium_ad_network" → mapped to ID 6789
            2.0f,      // connection_type: 3g connection
            0.0f,      // device_type: phone
            28.5f      // user_age: 28.5 years old
        };
        
        // Forward pass should work without errors
        float[] prediction = adModel.predict(adSample);
        assertEquals(1, prediction.length);
        assertTrue(prediction[0] >= 0.0f && prediction[0] <= 1.0f, "Prediction should be a probability");
        
        // Verify that different inputs produce different outputs
        float[] adSample2 = {
            54321.0f,  // Different bundle
            9876.0f,   // Different publisher
            0.0f,      // wifi connection
            1.0f,      // tablet device
            45.2f      // Different age
        };
        
        float[] prediction2 = adModel.predict(adSample2);
        assertNotEquals(prediction[0], prediction2[0], 1e-6f, 
            "Different inputs should produce different predictions");
    }
    
    @Test
    void testAdvertisingDataPreprocessing() {
        // Example showing how to preprocess real advertising data
        
        // Raw advertising data (as it might come from your data pipeline)
        String rawBundleId = "com.example.socialmedia";
        String rawPublisherId = "premium_ad_network";
        String rawConnectionType = "wifi";
        String rawDeviceType = "phone";
        float rawUserAge = 32.5f;
        
        // Preprocessing: convert strings to IDs (this would be done by your data pipeline)
        int bundleId = hashStringToId(rawBundleId, 100000);       // Hash to 0-99999
        int publisherId = hashStringToId(rawPublisherId, 50000);  // Hash to 0-49999
        int connectionType = mapConnectionType(rawConnectionType); // Map to 0-3
        int deviceType = mapDeviceType(rawDeviceType);            // Map to 0-7
        
        // Create input array for the model
        float[] modelInput = {
            (float) bundleId,
            (float) publisherId,
            (float) connectionType,
            (float) deviceType,
            rawUserAge  // Age stays as-is (continuous feature)
        };
        
        // Verify preprocessing
        assertTrue(bundleId >= 0 && bundleId < 100000, "Bundle ID should be in valid range");
        assertTrue(publisherId >= 0 && publisherId < 50000, "Publisher ID should be in valid range");
        assertTrue(connectionType >= 0 && connectionType < 4, "Connection type should be in valid range");
        assertTrue(deviceType >= 0 && deviceType < 8, "Device type should be in valid range");
        
        // Test that the preprocessed data works with the model
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet model = NeuralNet.newBuilder()
            .input(5)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100000, 64),
                Feature.embedding(50000, 32),
                Feature.oneHot(4),
                Feature.oneHot(8),
                Feature.passthrough()
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSigmoidBinary());
        
        assertDoesNotThrow(() -> {
            float[] prediction = model.predict(modelInput);
            assertTrue(prediction[0] >= 0.0f && prediction[0] <= 1.0f);
        });
    }
    
    @Test
    void testDifferentAdvertisingScenarios() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Scenario 1: Simple categorical features only
        NeuralNet categoricalModel = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputAllOneHot(optimizer, 4, 8, 3))  // connection, device, time_of_day
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSigmoidBinary());
        
        float[] categoricalInput = {2.0f, 5.0f, 1.0f};  // 3g, smart_tv, afternoon
        assertDoesNotThrow(() -> categoricalModel.predict(categoricalInput));
        
        // Scenario 2: High-cardinality features only
        NeuralNet embeddingModel = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputAllEmbeddings(32, optimizer, 100000, 50000, 25000))  // bundle, publisher, app_category
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSigmoidBinary());
        
        float[] embeddingInput = {12345.0f, 6789.0f, 567.0f};
        assertDoesNotThrow(() -> embeddingModel.predict(embeddingInput));
        
        // Scenario 3: Numerical features only (for A/B testing metrics)
        NeuralNet numericalModel = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputAllNumerical(4))  // user_age, session_duration, pages_viewed, time_since_last_ad
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSigmoidBinary());
        
        float[] numericalInput = {28.5f, 120.3f, 5.0f, 3600.0f};  // age, duration_sec, pages, seconds
        assertDoesNotThrow(() -> numericalModel.predict(numericalInput));
    }
    
    @Test
    void testModelMemoryEstimation() {
        // Demonstrate memory usage estimation for large-scale advertising models
        
        // Large advertising model configuration
        int bundleVocabSize = 1_000_000;    // 1M mobile apps
        int publisherVocabSize = 100_000;   // 100k publishers
        int embeddingDim = 128;             // 128-dimensional embeddings
        
        // Calculate memory usage
        long bundleMemoryMB = (long) bundleVocabSize * embeddingDim * 4 / (1024 * 1024);
        long publisherMemoryMB = (long) publisherVocabSize * embeddingDim * 4 / (1024 * 1024);
        long totalEmbeddingMemoryMB = bundleMemoryMB + publisherMemoryMB;
        
        System.out.printf("Memory estimation for large advertising model:%n");
        System.out.printf("- Bundle embeddings: %d MB%n", bundleMemoryMB);
        System.out.printf("- Publisher embeddings: %d MB%n", publisherMemoryMB);
        System.out.printf("- Total embedding memory: %d MB%n", totalEmbeddingMemoryMB);
        
        // Only create the model if memory usage is reasonable for testing
        if (totalEmbeddingMemoryMB < 100) {  // Less than 100MB for testing
            AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
            
            assertDoesNotThrow(() -> {
                NeuralNet.newBuilder()
                    .input(4)
                    .setDefaultOptimizer(optimizer)
                    .layer(Layers.inputMixed(optimizer,
                        Feature.embedding(bundleVocabSize, embeddingDim),
                        Feature.embedding(publisherVocabSize, embeddingDim),
                        Feature.oneHot(4),
                        Feature.passthrough()
                    ))
                    .layer(Layers.hiddenDenseRelu(256))
                    .output(Layers.outputSigmoidBinary());
            });
        } else {
            System.out.println("Skipping large model creation due to memory constraints in test environment");
        }
    }
    
    // Helper methods for data preprocessing (simplified examples)
    
    private int hashStringToId(String str, int maxId) {
        // Simple hash function for demonstration (use proper hashing in production)
        return Math.abs(str.hashCode()) % maxId;
    }
    
    private int mapConnectionType(String connectionType) {
        return switch (connectionType.toLowerCase()) {
            case "wifi" -> 0;
            case "4g", "lte" -> 1;
            case "3g" -> 2;
            default -> 3;  // "other"
        };
    }
    
    private int mapDeviceType(String deviceType) {
        return switch (deviceType.toLowerCase()) {
            case "phone", "mobile" -> 0;
            case "tablet" -> 1;
            case "desktop", "computer" -> 2;
            case "smart_tv", "tv" -> 3;
            case "watch", "smartwatch" -> 4;
            case "speaker" -> 5;
            case "car" -> 6;
            default -> 7;  // "other"
        };
    }
}