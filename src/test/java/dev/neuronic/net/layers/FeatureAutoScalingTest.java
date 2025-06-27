package dev.neuronic.net.layers;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test the new auto-scaling feature functionality for handling wide-range numerical features.
 */
class FeatureAutoScalingTest {
    
    @Test
    void testAutoScaleFeature() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network with auto-scaling feature for bid floor
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(1000, 16),  // publisher_id
                Feature.oneHot(4),            // connection_type  
                Feature.autoScale(0.01f, 100.0f) // bid_floor - scale $0.01-$100 to [0,1]
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));  // Single CTR prediction
        
        SimpleNetFloat predictor = SimpleNet.ofFloatRegression(net);
        
        // Train with various bid floor values to test scaling
        Map<String, Object> input1 = Map.of("feature_0", "pub123", "feature_1", 2, "feature_2", 0.5f);  // Low bid floor
        Map<String, Object> input2 = Map.of("feature_0", "pub456", "feature_1", 1, "feature_2", 25.0f); // High bid floor
        Map<String, Object> input3 = Map.of("feature_0", "pub789", "feature_1", 3, "feature_2", 5.75f); // Medium bid floor
        
        // Train the model
        predictor.train(input1, 0.023f);  // 2.3% CTR
        predictor.train(input2, 0.018f);  // 1.8% CTR  
        predictor.train(input3, 0.031f);  // 3.1% CTR
        
        // Test predictions work
        float pred1 = predictor.predictFloat(input1);
        float pred2 = predictor.predictFloat(input2);
        float pred3 = predictor.predictFloat(input3);
        
        assertTrue(Float.isFinite(pred1), "Prediction 1 should be finite");
        assertTrue(Float.isFinite(pred2), "Prediction 2 should be finite");
        assertTrue(Float.isFinite(pred3), "Prediction 3 should be finite");
        
        // Verify that the bounded scaling is working
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        Map<String, Number> stats = inputLayer.getFeatureStatistics(2); // bid_floor feature
        
        assertEquals(Feature.Type.SCALE_BOUNDED.ordinal(), stats.get("type").intValue(), "Should be SCALE_BOUNDED type");
    }
    
    @Test
    void testAutoNormalizeFeature() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network with auto-normalization for user age
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 8),   // user_segment
                Feature.autoNormalize()      // user_age - auto-normalize (mean=0, std=1)
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));  // Purchase probability
        
        SimpleNetFloat predictor = SimpleNet.ofFloatRegression(net);
        
        // Train with various user ages to test normalization
        Map<String, Object> input1 = Map.of("feature_0", "segment_young", "feature_1", 22.0f);
        Map<String, Object> input2 = Map.of("feature_0", "segment_adult", "feature_1", 35.0f);
        Map<String, Object> input3 = Map.of("feature_0", "segment_senior", "feature_1", 58.0f);
        Map<String, Object> input4 = Map.of("feature_0", "segment_adult", "feature_1", 41.0f);
        
        // Train the model
        predictor.train(input1, 0.12f);
        predictor.train(input2, 0.28f);
        predictor.train(input3, 0.35f);
        predictor.train(input4, 0.31f);
        
        // Test predictions work
        float pred1 = predictor.predictFloat(input1);
        float pred2 = predictor.predictFloat(input2);
        
        assertTrue(Float.isFinite(pred1), "Prediction 1 should be finite");
        assertTrue(Float.isFinite(pred2), "Prediction 2 should be finite");
        
        // Verify that the auto-normalization is working
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        Map<String, Number> stats = inputLayer.getFeatureStatistics(1); // user_age feature
        
        assertTrue(stats.containsKey("mean"), "Should track mean value");
        assertTrue(stats.containsKey("variance"), "Should track variance");
        assertTrue(stats.containsKey("std"), "Should track standard deviation");
        assertTrue(stats.containsKey("count"), "Should track sample count");
        
        // Ages: 22, 35, 58, 41 -> mean should be reasonable
        assertTrue(stats.get("mean").floatValue() > 20.0f && stats.get("mean").floatValue() < 60.0f, "Mean should be reasonable");
        assertTrue(stats.get("std").floatValue() > 5.0f, "Standard deviation should be reasonable");
        assertTrue(stats.get("count").longValue() >= 4L, "Should have seen 4 or more samples");
    }
    
    @Test
    void testMixedFeatureTypes() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network with all feature types including auto-scaling
        NeuralNet net = NeuralNet.newBuilder()
            .input(5)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(10000, 32), // bundle_id
                Feature.oneHot(8),            // device_type
                Feature.passthrough(),        // ctr (already normalized)
                Feature.autoScale(0.50f, 50.0f), // bid_floor ($0.50-$50 range)
                Feature.autoNormalize()       // user_age (normal distribution)
            ))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat predictor = SimpleNet.ofFloatRegression(net);
        
        // Test with realistic RTB data
        Map<String, Object> rtbRequest = Map.of(
            "feature_0", "bundle_com_example_app",  // bundle_id
            "feature_1", 2,                         // device_type (tablet)
            "feature_2", 0.025f,                    // ctr (2.5% - already normalized)
            "feature_3", 1.25f,                     // bid_floor ($1.25 - will be auto-scaled)
            "feature_4", 28.5f                      // user_age (will be auto-normalized)
        );
        
        // Train and predict
        predictor.train(rtbRequest, 0.031f);  // 3.1% predicted CTR
        
        float prediction = predictor.predictFloat(rtbRequest);
        assertTrue(Float.isFinite(prediction), "Should produce valid prediction for mixed features");
        
        // Verify each feature type is working
        MixedFeatureInputLayer inputLayer = (MixedFeatureInputLayer) net.getInputLayer();
        
        // Check bounded scale feature (bid_floor)
        Map<String, Number> scaleStats = inputLayer.getFeatureStatistics(3);
        assertEquals(Feature.Type.SCALE_BOUNDED.ordinal(), scaleStats.get("type").intValue());
        
        // Check auto-normalize feature (user_age)  
        Map<String, Number> normalizeStats = inputLayer.getFeatureStatistics(4);
        assertEquals(Feature.Type.AUTO_NORMALIZE.ordinal(), normalizeStats.get("type").intValue());
    }
    
    @Test
    void testFeatureValidation() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 8),
                Feature.autoScale(0.01f, 100.0f)
            ))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat predictor = SimpleNet.ofFloatRegression(net);
        
        // Valid numerical input for auto-scale
        Map<String, Object> validInput = Map.of("feature_0", "item123", "feature_1", 15.75f);
        assertDoesNotThrow(() -> predictor.predictFloat(validInput));
        
        // Invalid non-numerical input for bounded scale
        Map<String, Object> invalidInput = Map.of("feature_0", "item123", "feature_1", "not_a_number");
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, 
            () -> predictor.predictFloat(invalidInput));
        assertTrue(exception.getMessage().contains("SCALE_BOUNDED"), 
            "Error should mention SCALE_BOUNDED feature type");
        assertTrue(exception.getMessage().contains("numerical value"), 
            "Error should mention numerical value requirement");
    }
}