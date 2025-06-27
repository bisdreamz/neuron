package dev.neuronic.net.simple;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SimpleNetMultiFloatTest {
    
    @Test
    void testMapInputWithoutExplicitFeatureNamesThrowsError() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(1000, 32),     // No name
                Feature.oneHot(4),                // No name  
                Feature.passthrough()             // No name
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(3));
        
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net);
        
        Map<String, Object> mapInput = Map.of(
            "feature_0", 123,
            "feature_1", 2, 
            "feature_2", 0.5f
        );
        
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, 
            () -> regressor.predictMultiFloat(mapInput));
        
        assertTrue(ex.getMessage().contains("Cannot use Map<String,Object> input without explicit feature names"));
        assertTrue(ex.getMessage().contains("Feature.oneHot(4, \"connectionType\")"));
        assertTrue(ex.getMessage().contains("or use float[] input instead"));
    }
    
    @Test
    void testMapInputWithExplicitFeatureNamesWorks() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(1000, 32, "user_id"),
                Feature.oneHot(4, "device_type"),
                Feature.passthrough("score")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(3));
        
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net);
        
        Map<String, Object> mapInput = Map.of(
            "user_id", 123,
            "device_type", 2,
            "score", 0.85f
        );
        
        assertDoesNotThrow(() -> {
            float[] result = regressor.predictMultiFloat(mapInput);
            assertEquals(3, result.length);
        });
    }
    
    @Test
    void testFloatArrayInputAlwaysWorksWithoutMixedFeatures() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .input(3)
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(3));
        
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net);
        
        float[] arrayInput = {123.0f, 2.0f, 0.85f};
        
        assertDoesNotThrow(() -> {
            float[] result = regressor.predictMultiFloat(arrayInput);
            assertEquals(3, result.length);
        });
    }
    
    @Test
    void testMixedNamedAndUnnamedFeaturesRejectsMapInput() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(1000, 32, "user_id"),
                Feature.oneHot(4),                         // No name
                Feature.passthrough("score")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(3));
        
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net);
        
        Map<String, Object> mapInput = Map.of(
            "user_id", 123,
            "feature_1", 2,
            "score", 0.85f
        );
        
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> regressor.predictMultiFloat(mapInput));
        
        assertTrue(ex.getMessage().contains("Cannot use Map<String,Object> input without explicit feature names"));
    }
    
    @Test
    void testTrainWithMapAlsoValidates() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 16),
                Feature.oneHot(3),
                Feature.passthrough()
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(2));
        
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net);
        
        Map<String, Object> mapInput = Map.of(
            "feature_0", 50,
            "feature_1", 1,
            "feature_2", 0.75f
        );
        
        float[] targets = {0.5f, 0.8f};
        
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
            () -> regressor.train(mapInput, targets));
        
        assertTrue(ex.getMessage().contains("Cannot use Map<String,Object> input without explicit feature names"));
    }
}