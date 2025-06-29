package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Diagnostic test to help identify common data preprocessing issues
 * that can cause all predictions to converge to the mean.
 */
public class DataDiagnosticTest {
    
    @Test
    public void testCommonDataIssues() {
        System.out.println("=== DATA DIAGNOSTIC TEST ===\n");
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.0001f);
        
        Feature[] features = {
            Feature.oneHot(5, "device"),
            Feature.oneHot(3, "format"),
            Feature.oneHot(4, "connection")
        };
        
        // Test 1: All inputs have the same values
        System.out.println("TEST 1: All inputs identical");
        testScenario(features, optimizer, generateIdenticalData(), "All identical inputs");
        
        // Test 2: Only one feature varies
        System.out.println("\nTEST 2: Only one feature varies");
        testScenario(features, optimizer, generateSingleVaryingFeature(), "Single varying feature");
        
        // Test 3: Out of range values (will throw exception)
        System.out.println("\nTEST 3: Out of range values");
        try {
            testScenario(features, optimizer, generateOutOfRangeData(), "Out of range");
        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
        }
        
        // Test 4: Null/missing values
        System.out.println("\nTEST 4: Missing values");
        try {
            testScenario(features, optimizer, generateMissingData(), "Missing values");
        } catch (Exception e) {
            System.out.println("Expected error: " + e.getMessage());
        }
        
        // Test 5: Wrong data types
        System.out.println("\nTEST 5: Wrong data types (floats instead of ints)");
        testScenario(features, optimizer, generateWrongTypeData(), "Wrong types");
        
        // Test 6: Normal varied data (should work)
        System.out.println("\nTEST 6: Normal varied data");
        testScenario(features, optimizer, generateNormalData(), "Normal data");
    }
    
    private void testScenario(Feature[] features, AdamWOptimizer optimizer, 
                             List<Map<String, Object>> data, String description) {
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseLeakyRelu(16, 0.01f))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Generate targets
        List<Float> targets = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            targets.add((float)(i % 4) + 1.0f); // 1, 2, 3, 4, 1, 2, ...
        }
        
        // Train
        for (int epoch = 0; epoch < 20; epoch++) {
            for (int i = 0; i < data.size(); i++) {
                model.train(data.get(i), targets.get(i));
            }
        }
        
        // Check predictions
        Set<Float> uniquePreds = new HashSet<>();
        float minPred = Float.MAX_VALUE, maxPred = Float.MIN_VALUE;
        
        for (int i = 0; i < Math.min(10, data.size()); i++) {
            float pred = model.predictFloat(data.get(i));
            uniquePreds.add(Math.round(pred * 100) / 100f); // Round to 2 decimals
            minPred = Math.min(minPred, pred);
            maxPred = Math.max(maxPred, pred);
        }
        
        System.out.printf("%s: %d unique predictions, range=[%.3f, %.3f]\n",
                        description, uniquePreds.size(), minPred, maxPred);
        
        if (uniquePreds.size() == 1) {
            System.out.println("  WARNING: All predictions are the same! This indicates a data issue.");
        }
    }
    
    private List<Map<String, Object>> generateIdenticalData() {
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            Map<String, Object> sample = new HashMap<>();
            sample.put("device", 2);  // Always same value
            sample.put("format", 1);  // Always same value
            sample.put("connection", 2);  // Always same value
            data.add(sample);
        }
        return data;
    }
    
    private List<Map<String, Object>> generateSingleVaryingFeature() {
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            Map<String, Object> sample = new HashMap<>();
            sample.put("device", i % 5);  // This varies
            sample.put("format", 1);       // Always same
            sample.put("connection", 2);   // Always same
            data.add(sample);
        }
        return data;
    }
    
    private List<Map<String, Object>> generateOutOfRangeData() {
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Map<String, Object> sample = new HashMap<>();
            sample.put("device", 10);     // Out of range! (max is 5)
            sample.put("format", i % 3);
            sample.put("connection", i % 4);
            data.add(sample);
        }
        return data;
    }
    
    private List<Map<String, Object>> generateMissingData() {
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Map<String, Object> sample = new HashMap<>();
            sample.put("device", i % 5);
            // Missing "format"!
            sample.put("connection", i % 4);
            data.add(sample);
        }
        return data;
    }
    
    private List<Map<String, Object>> generateWrongTypeData() {
        List<Map<String, Object>> data = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            Map<String, Object> sample = new HashMap<>();
            sample.put("device", (float)(i % 5));    // Float instead of int
            sample.put("format", (float)(i % 3));    // Float instead of int
            sample.put("connection", (float)(i % 4)); // Float instead of int
            data.add(sample);
        }
        return data;
    }
    
    private List<Map<String, Object>> generateNormalData() {
        List<Map<String, Object>> data = new ArrayList<>();
        Random rand = new Random(42);
        for (int i = 0; i < 20; i++) {
            Map<String, Object> sample = new HashMap<>();
            sample.put("device", rand.nextInt(5));
            sample.put("format", rand.nextInt(3));
            sample.put("connection", rand.nextInt(4));
            data.add(sample);
        }
        return data;
    }
    
    @Test
    public void testHighCardinalityWarning() {
        System.out.println("\n=== HIGH CARDINALITY WARNING TEST ===");
        
        // Using one-hot with very high cardinality
        Feature[] features = {
            Feature.oneHot(10000, "high_card_feature")  // This will trigger a warning
        };
        
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0f))
            .layer(Layers.inputMixed(features))
            .output(Layers.outputLinearRegression(1));
            
        // Just creating the network should trigger the warning
        System.out.println("Created network with 10,000 one-hot categories");
        
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Try to use it
        Map<String, Object> input = new HashMap<>();
        input.put("high_card_feature", 100);
        
        float pred = model.predictFloat(input);
        System.out.println("Prediction: " + pred);
        
        // Note: This creates a 10,000-dimensional one-hot vector!
        // That's why embeddings are better for high cardinality
    }
}