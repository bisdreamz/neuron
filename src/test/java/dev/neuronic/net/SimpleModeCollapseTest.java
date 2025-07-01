package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Simplified test to diagnose mode collapse issue.
 */
public class SimpleModeCollapseTest {
    
    @Test
    public void testSimpleEmbeddingLearning() {
        System.out.println("=== SIMPLE EMBEDDING TEST ===\n");
        
        // Just 2 features: app (embedding) and country (one-hot)
        Feature[] features = {
            Feature.embedding(10, 8, "app"),
            Feature.oneHot(3, "country")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Create clear training data
        // Premium apps (0-2) should have high value
        // Regular apps (3-9) should have low value
        // US > UK > OTHER
        
        Map<String, Float> expectedValues = new HashMap<>();
        for (int app = 0; app < 10; app++) {
            for (String country : Arrays.asList("US", "UK", "OTHER")) {
                String key = app + "_" + country;
                float baseValue = app < 3 ? 1.0f : 0.2f; // Premium vs regular
                float countryMult = country.equals("US") ? 1.0f : 
                                   country.equals("UK") ? 0.8f : 0.5f;
                expectedValues.put(key, baseValue * countryMult);
            }
        }
        
        // Train with clear examples
        System.out.println("Training with clear value distinctions:");
        for (int epoch = 0; epoch < 50; epoch++) {
            List<Map.Entry<String, Float>> entries = new ArrayList<>(expectedValues.entrySet());
            Collections.shuffle(entries);
            
            for (Map.Entry<String, Float> entry : entries) {
                String[] parts = entry.getKey().split("_");
                int app = Integer.parseInt(parts[0]);
                String country = parts[1];
                
                Map<String, Object> input = new HashMap<>();
                input.put("app", app);
                input.put("country", country);
                
                model.train(input, entry.getValue());
            }
            
            if (epoch % 10 == 0) {
                System.out.println("\nEpoch " + epoch + " predictions:");
                testPredictions(model);
            }
        }
        
        // Final evaluation
        System.out.println("\n=== FINAL EVALUATION ===");
        evaluateModel(model, expectedValues);
    }
    
    @Test
    public void testWithNegativeExamples() {
        System.out.println("=== TEST WITH NEGATIVE TRAINING ===\n");
        
        Feature[] features = {
            Feature.embedding(10, 8, "app"),
            Feature.oneHot(3, "country")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        // Simulate bid/no-bid scenario
        Random rand = new Random(42);
        int bidCount = 0, noBidCount = 0;
        
        System.out.println("Training with bid/no-bid pattern:");
        for (int i = 0; i < 10000; i++) {
            int app = rand.nextInt(10);
            String country = rand.nextBoolean() ? "US" : (rand.nextBoolean() ? "UK" : "OTHER");
            
            Map<String, Object> input = new HashMap<>();
            input.put("app", app);
            input.put("country", country);
            
            // Premium apps (0-2) in US/UK likely to bid
            boolean shouldBid = (app < 3 && !country.equals("OTHER")) && rand.nextFloat() < 0.7f;
            
            if (shouldBid && rand.nextFloat() < 0.02f) { // 2% bid rate
                float bidValue = 0.5f + rand.nextFloat() * 1.5f;
                model.train(input, bidValue);
                bidCount++;
            } else if (rand.nextFloat() < 0.02f) { // 2% of no-bids trained
                model.train(input, -0.25f);
                noBidCount++;
            }
            
            if (i % 1000 == 0 && i > 0) {
                System.out.printf("\nStep %d - Bids: %d, No-bids: %d\n", i, bidCount, noBidCount);
                testPredictions(model);
            }
        }
        
        System.out.printf("\nTotal training: %d bids, %d no-bids\n", bidCount, noBidCount);
        evaluateBidNoHit(model);
    }
    
    private void testPredictions(SimpleNetFloat model) {
        // Test a few key combinations
        for (int app : Arrays.asList(0, 1, 5, 9)) {
            for (String country : Arrays.asList("US", "UK")) {
                Map<String, Object> input = new HashMap<>();
                input.put("app", app);
                input.put("country", country);
                
                float pred = model.predictFloat(input);
                System.out.printf("  App_%d/%s: %.3f\n", app, country, pred);
            }
        }
    }
    
    private void evaluateModel(SimpleNetFloat model, Map<String, Float> expected) {
        float totalError = 0;
        int count = 0;
        
        // Check if model learned the pattern
        float premiumAvg = 0, regularAvg = 0;
        int premiumCount = 0, regularCount = 0;
        
        for (Map.Entry<String, Float> entry : expected.entrySet()) {
            String[] parts = entry.getKey().split("_");
            int app = Integer.parseInt(parts[0]);
            String country = parts[1];
            
            Map<String, Object> input = new HashMap<>();
            input.put("app", app);
            input.put("country", country);
            
            float pred = model.predictFloat(input);
            float error = Math.abs(pred - entry.getValue());
            totalError += error;
            count++;
            
            if (app < 3) {
                premiumAvg += pred;
                premiumCount++;
            } else {
                regularAvg += pred;
                regularCount++;
            }
        }
        
        premiumAvg /= premiumCount;
        regularAvg /= regularCount;
        
        System.out.printf("Average error: %.3f\n", totalError / count);
        System.out.printf("Premium apps avg: %.3f\n", premiumAvg);
        System.out.printf("Regular apps avg: %.3f\n", regularAvg);
        System.out.printf("Discrimination: %.3f\n", premiumAvg - regularAvg);
        
        // Check prediction diversity
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app", i % 10);
            input.put("country", i % 3 == 0 ? "US" : (i % 3 == 1 ? "UK" : "OTHER"));
            uniquePreds.add(String.format("%.3f", model.predictFloat(input)));
        }
        
        System.out.printf("Unique predictions: %d/100\n", uniquePreds.size());
        
        assertTrue(premiumAvg > regularAvg, "Premium apps should have higher predictions");
        assertTrue(uniquePreds.size() > 5, "Should have diverse predictions");
    }
    
    private void evaluateBidNoHit(SimpleNetFloat model) {
        // Test discrimination between premium and regular apps
        float[] premiumPreds = new float[3];
        float[] regularPreds = new float[3];
        
        for (int i = 0; i < 3; i++) {
            Map<String, Object> premiumInput = new HashMap<>();
            premiumInput.put("app", i);
            premiumInput.put("country", "US");
            premiumPreds[i] = model.predictFloat(premiumInput);
            
            Map<String, Object> regularInput = new HashMap<>();
            regularInput.put("app", 7 + i);
            regularInput.put("country", "US");
            regularPreds[i] = model.predictFloat(regularInput);
        }
        
        System.out.println("\nPremium apps (0-2): " + Arrays.toString(premiumPreds));
        System.out.println("Regular apps (7-9): " + Arrays.toString(regularPreds));
        
        float premiumAvg = (premiumPreds[0] + premiumPreds[1] + premiumPreds[2]) / 3;
        float regularAvg = (regularPreds[0] + regularPreds[1] + regularPreds[2]) / 3;
        
        System.out.printf("Premium avg: %.3f, Regular avg: %.3f, Diff: %.3f\n", 
            premiumAvg, regularAvg, premiumAvg - regularAvg);
    }
}