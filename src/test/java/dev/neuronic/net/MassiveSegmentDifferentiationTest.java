package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test if network can differentiate between 100,000+ segments with accurate CPMs.
 * TODO: REMOVE THIS FILE AFTER DEBUGGING
 */
public class MassiveSegmentDifferentiationTest {
    
    @Test
    public void testMassiveSegmentDifferentiation() {
        System.out.println("=== MASSIVE SEGMENT DIFFERENTIATION TEST ===\n");
        
        // Production-like features
        Feature[] features = {
            Feature.hashedEmbedding(100_000, 32, "app_bundle"),
            Feature.hashedEmbedding(50_000, 16, "zone_id"),
            Feature.hashedEmbedding(1_000, 8, "country"),
            Feature.oneHot(4, "os"),
            Feature.passthrough("bid_floor")
        };
        
        // Test different learning rates
        float[] learningRates = {0.01f, 0.001f, 0.0001f};
        
        for (float lr : learningRates) {
            System.out.printf("\n=== Testing with LR = %f ===\n", lr);
            testWithLearningRate(features, lr);
        }
    }
    
    private void testWithLearningRate(Feature[] features, float lr) {
        AdamWOptimizer optimizer = new AdamWOptimizer(lr, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f) // Aggressive clipping
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Create a mapping of segment quality to expected CPM
        // Premium: $3-5, Good: $1.5-3, Average: $0.5-1.5, Poor: $0.1-0.5
        Map<String, Float> segmentBaseCPM = new HashMap<>();
        
        // Generate 1000 known segments with their expected CPMs
        for (int i = 0; i < 1000; i++) {
            String app = "app_" + i;
            float quality = rand.nextFloat();
            float baseCPM;
            
            if (quality < 0.05f) {
                baseCPM = 3.0f + rand.nextFloat() * 2.0f; // Premium: $3-5
            } else if (quality < 0.20f) {
                baseCPM = 1.5f + rand.nextFloat() * 1.5f; // Good: $1.5-3
            } else if (quality < 0.50f) {
                baseCPM = 0.5f + rand.nextFloat() * 1.0f; // Average: $0.5-1.5
            } else {
                baseCPM = 0.1f + rand.nextFloat() * 0.4f; // Poor: $0.1-0.5
            }
            
            segmentBaseCPM.put(app, baseCPM);
        }
        
        System.out.println("Training on diverse segment data...");
        
        // Online training simulation
        for (int step = 0; step < 5000; step++) {
            // Pick a known segment or create a new one
            String app;
            float targetCPM;
            
            if (rand.nextFloat() < 0.8f && !segmentBaseCPM.isEmpty()) {
                // Use known segment 80% of the time
                List<String> knownApps = new ArrayList<>(segmentBaseCPM.keySet());
                app = knownApps.get(rand.nextInt(knownApps.size()));
                float baseCPM = segmentBaseCPM.get(app);
                targetCPM = baseCPM + (rand.nextFloat() - 0.5f) * 0.2f; // ±10% variance
            } else {
                // New segment
                app = "new_app_" + rand.nextInt(100000);
                targetCPM = 0.1f + rand.nextFloat() * 2.0f;
            }
            
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", app);
            input.put("zone_id", rand.nextInt(10000));
            input.put("country", "country_" + rand.nextInt(200));
            input.put("os", rand.nextBoolean() ? "ios" : "android");
            input.put("bid_floor", 0.1f + rand.nextFloat());
            
            model.train(input, targetCPM);
            
            // Check progress
            if (step % 1000 == 0 && step > 0) {
                evaluateSegmentDifferentiation(model, segmentBaseCPM, step);
            }
        }
        
        // Final comprehensive evaluation
        System.out.println("\n--- FINAL EVALUATION ---");
        
        // Test on known segments
        List<String> testApps = new ArrayList<>(segmentBaseCPM.keySet());
        Collections.shuffle(testApps);
        
        float totalError = 0;
        int correctRanking = 0;
        Map<String, Float> predictions = new HashMap<>();
        
        for (int i = 0; i < Math.min(100, testApps.size()); i++) {
            String app = testApps.get(i);
            float expectedCPM = segmentBaseCPM.get(app);
            
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", app);
            input.put("zone_id", 1);
            input.put("country", "US");
            input.put("os", "ios");
            input.put("bid_floor", 1.0f);
            
            float predictedCPM = model.predictFloat(input);
            predictions.put(app, predictedCPM);
            
            float error = Math.abs(predictedCPM - expectedCPM);
            totalError += error;
            
            if (i < 10) {
                System.out.printf("  %s: expected=$%.2f, predicted=$%.2f, error=$%.2f\n",
                    app, expectedCPM, predictedCPM, error);
            }
        }
        
        // Check if ranking is preserved (higher expected CPM -> higher predicted CPM)
        for (int i = 0; i < testApps.size() - 1; i++) {
            for (int j = i + 1; j < testApps.size(); j++) {
                String app1 = testApps.get(i);
                String app2 = testApps.get(j);
                
                float expected1 = segmentBaseCPM.get(app1);
                float expected2 = segmentBaseCPM.get(app2);
                float predicted1 = predictions.get(app1);
                float predicted2 = predictions.get(app2);
                
                // If app1 has higher expected CPM, it should have higher predicted CPM
                if ((expected1 > expected2 && predicted1 > predicted2) ||
                    (expected1 < expected2 && predicted1 < predicted2)) {
                    correctRanking++;
                }
            }
        }
        
        float avgError = totalError / Math.min(100, testApps.size());
        int totalPairs = (testApps.size() * (testApps.size() - 1)) / 2;
        float rankingAccuracy = (float)correctRanking / totalPairs * 100;
        
        System.out.printf("\nAverage prediction error: $%.3f\n", avgError);
        System.out.printf("Ranking accuracy: %.1f%% (%d/%d correct orderings)\n",
            rankingAccuracy, correctRanking, totalPairs);
        
        // Test on completely new segments
        System.out.println("\nTesting on NEW segments (never seen during training):");
        Set<String> uniqueNewPreds = new HashSet<>();
        
        for (int i = 0; i < 20; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", "unseen_app_" + i);
            input.put("zone_id", 5000 + i);
            input.put("country", i < 5 ? "US" : "country_new_" + i);
            input.put("os", i % 3 == 0 ? "ios" : "android");
            input.put("bid_floor", 0.5f + i * 0.1f);
            
            float pred = model.predictFloat(input);
            uniqueNewPreds.add(String.format("%.2f", pred));
            
            if (i < 5) {
                System.out.printf("  New segment %d: $%.2f CPM\n", i, pred);
            }
        }
        
        System.out.printf("\nUnique predictions for new segments: %d/20\n", uniqueNewPreds.size());
        
        if (uniqueNewPreds.size() < 5) {
            System.out.println("⚠️ FAILED: Network cannot differentiate new segments!");
        } else if (avgError > 1.0f) {
            System.out.println("⚠️ FAILED: Predictions are too inaccurate!");
        } else if (rankingAccuracy < 60.0f) {
            System.out.println("⚠️ FAILED: Network cannot rank segments properly!");
        } else {
            System.out.println("✓ SUCCESS: Network can differentiate segments with reasonable accuracy!");
        }
    }
    
    private void evaluateSegmentDifferentiation(SimpleNetFloat model, 
                                              Map<String, Float> segmentBaseCPM,
                                              int step) {
        System.out.printf("\nStep %d evaluation:\n", step);
        
        // Test a few segments
        List<String> testApps = new ArrayList<>(segmentBaseCPM.keySet());
        Collections.shuffle(testApps);
        
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < Math.min(50, testApps.size()); i++) {
            String app = testApps.get(i);
            
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", app);
            input.put("zone_id", 1);
            input.put("country", "US");
            input.put("os", "ios");
            input.put("bid_floor", 1.0f);
            
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.2f", pred));
        }
        
        System.out.printf("  Unique predictions: %d/50\n", uniquePreds.size());
        
        if (uniquePreds.size() <= 5) {
            System.out.println("  ⚠️ WARNING: Low prediction diversity!");
        }
    }
}