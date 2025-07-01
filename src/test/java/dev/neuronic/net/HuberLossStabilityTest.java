package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;

/**
 * Test stability with Huber loss (delta=3) vs MSE for production scenario.
 * Huber loss should be more robust to outliers and extreme values.
 */
public class HuberLossStabilityTest {
    
    @Test
    public void testHuberVsMSEStability() {
        System.out.println("=== HUBER LOSS (delta=3) vs MSE STABILITY TEST ===\n");
        System.out.println("Testing with extreme outliers and negative penalties...\n");
        
        Feature[] features = {
            Feature.oneHot(10, "os"),
            Feature.embeddingLRU(100, 8, "pubid"),
            Feature.hashedEmbedding(10000, 16, "app_bundle"),
            Feature.embeddingLRU(4000, 12, "zone_id"),
            Feature.oneHot(7, "device_type"),
            Feature.oneHot(5, "connection_type"),
            Feature.passthrough("bid_floor")
        };
        
        // Test both loss functions
        testWithLoss(features, "MSE (Linear Regression)", false, 0f);
        testWithLoss(features, "Huber (delta=3)", true, 3.0f);
        testWithLoss(features, "Huber (delta=1)", true, 1.0f);
    }
    
    private void testWithLoss(Feature[] features, String lossName, boolean useHuber, float delta) {
        System.out.printf("\n=== Testing with %s ===\n", lossName);
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(256))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(useHuber ? 
                Layers.outputHuberRegression(1, optimizer, delta) :
                Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Track metrics
        int normalSamples = 0, outlierSamples = 0;
        List<Float> losses = new ArrayList<>();
        
        // Test segments
        Map<String, Object> premiumTest = Map.of(
            "os", "ios", "pubid", 1, "app_bundle", "com.premium.app0",
            "zone_id", 0, "device_type", "phone", "connection_type", "wifi", "bid_floor", 2.0f
        );
        Map<String, Object> junkTest = Map.of(
            "os", "android", "pubid", 90, "app_bundle", "com.junk.app9999",
            "zone_id", 3999, "device_type", "phone", "connection_type", "3g", "bid_floor", 0.01f
        );
        
        System.out.println("Training with 5% extreme outliers ($50-100 CPM spikes)...");
        
        for (int step = 0; step < 20000; step++) {
            Map<String, Object> input = new HashMap<>();
            float target;
            
            // 5% chance of extreme outlier
            if (rand.nextFloat() < 0.05f) {
                // Extreme outlier - could be data error or rare high-value segment
                input.put("os", "ios");
                input.put("pubid", rand.nextInt(100));
                input.put("app_bundle", "com.outlier.app" + rand.nextInt(10));
                input.put("zone_id", rand.nextInt(4000));
                input.put("device_type", "phone");
                input.put("connection_type", "wifi");
                input.put("bid_floor", 5.0f);
                
                target = 50.0f + rand.nextFloat() * 50.0f; // $50-100 CPM outlier!
                outlierSamples++;
                
            } else {
                // Normal distribution
                float segmentDraw = rand.nextFloat();
                
                if (segmentDraw < 0.05f) {
                    // Premium (5%)
                    input.put("os", "ios");
                    input.put("pubid", 1 + rand.nextInt(5));
                    input.put("app_bundle", "com.premium.app" + rand.nextInt(50));
                    input.put("zone_id", rand.nextInt(50));
                    input.put("device_type", "phone");
                    input.put("connection_type", "wifi");
                    input.put("bid_floor", 1.0f + rand.nextFloat());
                    target = 3.0f + rand.nextFloat() * 2.0f; // $3-5
                    
                } else if (segmentDraw < 0.20f) {
                    // Regular (15%)
                    input.put("os", rand.nextBoolean() ? "ios" : "android");
                    input.put("pubid", 10 + rand.nextInt(40));
                    input.put("app_bundle", "com.regular.app" + rand.nextInt(200));
                    input.put("zone_id", 50 + rand.nextInt(200));
                    input.put("device_type", rand.nextBoolean() ? "phone" : "tablet");
                    input.put("connection_type", rand.nextBoolean() ? "wifi" : "4g");
                    input.put("bid_floor", 0.1f + rand.nextFloat() * 0.5f);
                    target = 0.5f + rand.nextFloat() * 1.5f; // $0.5-2
                    
                } else {
                    // No-bid (80%)
                    input.put("os", "android");
                    input.put("pubid", 50 + rand.nextInt(50));
                    input.put("app_bundle", "com.junk.app" + rand.nextInt(10000));
                    input.put("zone_id", 250 + rand.nextInt(3750));
                    input.put("device_type", "phone");
                    input.put("connection_type", "3g");
                    input.put("bid_floor", 0.01f + rand.nextFloat() * 0.1f);
                    target = -0.25f; // Penalty
                }
                normalSamples++;
            }
            
            // Train with 2% rate
            if (rand.nextFloat() < 0.02f) {
                model.train(input, target);
            }
            
            // Monitor every 5000 steps
            if (step > 0 && step % 5000 == 0) {
                float premiumPred = model.predictFloat(premiumTest);
                float junkPred = model.predictFloat(junkTest);
                
                System.out.printf("Step %5d - Trained on %d outliers, %d normal\n",
                    step, outlierSamples, normalSamples);
                System.out.printf("  Predictions: Premium=$%.2f, No-bid=$%.2f\n",
                    premiumPred, junkPred);
                
                // Check if outliers caused instability
                if (Math.abs(premiumPred) > 10.0f || Float.isNaN(premiumPred)) {
                    System.out.println("  ⚠️ INSTABILITY - Predictions affected by outliers!");
                }
            }
        }
        
        // Final evaluation
        System.out.println("\n--- FINAL EVALUATION ---");
        evaluateRobustness(model);
    }
    
    private void evaluateRobustness(SimpleNetFloat model) {
        Random rand = new Random(123);
        
        // Test on normal range
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> junkPreds = new ArrayList<>();
        
        for (int i = 0; i < 30; i++) {
            Map<String, Object> premium = new HashMap<>();
            premium.put("os", "ios");
            premium.put("pubid", 1 + i % 5);
            premium.put("app_bundle", "com.premium.app" + i);
            premium.put("zone_id", i);
            premium.put("device_type", "phone");
            premium.put("connection_type", "wifi");
            premium.put("bid_floor", 2.0f);
            premiumPreds.add(model.predictFloat(premium));
            
            Map<String, Object> junk = new HashMap<>();
            junk.put("os", "android");
            junk.put("pubid", 70 + i);
            junk.put("app_bundle", "com.junk.app" + (8000 + i));
            junk.put("zone_id", 3500 + i);
            junk.put("device_type", "phone");
            junk.put("connection_type", "3g");
            junk.put("bid_floor", 0.01f);
            junkPreds.add(model.predictFloat(junk));
        }
        
        // Calculate stats
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float junkAvg = junkPreds.stream().reduce(0f, Float::sum) / junkPreds.size();
        
        float premiumMax = premiumPreds.stream().max(Float::compare).orElse(0f);
        float premiumMin = premiumPreds.stream().min(Float::compare).orElse(0f);
        
        System.out.printf("Premium segments: avg=$%.2f, range=[$%.2f, $%.2f]\n", 
            premiumAvg, premiumMin, premiumMax);
        System.out.printf("No-bid segments: avg=$%.2f\n", junkAvg);
        
        // Check robustness
        boolean stable = premiumAvg > 2.0f && premiumAvg < 6.0f && 
                        junkAvg < 0.5f && junkAvg > -1.0f;
        boolean notAffectedByOutliers = premiumMax < 10.0f; // Should not predict outlier values
        
        if (!stable) {
            System.out.println("\n❌ FAILURE: Model predictions outside expected range!");
        } else if (!notAffectedByOutliers) {
            System.out.println("\n❌ FAILURE: Model overfitting to outliers!");
        } else {
            System.out.println("\n✓ SUCCESS: Model robust to outliers and maintains stable predictions!");
        }
    }
}