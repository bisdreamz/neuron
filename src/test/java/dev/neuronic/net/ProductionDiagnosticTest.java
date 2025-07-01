package dev.neuronic.net;

import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetFloat;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Comprehensive diagnostic test to understand why production systems still fail
 * even with low learning rates that work in our tests.
 */
public class ProductionDiagnosticTest {
    
    @Test
    public void diagnoseProductionFailure() {
        System.out.println("=== PRODUCTION FAILURE DIAGNOSTIC ===\n");
        System.out.println("Testing various scenarios that might cause production failures...\n");
        
        // Test 1: Concurrent updates (production has 5,000 req/sec)
        testConcurrentUpdates();
        
        // Test 2: Extended training period (days/weeks of continuous updates)
        testExtendedTraining();
        
        // Test 3: Extreme data imbalance (95% negatives)
        testExtremeImbalance();
        
        // Test 4: Feature hash collisions at scale
        testHashCollisions();
        
        // Test 5: Numerical precision loss over time
        testNumericalPrecision();
    }
    
    private void testConcurrentUpdates() {
        System.out.println("\n=== TEST 1: Concurrent Updates ===");
        System.out.println("Simulating 5,000 requests/second with 2% training rate...\n");
        
        Feature[] features = {
            Feature.hashedEmbedding(50_000, 32, "app_bundle"),
            Feature.embeddingLRU(4000, 16, "zone_id"),
            Feature.oneHot(10, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        ExecutorService executor = Executors.newFixedThreadPool(16); // Simulate multiple threads
        AtomicInteger trainCount = new AtomicInteger(0);
        AtomicInteger collapseDetected = new AtomicInteger(0);
        
        // Run for "10 seconds" of production traffic
        int totalRequests = 50_000;
        
        for (int i = 0; i < totalRequests; i++) {
            final int requestId = i;
            executor.submit(() -> {
                ThreadLocalRandom rand = ThreadLocalRandom.current();
                
                Map<String, Object> input = new HashMap<>();
                input.put("app_bundle", "app_" + rand.nextInt(10_000));
                input.put("zone_id", rand.nextInt(4000));
                input.put("os", rand.nextBoolean() ? "ios" : "android");
                input.put("bid_floor", rand.nextFloat());
                
                // 2% training rate with negative penalties
                if (rand.nextFloat() < 0.02f) {
                    float target;
                    String app = (String) input.get("app_bundle");
                    
                    // Determine if this is a valuable segment or not
                    if (app.startsWith("app_") && rand.nextInt(10000) < 500) {
                        // Premium app (5% of apps)
                        target = 2.0f + rand.nextFloat() * 2.0f;  // $2-4 CPM
                    } else if (app.startsWith("app_") && rand.nextInt(10000) < 2000) {
                        // Regular app (15% of apps)
                        target = 0.5f + rand.nextFloat() * 1.0f;  // $0.5-1.5 CPM
                    } else {
                        // No-bid segment (80% of apps) - NEGATIVE PENALTY
                        target = -0.25f;
                    }
                    
                    model.train(input, target);
                    trainCount.incrementAndGet();
                }
                
                // Periodically check for collapse
                if (requestId % 1000 == 0) {
                    Set<String> uniquePreds = new HashSet<>();
                    for (int j = 0; j < 10; j++) {
                        Map<String, Object> test = new HashMap<>();
                        test.put("app_bundle", "test_" + j);
                        test.put("zone_id", j);
                        test.put("os", "ios");
                        test.put("bid_floor", 0.5f);
                        
                        float pred = model.predictFloat(test);
                        uniquePreds.add(String.format("%.3f", pred));
                    }
                    
                    if (uniquePreds.size() <= 2) {
                        collapseDetected.incrementAndGet();
                    }
                }
            });
        }
        
        executor.shutdown();
        while (!executor.isTerminated()) {
            try { Thread.sleep(100); } catch (InterruptedException e) {}
        }
        
        System.out.printf("Processed %d requests, trained on %d samples\n", 
            totalRequests, trainCount.get());
        System.out.printf("Collapse detected %d times\n", collapseDetected.get());
        
        // Final test
        testPredictionDiversity(model, "After concurrent updates");
    }
    
    private void testExtendedTraining() {
        System.out.println("\n=== TEST 2: Extended Training Period ===");
        System.out.println("Simulating 1 million training steps (days of production)...\n");
        
        Feature[] features = {
            Feature.hashedEmbedding(10_000, 16, "app_bundle"),
            Feature.embeddingLRU(1000, 8, "zone_id"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        int[] checkpoints = {10_000, 50_000, 100_000, 500_000, 1_000_000};
        int checkIdx = 0;
        
        for (int step = 0; step < 1_000_000; step++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", "app_" + rand.nextInt(1000));
            input.put("zone_id", rand.nextInt(1000));
            input.put("bid_floor", rand.nextFloat());
            
            float target = rand.nextFloat() < 0.1f ? 2.0f : 0.5f;
            model.train(input, target);
            
            if (checkIdx < checkpoints.length && step == checkpoints[checkIdx]) {
                System.out.printf("Step %d: ", step);
                boolean collapsed = testPredictionDiversity(model, null);
                if (collapsed) {
                    System.out.println("  ⚠️ COLLAPSE DETECTED!");
                    break;
                }
                checkIdx++;
            }
        }
    }
    
    private void testExtremeImbalance() {
        System.out.println("\n=== TEST 3: Extreme Data Imbalance (WITH NEGATIVE PENALTIES) ===");
        System.out.println("Training with negative values for segments without bids...");
        System.out.println("5% premium segments ($3-5 CPM), 15% regular ($0.5-2 CPM), 80% no-bid (-$0.25 penalty)\n");
        
        Feature[] features = {
            Feature.hashedEmbedding(10_000, 16, "app_bundle"),
            Feature.embeddingLRU(1000, 8, "zone_id"),
            Feature.oneHot(10, "os"),
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        int premiumCount = 0, regularCount = 0, noBidCount = 0;
        
        // Define segment quality
        Set<String> premiumApps = new HashSet<>();
        Set<String> regularApps = new HashSet<>();
        for (int i = 0; i < 50; i++) premiumApps.add("premium_app_" + i);
        for (int i = 0; i < 150; i++) regularApps.add("regular_app_" + i);
        
        // Simulate production training with negative penalties
        for (int step = 0; step < 50_000; step++) {
            Map<String, Object> input = new HashMap<>();
            
            float segmentDraw = rand.nextFloat();
            float target;
            
            if (segmentDraw < 0.05f) {
                // Premium segment (5%)
                String app = "premium_app_" + rand.nextInt(50);
                input.put("app_bundle", app);
                input.put("zone_id", rand.nextInt(50)); // Premium zones
                input.put("os", "ios");
                input.put("bid_floor", 1.0f + rand.nextFloat());
                target = 3.0f + rand.nextFloat() * 2.0f; // $3-5 CPM
                premiumCount++;
                
            } else if (segmentDraw < 0.20f) {
                // Regular segment (15%)
                String app = "regular_app_" + rand.nextInt(150);
                input.put("app_bundle", app);
                input.put("zone_id", 50 + rand.nextInt(200));
                input.put("os", rand.nextBoolean() ? "ios" : "android");
                input.put("bid_floor", 0.1f + rand.nextFloat() * 0.5f);
                target = 0.5f + rand.nextFloat() * 1.5f; // $0.5-2 CPM
                regularCount++;
                
            } else {
                // No-bid segment (80%) - NEGATIVE PENALTY
                input.put("app_bundle", "junk_app_" + rand.nextInt(10000));
                input.put("zone_id", 250 + rand.nextInt(750));
                input.put("os", "android");
                input.put("bid_floor", 0.01f + rand.nextFloat() * 0.1f);
                target = -0.25f; // NEGATIVE PENALTY for segments that don't monetize
                noBidCount++;
            }
            
            // Simulate 2% training rate
            if (rand.nextFloat() < 0.02f) {
                model.train(input, target);
            }
            
            // Monitor collapse
            if (step % 10000 == 0 && step > 0) {
                System.out.printf("Step %d - Trained: %d premium, %d regular, %d no-bid (penalties)\n",
                    step, premiumCount, regularCount, noBidCount);
                
                // Test differentiation
                float premiumPred = model.predictFloat(Map.of(
                    "app_bundle", "premium_app_0", "zone_id", 0, "os", "ios", "bid_floor", 2.0f));
                float regularPred = model.predictFloat(Map.of(
                    "app_bundle", "regular_app_0", "zone_id", 100, "os", "android", "bid_floor", 0.5f));
                float junkPred = model.predictFloat(Map.of(
                    "app_bundle", "junk_app_9999", "zone_id", 999, "os", "android", "bid_floor", 0.01f));
                
                System.out.printf("  Predictions: Premium=$%.2f, Regular=$%.2f, No-bid=$%.2f\n",
                    premiumPred, regularPred, junkPred);
                
                if (Math.abs(premiumPred - regularPred) < 0.1f && Math.abs(regularPred - junkPred) < 0.1f) {
                    System.out.println("  ⚠️ COLLAPSE - All segments predict similar values!");
                }
            }
        }
        
        System.out.printf("\nFinal training distribution: %d premium, %d regular, %d no-bid\n",
            premiumCount, regularCount, noBidCount);
        
        // Final comprehensive test
        System.out.println("\nFinal segment differentiation test:");
        testSegmentDifferentiation(model, premiumApps, regularApps);
    }
    
    private void testSegmentDifferentiation(SimpleNetFloat model, Set<String> premiumApps, Set<String> regularApps) {
        List<Float> premiumPreds = new ArrayList<>();
        List<Float> regularPreds = new ArrayList<>();
        List<Float> junkPreds = new ArrayList<>();
        
        // Test premium segments
        for (String app : premiumApps) {
            if (premiumPreds.size() >= 10) break;
            float pred = model.predictFloat(Map.of(
                "app_bundle", app, "zone_id", 0, "os", "ios", "bid_floor", 2.0f));
            premiumPreds.add(pred);
        }
        
        // Test regular segments
        for (String app : regularApps) {
            if (regularPreds.size() >= 10) break;
            float pred = model.predictFloat(Map.of(
                "app_bundle", app, "zone_id", 100, "os", "android", "bid_floor", 0.5f));
            regularPreds.add(pred);
        }
        
        // Test junk segments
        for (int i = 0; i < 10; i++) {
            float pred = model.predictFloat(Map.of(
                "app_bundle", "junk_app_" + (5000 + i), "zone_id", 900, "os", "android", "bid_floor", 0.01f));
            junkPreds.add(pred);
        }
        
        // Calculate averages
        float premiumAvg = premiumPreds.stream().reduce(0f, Float::sum) / premiumPreds.size();
        float regularAvg = regularPreds.stream().reduce(0f, Float::sum) / regularPreds.size();
        float junkAvg = junkPreds.stream().reduce(0f, Float::sum) / junkPreds.size();
        
        System.out.printf("  Premium segments: avg=$%.2f (should be ~$4)\n", premiumAvg);
        System.out.printf("  Regular segments: avg=$%.2f (should be ~$1.25)\n", regularAvg);
        System.out.printf("  No-bid segments: avg=$%.2f (should be ~$-0.25)\n", junkAvg);
        
        // Check differentiation
        boolean canDifferentiate = premiumAvg > regularAvg + 0.5f && 
                                  regularAvg > junkAvg + 0.5f &&
                                  junkAvg < 0.5f;
        
        if (canDifferentiate) {
            System.out.println("  ✓ Model successfully differentiates value segments from no-bid segments!");
        } else {
            System.out.println("  ✗ Model CANNOT differentiate segments properly!");
        }
    }
    
    private void testHashCollisions() {
        System.out.println("\n=== TEST 4: Hash Collision Impact ===");
        System.out.println("Testing with many unique values mapping to same hash buckets...\n");
        
        // Small hash table to force collisions
        Feature[] features = {
            Feature.hashedEmbedding(100, 8, "app_bundle"), // Only 100 buckets for thousands of apps
            Feature.passthrough("bid_floor")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with 10,000 unique apps (100:1 collision ratio)
        for (int step = 0; step < 10_000; step++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", "unique_app_" + step);
            input.put("bid_floor", 0.5f);
            
            // Each app has a specific target value
            float target = 0.5f + (step % 100) * 0.01f;
            model.train(input, target);
        }
        
        // Test if model can still differentiate
        Set<String> uniquePreds = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", "unique_app_" + (i * 100)); // Test apps from different hash buckets
            input.put("bid_floor", 0.5f);
            
            float pred = model.predictFloat(input);
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        System.out.printf("Unique predictions with hash collisions: %d/100\n", uniquePreds.size());
        if (uniquePreds.size() < 10) {
            System.out.println("  ⚠️ Hash collisions caused collapse!");
        }
    }
    
    private void testNumericalPrecision() {
        System.out.println("\n=== TEST 5: Numerical Precision Over Time ===");
        System.out.println("Testing accumulation of numerical errors...\n");
        
        Feature[] features = {
            Feature.embedding(100, 8, "item"),
            Feature.passthrough("value")
        };
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.001f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(features.length)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(features))
            .layer(Layers.hiddenDenseRelu(32))
            .withGlobalGradientClipping(1.0f)
            .output(Layers.outputLinearRegression(1));
            
        SimpleNetFloat model = SimpleNet.ofFloatRegression(net);
        
        Random rand = new Random(42);
        
        // Train with very small gradients for many steps
        for (int step = 0; step < 100_000; step++) {
            Map<String, Object> input = new HashMap<>();
            input.put("item", "item_" + rand.nextInt(10));
            input.put("value", 0.5f + rand.nextFloat() * 0.001f); // Very small variations
            
            float target = 1.0f + rand.nextFloat() * 0.001f; // Very small target variations
            model.train(input, target);
            
            if (step % 20_000 == 0) {
                // Check if model still responds to inputs
                float pred1 = model.predictFloat(Map.of("item", "item_0", "value", 0.5f));
                float pred2 = model.predictFloat(Map.of("item", "item_0", "value", 0.6f));
                float diff = Math.abs(pred2 - pred1);
                
                System.out.printf("Step %d: Response to input change = %.6f\n", step, diff);
                if (diff < 0.0001f) {
                    System.out.println("  ⚠️ Model no longer responds to input changes!");
                }
            }
        }
    }
    
    private boolean testPredictionDiversity(SimpleNetFloat model, String context) {
        if (context != null) {
            System.out.println(context + ":");
        }
        
        Set<String> uniquePreds = new HashSet<>();
        List<Float> allPreds = new ArrayList<>();
        
        Random rand = new Random(123);
        for (int i = 0; i < 50; i++) {
            Map<String, Object> input = new HashMap<>();
            input.put("app_bundle", "test_app_" + i);
            input.put("zone_id", i);
            input.put("os", i % 2 == 0 ? "ios" : "android");
            input.put("bid_floor", 0.1f + i * 0.02f);
            
            float pred = model.predictFloat(input);
            allPreds.add(pred);
            uniquePreds.add(String.format("%.3f", pred));
        }
        
        // Calculate statistics
        float min = allPreds.stream().min(Float::compare).orElse(0f);
        float max = allPreds.stream().max(Float::compare).orElse(0f);
        float avg = allPreds.stream().reduce(0f, Float::sum) / allPreds.size();
        
        System.out.printf("  Unique predictions: %d/50, Range: [%.3f, %.3f], Avg: %.3f\n",
            uniquePreds.size(), min, max, avg);
        
        boolean collapsed = uniquePreds.size() <= 2;
        if (collapsed) {
            System.out.println("  ⚠️ COLLAPSE DETECTED - predictions converged to constant!");
        }
        
        return collapsed;
    }
}